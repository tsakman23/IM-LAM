"""Tests for the IM-LAM interaction bottleneck module (Phase 2).

Run:  conda_env/bin/python tests/test_interaction_module.py

Built up task by task. Covers so far:
  - the token embeddings: a spatial positional embedding (one vector per bottleneck
    cell) and a temporal embedding marking current vs predicted-next state (no
    per-entity embedding - separation comes from the mask biases + separate weights);
  - mask pooling: average-pooling a binary current-frame mask to bottleneck
    resolution, yielding the (B, N) fractional occupancy W that feeds beta_h * W_j.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ifo.common.nets.interaction import (
    InteractionEmbeddings,
    InteractionModule,
    pool_mask_occupancy,
)


class InteractionEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dim = 16
        self.num_tokens = 9
        self.emb = InteractionEmbeddings(self.dim, num_tokens=self.num_tokens)

    def test_embedding_shapes(self):
        self.assertEqual(tuple(self.emb.spatial.shape), (self.num_tokens, self.dim))
        self.assertEqual(tuple(self.emb.temporal_cur.shape), (self.dim,))
        self.assertEqual(tuple(self.emb.temporal_pred.shape), (self.dim,))

    def test_tag_without_temporal_adds_only_spatial(self):
        x = torch.randn(2, self.num_tokens, self.dim)
        out = self.emb.tag(x)
        self.assertTrue(torch.allclose(out, x + self.emb.spatial))

    def test_tag_adds_the_requested_temporal_vector(self):
        x = torch.randn(2, self.num_tokens, self.dim)
        cur = self.emb.tag(x, temporal="cur")
        pred = self.emb.tag(x, temporal="pred")
        self.assertTrue(torch.allclose(cur, x + self.emb.spatial + self.emb.temporal_cur))
        self.assertTrue(torch.allclose(pred, x + self.emb.spatial + self.emb.temporal_pred))

    def test_current_and_predicted_tags_differ_at_init(self):
        # Load-bearing: F_A is a residual update, so A_t and \hat{A}_{t+1} are nearly
        # identical early in training. The temporal tag must break that degeneracy
        # from step 0, not only after learning.
        x = torch.randn(2, self.num_tokens, self.dim)
        cur = self.emb.tag(x, temporal="cur")
        pred = self.emb.tag(x, temporal="pred")
        self.assertFalse(torch.allclose(cur, pred))

    def test_spatial_embedding_is_position_dependent(self):
        # Different bottleneck cells must get different vectors, otherwise the
        # position-aligned write-back has no location information to work with.
        rows = self.emb.spatial.detach()
        self.assertFalse(torch.allclose(rows[0], rows[1]))

    def test_tag_broadcasts_over_batch(self):
        for b in (1, 5):
            x = torch.randn(b, self.num_tokens, self.dim)
            self.assertEqual(tuple(self.emb.tag(x, temporal="cur").shape), (b, self.num_tokens, self.dim))

    def test_no_entity_embedding_exists(self):
        # Deliberate design choice: entity identity is carried by the mask bias and
        # by separate attention weight sets, not by a learned per-entity tag.
        self.assertEqual(
            set(name for name, _ in self.emb.named_parameters()),
            {"spatial", "temporal_cur", "temporal_pred"},
        )

    def test_invalid_temporal_raises(self):
        x = torch.randn(2, self.num_tokens, self.dim)
        with self.assertRaises(ValueError):
            self.emb.tag(x, temporal="future")

    def test_embeddings_receive_gradient(self):
        x = torch.randn(2, self.num_tokens, self.dim)
        (self.emb.tag(x, temporal="cur").sum() + self.emb.tag(x, temporal="pred").sum()).backward()
        self.assertGreater(self.emb.spatial.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.emb.temporal_cur.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.emb.temporal_pred.grad.abs().sum().item(), 0.0)


class PoolMaskOccupancyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shape_dmw_and_finer_grid(self):
        mask = torch.zeros(2, 1, 128, 128)
        self.assertEqual(tuple(pool_mask_occupancy(mask, 16).shape), (2, 256))
        self.assertEqual(tuple(pool_mask_occupancy(mask, 32).shape), (2, 1024))

    def test_full_and_empty_masks(self):
        self.assertTrue(torch.all(pool_mask_occupancy(torch.ones(2, 1, 4, 4), 2) == 1.0))
        self.assertTrue(torch.all(pool_mask_occupancy(torch.zeros(2, 1, 4, 4), 2) == 0.0))

    def test_single_cell_and_row_major_order(self):
        # 4x4 input, 2x2 grid, factor 2. Cells flatten row-major: (0,0)->0, (0,1)->1,
        # (1,0)->2, (1,1)->3. This ordering must match the row-major flatten of B_t.
        def cell(rows, cols):
            m = torch.zeros(1, 1, 4, 4)
            m[:, :, rows, cols] = 1.0
            return pool_mask_occupancy(m, 2)[0]

        self.assertTrue(torch.allclose(cell(slice(0, 2), slice(0, 2)), torch.tensor([1.0, 0, 0, 0])))
        self.assertTrue(torch.allclose(cell(slice(0, 2), slice(2, 4)), torch.tensor([0, 1.0, 0, 0])))
        self.assertTrue(torch.allclose(cell(slice(2, 4), slice(0, 2)), torch.tensor([0, 0, 1.0, 0])))
        self.assertTrue(torch.allclose(cell(slice(2, 4), slice(2, 4)), torch.tensor([0, 0, 0, 1.0])))

    def test_fractional_occupancy_is_true_area_fraction(self):
        # Half of one cell's 2x2=4-pixel patch set to 1 -> occupancy 0.5 (avg-pool the
        # binary mask, not a resized/thresholded image).
        m = torch.zeros(1, 1, 4, 4)
        m[:, :, 0, 0] = 1.0
        m[:, :, 0, 1] = 1.0  # 2 of the 4 pixels in cell (0,0)
        self.assertAlmostEqual(pool_mask_occupancy(m, 2)[0, 0].item(), 0.5, places=6)

    def test_pool_factor_is_derived_not_hardcoded(self):
        # 8x8 input, 2x2 grid -> factor 4 (not 8). A full 4x4 patch at cell (0,0) -> 1.0.
        m = torch.zeros(1, 1, 8, 8)
        m[:, :, 0:4, 0:4] = 1.0
        occ = pool_mask_occupancy(m, 2)
        self.assertEqual(tuple(occ.shape), (1, 4))
        self.assertAlmostEqual(occ[0, 0].item(), 1.0, places=6)

    def test_values_in_unit_interval(self):
        occ = pool_mask_occupancy((torch.rand(2, 1, 8, 8) > 0.5).float(), 4)
        self.assertTrue(torch.all(occ >= 0.0) and torch.all(occ <= 1.0))

    def test_accepts_bhw_and_bchw(self):
        m = (torch.rand(2, 4, 4) > 0.5).float()
        self.assertTrue(torch.allclose(pool_mask_occupancy(m, 2), pool_mask_occupancy(m.unsqueeze(1), 2)))

    def test_non_divisible_grid_raises(self):
        with self.assertRaises((AssertionError, ValueError)):
            pool_mask_occupancy(torch.zeros(1, 1, 5, 5), 2)

    def test_rejects_non_binary_mask(self):
        # A raw {0,255} mask bypassing ProcessMask would drive occupancy > 1, making beta*W a hard
        # ~510 gate - exactly the regime the soft bias avoids.
        with self.assertRaises(ValueError):
            pool_mask_occupancy(torch.full((1, 1, 4, 4), 255.0), 2)


class EntityExtractionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dim, self.heads, self.n, self.b = 32, 4, 9, 2
        self.module = InteractionModule(self.dim, num_tokens=self.n, num_heads=self.heads, num_res_blocks=2)
        self.b_t = torch.randn(self.b, self.n, self.dim)
        self.w_a = torch.rand(self.b, self.n)
        self.w_o = torch.rand(self.b, self.n)

    def test_extract_shapes(self):
        a_t, o_t = self.module.extract(self.b_t, self.w_a, self.w_o)
        self.assertEqual(tuple(a_t.shape), (self.b, self.n, self.dim))
        self.assertEqual(tuple(o_t.shape), (self.b, self.n, self.dim))

    def test_heads_are_separately_parameterized(self):
        self.assertIsNot(self.module.msa_agent, self.module.msa_object)
        agent_ids = {id(p) for p in self.module.msa_agent.parameters()}
        object_ids = {id(p) for p in self.module.msa_object.parameters()}
        self.assertTrue(agent_ids.isdisjoint(object_ids))

    def test_independent_heads_give_different_readouts_for_same_mask(self):
        # Same occupancy fed to both -> outputs still differ, because the two heads
        # have independent weights (this is what separates agent from object).
        a_t, o_t = self.module.extract(self.b_t, self.w_a, self.w_a)
        self.assertFalse(torch.allclose(a_t, o_t))

    def test_mask_routes_to_matching_head(self):
        # Make the two heads identical functions; then the only thing distinguishing
        # a_t from o_t is which occupancy each receives. Swapping the masks must swap
        # the outputs - proving w_agent drives a_t and w_object drives o_t (not crossed).
        self.module.msa_object.load_state_dict(self.module.msa_agent.state_dict())
        a_t, o_t = self.module.extract(self.b_t, self.w_a, self.w_o)
        a_t2, o_t2 = self.module.extract(self.b_t, self.w_o, self.w_a)
        self.assertTrue(torch.allclose(a_t2, o_t, atol=1e-6))
        self.assertTrue(torch.allclose(o_t2, a_t, atol=1e-6))

    def test_gradient_flows_to_embeddings_and_both_heads(self):
        a_t, o_t = self.module.extract(self.b_t, self.w_a, self.w_o)
        (a_t.sum() + o_t.sum()).backward()
        self.assertGreater(self.module.embeddings.spatial.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.module.embeddings.temporal_cur.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.module.msa_agent.beta.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.module.msa_object.beta.grad.abs().sum().item(), 0.0)

    def test_extract_runs_under_bf16_autocast(self):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            a_t, o_t = self.module.extract(self.b_t, self.w_a, self.w_o)
        self.assertTrue(torch.isfinite(a_t).all() and torch.isfinite(o_t).all())


class AgentDynamicsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dim, self.heads, self.n, self.b, self.code = 32, 4, 9, 2, 8
        self.module = InteractionModule(
            self.dim, num_tokens=self.n, num_heads=self.heads, code_dim=self.code, num_res_blocks=2
        )
        self.a_t = torch.randn(self.b, self.n, self.dim)
        self.z = torch.randn(self.b, self.code)

    def test_shape(self):
        a_hat = self.module.agent_dynamics(self.a_t, self.z)
        self.assertEqual(tuple(a_hat.shape), (self.b, self.n, self.dim))

    def test_latent_action_influences_output(self):
        # z_t must actually drive the agent transition, not be ignored.
        a1 = self.module.agent_dynamics(self.a_t, self.z)
        a2 = self.module.agent_dynamics(self.a_t, torch.randn(self.b, self.code))
        self.assertFalse(torch.allclose(a1, a2))

    def test_agent_features_influence_output(self):
        a1 = self.module.agent_dynamics(self.a_t, self.z)
        a2 = self.module.agent_dynamics(torch.randn(self.b, self.n, self.dim), self.z)
        self.assertFalse(torch.allclose(a1, a2))

    def test_act_proj_projects_to_full_bottleneck(self):
        # Reuses MaskLAM's mechanism: Linear(code_dim, dim * num_tokens).
        self.assertEqual(tuple(self.module.f_a.act_proj.weight.shape), (self.dim * self.n, self.code))

    def test_gradient_flows_to_latent_action(self):
        # The z_t -> \hat{A}_{t+1} path must carry gradient - this is how L_O reaches z_t later.
        z = self.z.clone().requires_grad_(True)
        self.module.agent_dynamics(self.a_t, z).sum().backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(z.grad.abs().sum().item(), 0.0)

    def test_gradient_flows_to_act_proj_and_conv(self):
        self.module.agent_dynamics(self.a_t, self.z).sum().backward()
        self.assertGreater(self.module.f_a.act_proj.weight.grad.abs().sum().item(), 0.0)
        conv_grads = [p.grad for p in self.module.f_a.res_blocks.parameters() if p.grad is not None]
        self.assertTrue(conv_grads and any(g.abs().sum().item() > 0 for g in conv_grads))

    def test_non_square_grid_raises(self):
        with self.assertRaises((ValueError, AssertionError)):
            InteractionModule(self.dim, num_tokens=8, num_heads=self.heads, code_dim=self.code, num_res_blocks=2)

    def test_num_res_blocks_is_required(self):
        # No default: a forgotten thread must be a loud TypeError, not a silent 2, so F_A's
        # residual depth cannot drift from the FDM's encoder_num_res_blocks (the confound).
        with self.assertRaises(TypeError):
            InteractionModule(self.dim, num_tokens=self.n, num_heads=self.heads, code_dim=self.code)

    def test_num_res_blocks_exposed_for_confound_guard(self):
        # Stored so InteractionWorldModel can assert f_a.num_res_blocks == encoder_num_res_blocks.
        self.assertEqual(self.module.f_a.num_res_blocks, 2)

    def test_runs_under_bf16_autocast(self):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = self.module.agent_dynamics(self.a_t, self.z)
        self.assertTrue(torch.isfinite(out).all())


class ObjectDynamicsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dim, self.heads, self.n, self.b, self.code = 32, 4, 9, 4, 8
        self.module = InteractionModule(
            self.dim, num_tokens=self.n, num_heads=self.heads, code_dim=self.code, num_res_blocks=2
        )
        self.o_t = torch.randn(self.b, self.n, self.dim)
        self.a_t = torch.randn(self.b, self.n, self.dim)
        self.a_hat = torch.randn(self.b, self.n, self.dim)
        self.z = torch.randn(self.b, self.code)

    def test_shape(self):
        o_hat = self.module.object_dynamics(self.o_t, self.a_t, self.a_hat)
        self.assertEqual(tuple(o_hat.shape), (self.b, self.n, self.dim))

    def test_object_attends_to_predicted_agent(self):
        # The hypothesis: object dynamics depend on the predicted agent transition.
        o1 = self.module.object_dynamics(self.o_t, self.a_t, self.a_hat)
        o2 = self.module.object_dynamics(self.o_t, self.a_t, torch.randn(self.b, self.n, self.dim))
        self.assertFalse(torch.allclose(o1, o2))

    def test_object_uses_current_agent(self):
        o1 = self.module.object_dynamics(self.o_t, self.a_t, self.a_hat)
        o2 = self.module.object_dynamics(self.o_t, torch.randn(self.b, self.n, self.dim), self.a_hat)
        self.assertFalse(torch.allclose(o1, o2))

    def test_no_direct_z_by_default(self):
        # Default: the object branch has NO direct z_t path (z flows only via A_hat).
        o1 = self.module.object_dynamics(self.o_t, self.a_t, self.a_hat, z=self.z)
        o2 = self.module.object_dynamics(self.o_t, self.a_t, self.a_hat, z=torch.randn(self.b, self.code))
        self.assertTrue(torch.allclose(o1, o2))

    def test_direct_z_to_object_flag_injects_z(self):
        # Matched ablation: with the flag on, z_t enters the object query directly.
        mod = InteractionModule(
            self.dim, num_tokens=self.n, num_heads=self.heads, code_dim=self.code,
            direct_z_to_object=True, num_res_blocks=2,
        )
        o1 = mod.object_dynamics(self.o_t, self.a_t, self.a_hat, z=self.z)
        o2 = mod.object_dynamics(self.o_t, self.a_t, self.a_hat, z=torch.randn(self.b, self.code))
        self.assertFalse(torch.allclose(o1, o2))

    def test_direct_z_requires_z(self):
        mod = InteractionModule(
            self.dim, num_tokens=self.n, num_heads=self.heads, code_dim=self.code,
            direct_z_to_object=True, num_res_blocks=2,
        )
        with self.assertRaises((ValueError, TypeError)):
            mod.object_dynamics(self.o_t, self.a_t, self.a_hat, z=None)

    def test_agent_ctx_mode_no_transition_differs(self):
        # Diagnostic substitution (eval-time): replace the predicted block with the current one.
        normal = self.module.object_dynamics(self.o_t, self.a_t, self.a_hat, agent_ctx_mode="normal")
        no_trans = self.module.object_dynamics(self.o_t, self.a_t, self.a_hat, agent_ctx_mode="no_transition")
        self.assertFalse(torch.allclose(normal, no_trans))

    def test_agent_ctx_mode_shuffled_differs(self):
        normal = self.module.object_dynamics(self.o_t, self.a_t, self.a_hat, agent_ctx_mode="normal")
        shuffled = self.module.object_dynamics(self.o_t, self.a_t, self.a_hat, agent_ctx_mode="shuffled")
        self.assertFalse(torch.allclose(normal, shuffled))

    def test_invalid_agent_ctx_mode_raises(self):
        with self.assertRaises(ValueError):
            self.module.object_dynamics(self.o_t, self.a_t, self.a_hat, agent_ctx_mode="bogus")

    def test_gradient_flows_to_predicted_agent(self):
        # The z_t -> A_hat -> O_hat path: L_O must reach A_hat (and thence z_t).
        a_hat = self.a_hat.clone().requires_grad_(True)
        self.module.object_dynamics(self.o_t, self.a_t, a_hat).sum().backward()
        self.assertIsNotNone(a_hat.grad)
        self.assertGreater(a_hat.grad.abs().sum().item(), 0.0)

    def test_runs_under_bf16_autocast(self):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = self.module.object_dynamics(self.o_t, self.a_t, self.a_hat)
        self.assertTrue(torch.isfinite(out).all())

    def test_shuffled_requires_batch_size_ge_2(self):
        # At B=1 the batch roll is the identity, so R_shuffled would silently read 1.0 ("no dependence").
        with self.assertRaises(ValueError):
            self.module.object_dynamics(
                self.o_t[:1], self.a_t[:1], self.a_hat[:1], agent_ctx_mode="shuffled"
            )

    def test_direct_z_reaches_content_not_only_attention(self):
        # z must reach o_hat's content, not merely reweight the softmax over agent values. Zero
        # cross_attn.out_proj so the attention path contributes nothing; any remaining z-dependence is
        # the direct content path (a query-only injection would leave o_hat z-independent here).
        mod = InteractionModule(
            self.dim, num_tokens=self.n, num_heads=self.heads, code_dim=self.code,
            num_res_blocks=2, direct_z_to_object=True,
        )
        with torch.no_grad():
            mod.cross_attn.out_proj.weight.zero_()
            mod.cross_attn.out_proj.bias.zero_()
        o1 = mod.object_dynamics(self.o_t, self.a_t, self.a_hat, z=self.z)
        o2 = mod.object_dynamics(self.o_t, self.a_t, self.a_hat, z=torch.randn(self.b, self.code))
        self.assertFalse(torch.allclose(o1, o2))

    def test_cross_attn_carries_no_mask_bias_parameter(self):
        # F_O's cross-attention is always unbiased (mask_bias=None), so it holds no beta parameter.
        self.assertIsNone(self.module.cross_attn.beta)


class InteractionNormalizationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dim, self.n, self.b = 32, 9, 2
        self.module = InteractionModule(self.dim, num_tokens=self.n, num_heads=4, code_dim=8, num_res_blocks=2)

    def test_extract_qk_norm_controls_input_scale(self):
        # The point of norm_extract: the attention Q/K are scale-controlled regardless of B_t
        # magnitude, so the mask bias beta*W is not swamped by content logits and the logged beta
        # stays interpretable. LayerNorm is positive-scale-invariant: norm(B_t) == norm(c*B_t).
        b_t = torch.randn(self.b, self.n, self.dim)
        self.assertTrue(
            torch.allclose(self.module.norm_extract(b_t), self.module.norm_extract(100.0 * b_t), atol=1e-4)
        )

    def test_extraction_values_stay_unnormalized(self):
        # Values must keep IMPALA scale (they feed F_A's conv blocks and F_O's residual). Since Q/K
        # are scale-invariant but V=B_t is not, scaling B_t must still change the read-out - if V were
        # normalized the whole op would be scale-invariant and the read-outs would coincide.
        b_t = torch.randn(self.b, self.n, self.dim)
        w = torch.rand(self.b, self.n)
        a1, _ = self.module.extract(b_t, w, w)
        a2, _ = self.module.extract(100.0 * b_t, w, w)
        self.assertFalse(torch.allclose(a1, a2))

    def test_all_norm_layers_receive_gradient(self):
        b_t = torch.randn(self.b, self.n, self.dim)
        w = torch.rand(self.b, self.n)
        o_t = torch.randn(self.b, self.n, self.dim)
        a_t = torch.randn(self.b, self.n, self.dim)
        a_hat = torch.randn(self.b, self.n, self.dim)
        loss = self.module.extract(b_t, w, w)[0].sum() + self.module.object_dynamics(o_t, a_t, a_hat).sum()
        loss.backward()
        for norm in (self.module.norm_extract, self.module.norm_obj, self.module.norm_ctx, self.module.norm_ffn):
            self.assertGreater(norm.weight.grad.abs().sum().item(), 0.0)


class WriteBackTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dim, self.heads, self.side, self.n, self.b = 32, 4, 3, 9, 2
        self.module = InteractionModule(
            self.dim, num_tokens=self.n, num_heads=self.heads, code_dim=8, num_res_blocks=2
        )
        self.b_t = torch.randn(self.b, self.dim, self.side, self.side)   # spatial bottleneck
        self.a_hat = torch.randn(self.b, self.n, self.dim)
        self.o_hat = torch.randn(self.b, self.n, self.dim)
        self.w_a = torch.rand(self.b, self.n)
        self.w_o = torch.rand(self.b, self.n)

    def test_write_back_shape(self):
        b_hat = self.module.write_back(self.b_t, self.a_hat, self.o_hat, self.w_a, self.w_o)
        self.assertEqual(tuple(b_hat.shape), (self.b, self.dim, self.side, self.side))

    def test_proj_zero_initialized(self):
        self.assertTrue(torch.all(self.module.proj_a.weight == 0))
        self.assertTrue(torch.all(self.module.proj_o.weight == 0))
        self.assertTrue(torch.all(self.module.proj_a.bias == 0))
        self.assertTrue(torch.all(self.module.proj_o.bias == 0))

    def test_write_back_is_identity_at_init(self):
        # Zero-init Proj => Delta B = 0 => B_hat == B_t bitwise (graceful-degradation guarantee).
        b_hat = self.module.write_back(self.b_t, self.a_hat, self.o_hat, self.w_a, self.w_o)
        self.assertTrue(torch.equal(b_hat, self.b_t))

    def test_proj_receives_gradient_despite_zero_init(self):
        # The zero-conv leaves zero on the first step: grad w.r.t. the Proj weight depends on its
        # input, not its (zero) value, so the interaction path opens from step 1.
        ones = torch.ones(self.b, self.n)
        self.module.write_back(self.b_t, self.a_hat, self.o_hat, ones, ones).sum().backward()
        self.assertGreater(self.module.proj_a.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.module.proj_o.weight.grad.abs().sum().item(), 0.0)

    def test_write_back_gated_by_occupancy(self):
        # With non-zero Proj: zero occupancy => no update (gate kills it); non-zero => update.
        with torch.no_grad():
            self.module.proj_a.weight.fill_(0.1)
            self.module.proj_o.weight.fill_(0.1)
        zero = torch.zeros(self.b, self.n)
        self.assertTrue(torch.equal(self.module.write_back(self.b_t, self.a_hat, self.o_hat, zero, zero), self.b_t))
        ones = torch.ones(self.b, self.n)
        self.assertFalse(torch.allclose(self.module.write_back(self.b_t, self.a_hat, self.o_hat, ones, ones), self.b_t))

    def test_dilate_expands_occupancy(self):
        # A one-hot occupancy cell should, after dilation, gate its neighbours too.
        occ = torch.zeros(1, 1, self.side, self.side)
        occ[0, 0, 1, 1] = 1.0
        dilated = self.module.dilate(occ)
        self.assertGreater(dilated[0, 0, 0, 0].item(), 0.0)

    def test_dilate_iters_must_be_at_least_1(self):
        # dilate_iters=0 reopens the t+1-coverage gap the dilation exists to close.
        with self.assertRaises(ValueError):
            InteractionModule(self.dim, num_tokens=self.n, num_heads=self.heads, num_res_blocks=2, dilate_iters=0)


class InteractionForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dim, self.heads, self.side, self.n, self.b, self.code = 32, 4, 3, 9, 2, 8
        self.module = InteractionModule(
            self.dim, num_tokens=self.n, num_heads=self.heads, code_dim=self.code, num_res_blocks=2
        )
        self.b_t = torch.randn(self.b, self.dim, self.side, self.side)
        self.w_a = torch.rand(self.b, self.n)
        self.w_o = torch.rand(self.b, self.n)
        self.z = torch.randn(self.b, self.code)

    def test_forward_shape(self):
        b_hat = self.module(self.b_t, self.w_a, self.w_o, self.z)
        self.assertEqual(tuple(b_hat.shape), (self.b, self.dim, self.side, self.side))

    def test_forward_is_identity_at_init(self):
        # The whole module is a zero-init residual on B_t: at init it is an exact no-op, so it can
        # only underperform its MaskLAM backbone by actively learning to.
        self.assertTrue(torch.equal(self.module(self.b_t, self.w_a, self.w_o, self.z), self.b_t))

    def test_forward_not_identity_after_proj_nonzero(self):
        with torch.no_grad():
            self.module.proj_a.weight.fill_(0.05)
        self.assertFalse(torch.allclose(self.module(self.b_t, self.w_a, self.w_o, self.z), self.b_t))

    def test_forward_return_attn_contract(self):
        out = self.module(self.b_t, self.w_a, self.w_o, self.z)
        self.assertIsInstance(out, torch.Tensor)
        b_hat, attn = self.module(self.b_t, self.w_a, self.w_o, self.z, return_attn=True)
        self.assertEqual(tuple(attn.shape), (self.b, self.heads, self.n, 2 * self.n))

    def test_forward_runs_under_bf16_autocast(self):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            b_hat = self.module(self.b_t, self.w_a, self.w_o, self.z)
        self.assertTrue(torch.isfinite(b_hat).all())

    def test_identity_survives_orthogonal_init_after_zero_init_write_back(self):
        # orthogonal_init (which the DMW config runs) re-inits every Conv2d, including the write-back
        # projections, and would silently destroy the identity; zero_init_write_back() must restore it.
        from ifo.common.utils.functions import orthogonal_init

        self.module.apply(orthogonal_init)
        self.assertFalse(torch.equal(self.module(self.b_t, self.w_a, self.w_o, self.z), self.b_t))
        self.module.zero_init_write_back()
        self.assertTrue(torch.equal(self.module(self.b_t, self.w_a, self.w_o, self.z), self.b_t))

    def test_forward_threads_agent_ctx_mode(self):
        # forward must thread agent_ctx_mode through to b_hat (else the agent-path ratios silently read 1.0).
        with torch.no_grad():
            self.module.proj_o.weight.fill_(0.05)  # open the object write-back so o_hat reaches b_hat
        normal = self.module(self.b_t, self.w_a, self.w_o, self.z, agent_ctx_mode="normal")
        no_trans = self.module(self.b_t, self.w_a, self.w_o, self.z, agent_ctx_mode="no_transition")
        self.assertFalse(torch.allclose(normal, no_trans))

    def test_forward_threads_direct_z(self):
        mod = InteractionModule(
            self.dim, num_tokens=self.n, num_heads=self.heads, code_dim=self.code,
            num_res_blocks=2, direct_z_to_object=True,
        )
        with torch.no_grad():
            mod.proj_o.weight.fill_(0.05)
        o1 = mod(self.b_t, self.w_a, self.w_o, self.z)
        o2 = mod(self.b_t, self.w_a, self.w_o, torch.randn(self.b, self.code))
        self.assertFalse(torch.allclose(o1, o2))


if __name__ == "__main__":
    unittest.main()
