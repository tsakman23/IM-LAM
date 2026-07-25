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


if __name__ == "__main__":
    unittest.main()
