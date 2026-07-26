"""Tests for the IM-LAM IDM net (IMLAMIDM, Phase 4).

Run:  conda_env/bin/python tests/test_imlam_net.py

IMLAMIDM subclasses SLAPOIDM and replaces ONLY the FDM (self.decoder) with the
InteractionWorldModel, rebuilt at c+2 input channels (RGB + agent + object mask). The IDM
(self.encoder) is left byte-identical to MaskLAM's so a MaskLAM Stage-1 checkpoint's
`net.encoder.*` keys (32 of them) still load into Stage 2/3. These tests pin the FDM
rebuild, the object-mask routing, the checkpoint contract, and - the Phase-3 review's live
hazard - that use_orthogonal_init actually reaches the rebuilt decoder.
"""
import functools
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ifo.common.nets.interaction_world_model import InteractionWorldModel
from ifo.common.nets.world_model import ImpalaWorldModel
from ifo.common.nets.quantizer import IdentityQuantizer
from ifo.modules.slapo.nets import IMLAMIDM, SLAPOIDM

FRAME_STACK = 3
OFFSET = 2
C = 3          # RGB
H = W = 32     # 3 encoder blocks -> 4x4 square bottleneck
ACT_DIM = 4
ENC_KEY_COUNT = 32  # Phase-0 recorded target: net.encoder keys in a real MaskLAM checkpoint


def _spaces():
    obs = spaces.Box(low=0.0, high=1.0, shape=(FRAME_STACK + OFFSET, C, H, W), dtype=np.float32)
    act = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
    return obs, act


def _common_kwargs(world_model):
    obs, act = _spaces()
    return dict(
        observation_space=obs,
        action_space=act,
        channel_multiplier=1,
        quantizer=IdentityQuantizer(),
        world_model=world_model,
        code_dim=8,
        num_latents=1,
        frame_stack=FRAME_STACK,
        future_obs_offset=OFFSET,
        add_mask_to_observation=True,
    )


def _imlam(**kw):
    wm = functools.partial(
        InteractionWorldModel, channel_multiplier=1, encoder_channels=(16, 32, 32),
        encoder_num_res_blocks=2, num_heads=4,
    )
    kwargs = _common_kwargs(wm)
    kwargs.update(kw)
    return IMLAMIDM(**kwargs)


def _slapoidm(**kw):
    wm = functools.partial(
        ImpalaWorldModel, channel_multiplier=1, encoder_channels=(16, 32, 32), encoder_num_res_blocks=2,
    )
    kwargs = _common_kwargs(wm)
    kwargs.update(kw)
    return SLAPOIDM(**kwargs)


def _first_conv(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise AssertionError("no Conv2d found")


class IMLAMIDMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.b = 2
        self.x = torch.randn(self.b, FRAME_STACK + OFFSET, C, H, W)
        self.agent_mask = (torch.rand(self.b, FRAME_STACK + OFFSET, 1, H, W) > 0.5).float()
        self.object_mask = (torch.rand(self.b, FRAME_STACK + OFFSET, 1, H, W) > 0.5).float()

    def test_forward_returns_four_tuple_with_correct_shapes(self):
        net = _imlam().eval()
        out = net(self.x, self.agent_mask, object_mask=self.object_mask)
        self.assertEqual(len(out), 4)
        next_observation, action_distribution, vq_loss, perplexity = out
        self.assertEqual(tuple(next_observation.shape), (self.b, C, H, W))
        self.assertTrue(torch.all(next_observation >= -0.5) and torch.all(next_observation <= 0.5))
        self.assertEqual(action_distribution.mean.shape[-1], ACT_DIM)
        self.assertTrue(torch.isfinite(vq_loss).all() and torch.isfinite(perplexity).all())

    def test_fdm_consumes_c_plus_2_while_idm_stays_agent_only(self):
        # The FDM (decoder) encoder ingests RGB + agent + object mask = c+2 per frame; the IDM
        # encoder stays agent-mask-only (2*frame_stack*(c+1), current+future) - the checkpoint-compat asymmetry.
        net = _imlam()
        self.assertEqual(_first_conv(net.decoder.encoder).in_channels, FRAME_STACK * (C + 2))       # 15
        self.assertEqual(_first_conv(net.encoder).in_channels, 2 * FRAME_STACK * (C + 1))            # 24

    def test_encoder_key_contract_matches_plain_slapoidm_and_is_32(self):
        # Stage 2/3 load the frozen `net.encoder`; IMLAMIDM must keep it byte-identical to a plain
        # SLAPOIDM's (same key names, same shapes) so the MaskLAM 32-key load still matches.
        imlam_sd, plain_sd = _imlam().state_dict(), _slapoidm().state_dict()
        imlam_keys = {k for k in imlam_sd if k.startswith("encoder.")}
        plain_keys = {k for k in plain_sd if k.startswith("encoder.")}
        self.assertEqual(imlam_keys, plain_keys)
        self.assertEqual(len(imlam_keys), ENC_KEY_COUNT)
        for k in imlam_keys:
            self.assertEqual(imlam_sd[k].shape, plain_sd[k].shape)

    def test_orthogonal_init_reaches_rebuilt_decoder_and_rezeros_write_back(self):
        # Phase-3 review hazard: the net.world_model partial does NOT carry use_orthogonal_init, and
        # super().__init__'s apply(orthogonal_init) runs over the THROWAWAY decoder before the rebuild.
        # If IMLAMIDM forgets to thread use_orthogonal_init into the rebuild, the IM-LAM FDM silently
        # trains from PyTorch-default init while MaskLAM/Foreground's FDM is orthogonal. Pin it: a decoder
        # output conv must be semi-orthogonal (W W^T = I), and the write-back must still be re-zeroed.
        net = _imlam(use_orthogonal_init=True)
        conv = net.decoder.decoder[-1]  # final Conv2d(16, 3, 1); out_channels <= in_channels -> rows orthonormal
        w = conv.weight.data.view(conv.out_channels, -1)
        self.assertTrue(torch.allclose(w @ w.T, torch.eye(conv.out_channels), atol=1e-5))
        self.assertEqual(net.decoder.interaction.proj_a.weight.abs().sum().item(), 0.0)
        self.assertEqual(net.decoder.interaction.proj_o.weight.abs().sum().item(), 0.0)

    def test_parent_slapoidm_forward_accepts_and_ignores_object_mask(self):
        # SLAPOIDM.forward gains object_mask=None so the shared module can always pass it; MaskLAM/
        # Foreground must be byte-for-byte unaffected whether or not it is supplied.
        net = _slapoidm().eval()
        out_without = net(self.x, self.agent_mask)
        out_with = net(self.x, self.agent_mask, object_mask=self.object_mask)
        self.assertTrue(torch.equal(out_without[0], out_with[0]))

    def test_object_mask_is_routed_to_the_fdm(self):
        # Once the object write-back opens, a different object mask must change the prediction -
        # i.e. IMLAMIDM actually forwards object_mask to the FDM rather than dropping it.
        net = _imlam().eval()
        with torch.no_grad():
            net.decoder.interaction.proj_o.weight.normal_(std=0.1)
        y1 = net(self.x, self.agent_mask, object_mask=self.object_mask)[0]
        y2 = net(self.x, self.agent_mask, object_mask=1.0 - self.object_mask)[0]
        self.assertFalse(torch.equal(y1, y2))

    def test_agent_ctx_mode_plumbs_through_to_the_fdm(self):
        # The eval-time agent-path substitution must reach the interaction module (Phase-8 diagnostic).
        net = _imlam().eval()
        with torch.no_grad():
            net.decoder.interaction.proj_o.weight.normal_(std=0.1)
        y_normal = net(self.x, self.agent_mask, object_mask=self.object_mask, agent_ctx_mode="normal")[0]
        y_notrans = net(self.x, self.agent_mask, object_mask=self.object_mask, agent_ctx_mode="no_transition")[0]
        self.assertFalse(torch.equal(y_normal, y_notrans))

    def test_return_attn_appends_attention_maps(self):
        # return_attn=True keeps the 4-tuple and appends the object->agent attention (for diagnostics).
        net = _imlam().eval()
        out = net(self.x, self.agent_mask, object_mask=self.object_mask, return_attn=True)
        self.assertEqual(len(out), 5)
        attn = out[4]
        self.assertEqual(attn.shape[0], self.b)
        self.assertEqual(attn.dim(), 4)  # (B, heads, N, 2N)

    def test_forward_requires_object_mask(self):
        # IM-LAM's FDM is defined over agent+object masks; a missing object_mask is a usage error, not
        # a silent agent-only fallback.
        net = _imlam().eval()
        with self.assertRaises(ValueError):
            net(self.x, self.agent_mask)

    def test_disabling_fdm_object_mask_input_raises(self):
        # The FDM is architecturally defined around the object mask (it routes the interaction attention
        # and the write-back gate), so there is no agent-only FDM variant. fdm_object_mask_input=False is
        # unsupported and must fail loudly rather than be a silent no-op.
        with self.assertRaises(ValueError):
            _imlam(fdm_object_mask_input=False)


if __name__ == "__main__":
    unittest.main()
