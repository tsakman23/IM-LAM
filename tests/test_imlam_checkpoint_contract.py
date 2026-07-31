"""IM-LAM Stage-1 -> Stage-2 checkpoint contract on REAL serialization (Phase 8, integration check C).

The synthetic Phase-4 test asserts IMLAMIDM.encoder shares the same 32 keys as a plain SLAPOIDM. This
goes further and exercises the actual stage_2 load path (ifo/modules/slapo/experiment.py:145-146):

    state_dict = torch.load(checkpoint)["model"]
    net.inverse_dynamics_model.load_state_dict(filter_state_dict_by_prefix(state_dict, "net.encoder"))

It round-trips a real Stage-1 module state_dict through torch.save/torch.load, applies the REAL
SLAPOExperiment.filter_state_dict_by_prefix, and STRICT-loads the result into a real Stage-2
SLAPOLatentPolicy.inverse_dynamics_model - so a serialization / prefix / shape mismatch that the
key-set test would miss fails here instead of at the start of a Stage-2 run.

Run:  conda_env/bin/python tests/test_imlam_checkpoint_contract.py
"""
import functools
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ifo.common.nets.interaction_world_model import InteractionWorldModel
from ifo.common.nets.quantizer import IdentityQuantizer
from ifo.modules.slapo.experiment import SLAPOExperiment
from ifo.modules.slapo.module import SLAPOIDMModule
from ifo.modules.slapo.nets import IMLAMIDM, SLAPOLatentPolicy

FRAME_STACK, OFFSET, C, H, W, ACT_DIM = 3, 2, 3, 32, 32, 4
CM, CODE_DIM = 1, 8
ENC_KEYS = 32  # Phase-0 recorded net.encoder key count


def _spaces():
    return (spaces.Box(0.0, 1.0, (FRAME_STACK + OFFSET, C, H, W), np.float32),
            spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32))


def _imlam():
    obs, act = _spaces()
    wm = functools.partial(InteractionWorldModel, channel_multiplier=CM, encoder_channels=(16, 32, 32),
                           encoder_num_res_blocks=2, num_heads=4)
    return IMLAMIDM(observation_space=obs, action_space=act, channel_multiplier=CM, quantizer=IdentityQuantizer(),
                    world_model=wm, code_dim=CODE_DIM, num_latents=1, frame_stack=FRAME_STACK,
                    future_obs_offset=OFFSET, add_mask_to_observation=True)


def _latent_policy():
    # Stage-2 net; its inverse_dynamics_model is built identically to SLAPOIDM.encoder.
    obs, act = _spaces()
    return SLAPOLatentPolicy(observation_space=obs, action_space=act, channel_multiplier=CM, code_dim=CODE_DIM,
                             frame_stack=FRAME_STACK, future_obs_offset=OFFSET, add_mask_to_observation=True)


def _filter(state_dict, prefix):
    # The real, stateless stage_2 filter (bypass __init__ since the method uses no instance state).
    return SLAPOExperiment.filter_state_dict_by_prefix(object.__new__(SLAPOExperiment), state_dict, prefix)


class IMLAMCheckpointContractTest(unittest.TestCase):
    def _stage1_checkpoint_model(self, imlam):
        # A real Stage-1 checkpoint's "model" is the SLAPOIDMModule state_dict (net.* keys). Round-trip it
        # through disk so the test covers real serialization, not just an in-memory dict.
        module = SLAPOIDMModule(net=imlam, batch_size=1, optimizer=OmegaConf.create({}), mask_loss=True)
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "stage1.ckpt")
            torch.save({"model": module.state_dict()}, ckpt)
            return torch.load(ckpt, weights_only=False)["model"]

    def test_serialized_stage1_encoder_strict_loads_into_stage2_idm(self):
        imlam = _imlam()
        model_sd = self._stage1_checkpoint_model(imlam)
        enc = _filter(model_sd, "net.encoder")
        self.assertEqual(len(enc), ENC_KEYS)

        # Strict load: raises on any missing / unexpected / shape mismatch. Reaching the asserts = exact match.
        lp = _latent_policy()
        result = lp.inverse_dynamics_model.load_state_dict(enc, strict=True)
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])

        # Values survived serialization: a loaded IDM param equals the source IMLAMIDM encoder param.
        key = next(iter(enc))
        self.assertTrue(torch.equal(lp.inverse_dynamics_model.state_dict()[key], imlam.encoder.state_dict()[key]))

    def test_decoder_keys_do_not_satisfy_the_idm_contract(self):
        # Negative control: the FDM (net.decoder) keys must NOT strict-load into the IDM - proves the
        # positive test is actually exercising strict key/shape matching, not vacuously passing.
        imlam = _imlam()
        model_sd = self._stage1_checkpoint_model(imlam)
        dec = _filter(model_sd, "net.decoder")
        self.assertGreater(len(dec), 0)
        with self.assertRaises(RuntimeError):
            _latent_policy().inverse_dynamics_model.load_state_dict(dec, strict=True)


if __name__ == "__main__":
    unittest.main()
