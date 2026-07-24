"""Tests for the IM-LAM interaction bottleneck module (Phase 2).

Run:  conda_env/bin/python tests/test_interaction_module.py

Built up task by task. Currently covers the token embeddings: a spatial
positional embedding (one vector per bottleneck cell) and a temporal embedding
marking whether a token describes the current or the predicted-next state.
There is deliberately NO per-entity embedding - agent/object separation comes
from the distinct mask biases and separate attention weight sets instead.
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ifo.common.nets.interaction import InteractionEmbeddings


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


if __name__ == "__main__":
    unittest.main()
