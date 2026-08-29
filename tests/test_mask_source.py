import pytest
import torch
from tensordict import TensorDict

from ifo.common.transforms.data import ProcessMask, get_dataset_transform, metaworld
from ifo.common.utils.data import build_mask_columns
from ifo.modules.slapo.utils import apply_mask_source


def test_metaworld_transform_gt_excludes_pred_masks():
    dt = metaworld(with_object_mask=True, mask_source="gt")
    assert "pred_mask" not in dt.transform
    assert "pred_object_mask" not in dt.transform


def test_metaworld_transform_sam_processes_pred_masks_like_gt():
    dt = metaworld(with_object_mask=True, mask_source="sam")
    assert isinstance(dt.transform["pred_mask"], ProcessMask)
    assert isinstance(dt.transform["pred_object_mask"], ProcessMask)


def test_metaworld_transform_sam_agent_only_has_no_pred_object():
    dt = metaworld(with_object_mask=False, mask_source="sam")
    assert isinstance(dt.transform["pred_mask"], ProcessMask)
    assert "pred_object_mask" not in dt.transform


def test_get_dataset_transform_threads_mask_source_to_metaworld():
    dt = get_dataset_transform("Meta-World/masked-MT1-push-v3", with_object_mask=True, mask_source="sam")
    assert "pred_mask" in dt.transform and "pred_object_mask" in dt.transform


def test_build_columns_gt_agent_only():
    assert build_mask_columns(with_object_mask=False, with_object_state=False, mask_source="gt") == [
        "observation", "mask", "action"
    ]


def test_build_columns_gt_with_object():
    cols = build_mask_columns(with_object_mask=True, with_object_state=False, mask_source="gt")
    assert cols == ["observation", "mask", "action", "object_mask"]
    assert "pred_mask" not in cols  # gt never loads predicted columns


def test_build_columns_sam_agent_only_adds_pred_mask():
    cols = build_mask_columns(with_object_mask=False, with_object_state=False, mask_source="sam")
    assert "pred_mask" in cols
    assert "mask" in cols  # non-destructive: GT stays loaded too
    assert "pred_object_mask" not in cols


def test_build_columns_sam_with_object_adds_both_pred_columns():
    cols = build_mask_columns(with_object_mask=True, with_object_state=False, mask_source="sam")
    for c in ("mask", "object_mask", "pred_mask", "pred_object_mask"):
        assert c in cols


def test_build_columns_unknown_source_raises():
    with pytest.raises(ValueError):
        build_mask_columns(with_object_mask=False, with_object_state=False, mask_source="oracle")


def _batch(with_object=True, with_pred=True, with_pred_object=True):
    d = {
        "mask": torch.full((2, 3), 1.0),
        "pred_mask": torch.full((2, 3), 9.0),
    }
    if with_object:
        d["object_mask"] = torch.full((2, 3), 2.0)
    if with_pred_object:
        d["pred_object_mask"] = torch.full((2, 3), 8.0)
    if not with_pred:
        del d["pred_mask"]
    return TensorDict(d, batch_size=[2])


def test_gt_is_noop():
    batch = apply_mask_source(_batch(), "gt")
    assert torch.equal(batch["mask"], torch.full((2, 3), 1.0))
    assert torch.equal(batch["object_mask"], torch.full((2, 3), 2.0))


def test_sam_swaps_agent_and_object_masks():
    batch = apply_mask_source(_batch(), "sam")
    assert torch.equal(batch["mask"], torch.full((2, 3), 9.0))          # pred_mask
    assert torch.equal(batch["object_mask"], torch.full((2, 3), 8.0))   # pred_object_mask


def test_sam_agent_only_when_no_object_columns():
    batch = apply_mask_source(_batch(with_object=False, with_pred_object=False), "sam")
    assert torch.equal(batch["mask"], torch.full((2, 3), 9.0))
    assert "object_mask" not in batch.keys()


def test_sam_requires_pred_mask():
    with pytest.raises(KeyError):
        apply_mask_source(_batch(with_pred=False), "sam")


def test_sam_object_mask_without_pred_object_mask_raises():
    # GT object present but no SAM object -> would silently mix GT object with SAM agent.
    with pytest.raises(KeyError):
        apply_mask_source(_batch(with_pred_object=False), "sam")


def test_unknown_mask_source_raises():
    with pytest.raises(ValueError):
        apply_mask_source(_batch(), "predicted")
