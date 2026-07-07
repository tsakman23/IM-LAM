import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from make_env import make_metaworld_env  # noqa: E402


def _seg_env():
    return make_metaworld_env(
        env_name="Meta-World/MT1-push-v3", num_envs=1,
        camera_name="corner", seed=0, height=128, width=128,
        distracting=False, segmentation=True,
    )


def test_render_masks_agent_matches_legacy_arm_mask():
    env = _seg_env()
    obs, _ = env.reset(seed=0)
    # Legacy arm mask surfaced through obs["image"] (mode-L foreground).
    legacy = np.asarray(obs["image"][0])
    legacy_fg = np.any(legacy > 0, axis=-1) if legacy.ndim == 3 else legacy > 0
    masks = env.call("render_masks")[0]
    agent_fg = masks["agent"][:, :, 0] > 0
    assert agent_fg.shape == legacy_fg.shape
    assert agent_fg.sum() > 0, "agent mask unexpectedly empty (test would pass vacuously)"
    assert np.array_equal(agent_fg, legacy_fg), "render_masks agent must equal legacy arm mask"
    env.close()


def test_object_mask_nonempty_and_disjoint_from_agent():
    env = _seg_env()
    env.reset(seed=0)
    masks = env.call("render_masks")[0]
    agent_fg = masks["agent"][:, :, 0] > 0
    object_fg = masks["object"][:, :, 0] > 0
    assert object_fg.sum() > 0, "push-v3 puck should produce a non-empty object mask"
    assert not np.any(agent_fg & object_fg), "agent and object masks must be disjoint"
    env.close()


if __name__ == "__main__":
    test_render_masks_agent_matches_legacy_arm_mask()
    print("PASS: test_render_masks_agent_matches_legacy_arm_mask")
    test_object_mask_nonempty_and_disjoint_from_agent()
    print("PASS: test_object_mask_nonempty_and_disjoint_from_agent")
