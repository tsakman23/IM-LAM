# Ground-truth mask figures (Distracting Meta-World)

Report-ready visualizations of the ground-truth simulator segmentation masks produced by the
extended DMW dataset builder (`distracting-metaworld-dataset-main`): the existing agent `mask`
and the new `object_mask`.

Render settings: expert = Meta-World scripted oracle, `seed=0`, camera `corner`, `128x128`,
clean (non-distracted) observation. Frame indices match the deterministic seed-0 rollout.

## Contents

Per task (`push-v3`, `door-open-v3`), for two representative frames (`072`, `144`):

| file | description |
|---|---|
| `frameNNN_observation.png` | clean RGB observation (native 128x128) |
| `frameNNN_agent_mask.png`  | agent GT mask, binary 0/255, mode L (native 128x128) |
| `frameNNN_object_mask.png` | object GT mask, binary 0/255, mode L (native 128x128) |
| `frameNNN_overlay.png`     | observation with agent (green) + object (red) tinted (native 128x128) |
| `frameNNN_panel.png`       | composite strip `observation | agent | object | overlay`, upscaled 3x (nearest) |

`mask-gallery.html` is a self-contained viewer (all frames of both tasks, embedded); open in any browser.

The native-resolution single tiles are the source figures - upscale/vectorize as needed for the
report. The masks are crisp under nearest-neighbor upscaling because they are binary.

## Notes for the writeup

- Masks are read directly from the MuJoCo simulator (no post-processing), binarized 0/255,
  and are occlusion-correct: each pixel is labeled by the front-most geom, so agent and object
  masks are mutually exclusive (visible where the gripper contacts the door handle).
- Object geoms are selected by a movable-w.r.t.-world heuristic (a body reachable from the world
  through at least one joint), which covers both free bodies (the push puck) and articulated parts
  (the door hinge) with no per-task tuning. Rendered *sites* such as the green goal marker are
  excluded via an `objtype == mjOBJ_GEOM` filter, so the object mask tracks the puck / door, never
  the goal.
- The `push-v3` object (puck) is only ~10-17 px at 128x128 - small by nature, not a masking error.

## Reproduce

```bash
cd distracting-metaworld-dataset-main
# per-frame composite panels + counts (all frames):
MUJOCO_GL=egl PYTHONPATH=. python verify/render_mask_panels.py \
    --env Meta-World/MT1-push-v3 --steps 150 --num_frames 8 \
    --out datasets/verify/mask_panels/push-v3
```
Individual native tiles and this gallery were extracted from the same seed-0 rollout.
