"""Per-task action-variance constants for NMSE (MaskLAM/LAOM Eq. 5 convention).

Var(a): the mean per-dimension variance of the clipped ground-truth actions on
a task's training split - the denominator of normalized linear action-probe
MSE. Four values below are MaskLAM's own published Table 11 entries (their
MT10 tasks); two are computed directly from this project's own regenerated
datasets for tasks MaskLAM does not report (see expert_constants.json at the
project root / experiments.md for the exact computation).

dial-turn-v3 is intentionally absent: its Meta-World V3 scripted "expert"
never succeeds (0/1996 episodes in the generated dataset, independently
reproduced against an unmodified vanilla environment), so no usable NMSE
denominator exists for that task.
"""

from typing import Optional

# task slug -> Var(a)
ACTION_VARIANCE = {
    "push-v3": 0.1311,             # MaskLAM Table 11
    "door-open-v3": 0.4266,        # MaskLAM Table 11
    "pick-place-v3": 0.2286,       # MaskLAM Table 11
    "peg-insert-side-v3": 0.2497,  # MaskLAM Table 11
    "sweep-into-v3": 0.2605,       # computed, expert_constants.json
    "handle-pull-v3": 0.3532,      # computed, expert_constants.json
}


def task_slug(env_name: str) -> str:
    """'Meta-World/masked-MT1-push-v3' -> 'push-v3'.

    Mirrors the parsing in ifo.common.utils.data.get_metaworld_dataset.
    """
    name = env_name.split("/")[-1] if "/" in env_name else env_name
    name = name.replace("MT1-", "")
    return name.replace("distracting-", "").replace("masked-", "")


def get_action_variance(env_name: str) -> Optional[float]:
    """Look up Var(a) for a Meta-World env name.

    Returns None for any task not in ACTION_VARIANCE (e.g. dial-turn-v3, or a
    task not yet added) - the caller (SLAPOIDMModule) treats None as "disable
    the action_decoder_nmse metric" rather than raising.
    """
    return ACTION_VARIANCE.get(task_slug(env_name))
