import torch


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from {list(mapping)}")
    return mapping[dtype_str]


def resolve_task_config(config: dict, task: str) -> dict:
    """Flatten global defaults + the per-task entry into one config dict.

    Every top-level key except ``"tasks"`` is treated as a global default; the
    per-task entry under ``config["tasks"][task]`` overrides those defaults. The
    nested ``"tasks"`` table itself is not included in the result.
    """
    tasks = config.get("tasks", {})
    if task not in tasks:
        raise KeyError(f"Task {task!r} not found in config tasks: {sorted(tasks)}")
    defaults = {k: v for k, v in config.items() if k != "tasks"}
    return {**defaults, **tasks[task]}
