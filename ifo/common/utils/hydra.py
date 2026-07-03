import argparse
import functools
from textwrap import dedent
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar, Union

from hydra._internal.deprecation_warning import deprecation_warning
from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.core.utils import _flush_loggers
from hydra.main import _UNSPECIFIED_, _get_rerun_conf, version
from omegaconf import DictConfig

T = TypeVar("T")


class TaskFunction(Protocol):
    def __call__(self, cfg: DictConfig, **kwargs: Any) -> T: ...


def hydra_main_multistage(
    config_path: Optional[str] = _UNSPECIFIED_,
    config_name: Optional[Union[str, Dict[str, str]]] = None,
    version_base: Optional[str] = _UNSPECIFIED_,
) -> Callable[[TaskFunction], Any]:
    """
    A decorator for multi-stage Hydra configuration.

    Args:
        config_path (Optional[str]): The config path, a directory where Hydra will search for
            config files. This path is added to Hydra's searchpath. Relative paths are
            interpreted relative to the declaring python file. Alternatively, you can use
            the prefix `pkg://` to specify a python package to add to the searchpath.
            If config_path is None, no directory is added to the Config search path.
        config_name (Optional[Union[str, Dict[str, str]]]): The name of the config
            (usually the file name without the .yaml extension). Can be a string or a
            dictionary mapping stage names to config names. For a dictionary use
            `{stage_name: config_name}` for each stage.
        version_base (Optional[str]): The version base for Hydra configuration.

    Returns:
        Callable[[TaskFunction], Any]: A decorator that wraps a task function to enable multi-stage Hydra configuration.
    """

    version.setbase(version_base)

    if config_path is _UNSPECIFIED_:
        if version.base_at_least("1.2"):
            config_path = None
        elif version_base is _UNSPECIFIED_:
            url = "https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/changes_to_hydra_main_config_path"
            deprecation_warning(
                message=dedent(
                    f"""
                config_path is not specified in @hydra.main().
                See {url} for more information."""
                ),
                stacklevel=2,
            )
            config_path = "."
        else:
            config_path = "."

    def add_stage_args(args_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if isinstance(config_name, dict):
            args_parser.add_argument(
                "--stage",
                type=str,
                choices=config_name.keys(),
                required=True,
            )
        return args_parser

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args_parser = get_args_parser()
                args_parser = add_stage_args(args_parser)
                args, unknown_args = args_parser.parse_known_args()

                # Add unknown args to the overrides so Hydra can handle them
                if hasattr(args, "overrides"):
                    args.overrides.extend(unknown_args)
                else:
                    # Create overrides attribute if it doesn't exist
                    args.overrides = unknown_args
                stage_config_name: Optional[str] = None
                stage = None
                if hasattr(args, "stage") and isinstance(config_name, dict):
                    stage_config_name = config_name[args.stage]
                    stage = args.stage
                else:
                    stage_config_name = config_name if isinstance(config_name, str) else None
                if args.experimental_rerun is not None:
                    cfg = _get_rerun_conf(args.experimental_rerun, args.overrides)
                    task_function(cfg, stage=stage)
                    _flush_loggers()
                else:
                    # no return value from run_hydra() as it may sometime actually run the task_function
                    # multiple times (--multirun)
                    def wrapped_task_function(cfg: DictConfig) -> None:
                        task_function(cfg, stage=stage)

                    _run_hydra(
                        args=args,
                        args_parser=args_parser,
                        task_function=wrapped_task_function,
                        config_path=config_path,
                        config_name=stage_config_name,
                    )

        return decorated_main

    return main_decorator
