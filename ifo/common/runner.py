import inspect
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type

from ifo.common.experiment import Experiment
from ifo.common.utils.utility import display_banner


class Runner:
    """Runner for multi-stage experiments."""

    def __init__(self, experiment_cls: Type[Experiment]):
        """Initialize the runner with an experiment.

        Args:
            experiment_cls: The class of the experiment to run.
        """
        self.experiment_cls = experiment_cls

    def show_help(self):
        help_text = f"""
Usage: python script.py [GLOBAL_ARGS] [--selected_stages STAGES] [--stage STAGE_NAME STAGE_ARGS]...

Description:
  This script runs a multi-stage experiment pipeline using Hydra and Python.
  You can pass global arguments to all stages and specify stage-specific arguments.

Available stages: {self.experiment_cls.stages}

Options:
  --selected_stages=<stages>   Comma-separated list of stages to run (e.g., --selected_stages=[stage_1,stage_2]).
                               If not specified, all stages will run.
  --stage <name>               Set the current stage for the following arguments.
                               Arguments after --stage are passed only to that stage.
  GLOBAL_ARGS                  key=value arguments passed to all stages (before any --stage flag).

Examples:
  1. Run all stages with a global argument:
     python script.py env_name=test

  2. Run only specific stages:
     python script.py --selected_stages=[stage_1,stage_2] env_name=test

  3. Pass stage-specific arguments (runs all stages by default):
     python script.py env_name=test --stage stage_1 lr=0.001 --stage stage_2 epochs=10
     -> Passes env_name=test to all stages
     -> Passes lr=0.001 only to stage_1
     -> Passes epochs=10 only to stage_2

  4. Run specific stages with stage-specific arguments:
     python script.py --selected_stages=[stage_2,stage_3] --stage stage_2 model=ResNet --stage stage_3 resume=True

Flags:
  -h     Show this help message and exit
"""
        print(help_text)
        sys.exit(0)

    def parse_selected_stages(self, value: str) -> List[str]:
        """Parse the --selected_stages argument value.

        Args:
            value: The value string, e.g., "[stage_1,stage_2]" or "stage_1,stage_2"

        Returns:
            List of stage names to run.
        """
        # Remove brackets if present
        value = value.strip("[]")
        # Split by comma and strip whitespace
        stages = [s.strip() for s in value.split(",") if s.strip()]
        # Validate stages
        for stage in stages:
            if stage not in self.experiment_cls.stages:
                print(f"Invalid stage: {stage}. Valid stages: {self.experiment_cls.stages}")
                sys.exit(1)
        return stages

    def parse_arguments(self, argv: List[str]) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
        """Parse command line arguments into global and stage-specific arguments.

        Args:
            argv: List of command line arguments to parse.

        Returns:
            A tuple containing:
                - List of global arguments to be passed to all stages
                - Dictionary mapping stage names to their specific arguments
                - List of stages to run (defaults to all stages if none specified)

        Raises:
            SystemExit: If an unrecognized argument is encountered
        """
        global_args = []
        stage_args = defaultdict(list)
        selected_stages = []

        # Hydra flags that should be recognized and properly handled
        hydra_flags_with_values = {"-cn", "--config-name", "-cp", "--config-path", "-cd", "--config-dir", "--package"}

        hydra_flags_without_values = {
            "--help",
            "--hydra-help",
            "--version",
            "--cfg",
            "--resolve",
            "--run",
            "--multirun",
            "--shell-completion",
            "--info",
        }

        current_stage = None
        i = 0
        while i < len(argv):
            arg = argv[i]

            # Only -h triggers help display
            if arg == "-h":
                self.show_help()
            elif arg.startswith("--selected_stages="):
                # Handle --selected_stages=[stage_1,stage_2,...] format
                value = arg.split("=", 1)[1]
                selected_stages = self.parse_selected_stages(value)
            elif arg == "--selected_stages":
                # Handle --selected_stages [stage_1,stage_2,...] format (space separated)
                if i + 1 < len(argv):
                    selected_stages = self.parse_selected_stages(argv[i + 1])
                    i += 1  # Skip the value
                else:
                    print("--selected_stages flag requires a value (e.g., --selected_stages=[stage_1,stage_2])")
                    sys.exit(1)
            elif arg == "--stage":
                # Handle --stage flag: sets current stage for following arguments (does NOT select the stage)
                if i + 1 < len(argv):
                    current_stage = argv[i + 1]
                    if current_stage not in self.experiment_cls.stages:
                        print(f"Invalid stage: {current_stage}. Valid stages: {self.experiment_cls.stages}")
                        sys.exit(1)
                    i += 1  # Skip the stage name
                else:
                    print("--stage flag requires a stage name")
                    sys.exit(1)
            elif "=" in arg:
                # Handle key=value arguments
                if current_stage:
                    stage_args[current_stage].append(arg)
                else:
                    global_args.append(arg)
            elif arg in hydra_flags_with_values:
                # Handle Hydra flags that expect a value
                if i + 1 < len(argv):
                    if current_stage:
                        stage_args[current_stage].extend([arg, argv[i + 1]])
                    else:
                        global_args.extend([arg, argv[i + 1]])
                    i += 1  # Skip the next argument as we've already processed it
                else:
                    print(f"Flag {arg} requires a value")
                    sys.exit(1)
            elif arg in hydra_flags_without_values:
                # Handle Hydra flags without values
                if current_stage:
                    stage_args[current_stage].append(arg)
                else:
                    global_args.append(arg)
            elif arg.startswith("-"):
                # Handle other command line flags
                if (
                    i + 1 < len(argv)
                    and not argv[i + 1].startswith("-")
                    and argv[i + 1] not in self.experiment_cls.stages
                ):
                    # Flag with value
                    if current_stage:
                        stage_args[current_stage].extend([arg, argv[i + 1]])
                    else:
                        global_args.extend([arg, argv[i + 1]])
                    i += 1  # Skip the next argument as we've already processed it
                else:
                    # Flag without a value
                    if current_stage:
                        stage_args[current_stage].append(arg)
                    else:
                        global_args.append(arg)
            else:
                print(f"Unrecognized argument: {arg}")
                print("Run 'python script.py -h' for usage.")
                sys.exit(1)

            i += 1

        # If no specific stages given, run all
        if not selected_stages:
            selected_stages = list(self.experiment_cls.stages)

        return global_args, stage_args, selected_stages

    def run_stages(self, global_args: List[str], stage_args: Dict[str, List[str]], selected_stages: List[str]) -> None:
        """Execute the selected stages with their respective arguments.

        Args:
            global_args: Arguments to be passed to all stages
            stage_args: Dictionary mapping stage names to their specific arguments
            selected_stages: List of stages to run

        Raises:
            SystemExit: If any stage fails to execute successfully
        """

        for stage in selected_stages:
            cmd = ["python", self.get_module_path(), "--stage", stage, *global_args, *stage_args[stage]]
            print("Running:", " ".join(cmd))
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"❌ Stage failed: {stage}")
                sys.exit(result.returncode)

    def get_module_path(self) -> str:
        """Get the module path from the experiment class.

        Returns:
            str: The module path in the format 'module/submodule/file.py'.
        """
        module = inspect.getmodule(self.experiment_cls)
        if module is None:
            raise ValueError(f"Could not find module for class {self.experiment_cls.__name__}")

        # Get the file path relative to the workspace root
        file_path = module.__file__
        if file_path is None:
            raise ValueError(f"Could not find file path for module {module.__name__}")

        # Convert to relative path and remove .py extension
        relative_path = file_path.replace(sys.path[0] + "/", "")
        return relative_path

    def merge_args(self, default_args: List[str], parsed_args: List[str]) -> List[str]:
        """Merge default and parsed arguments.

        If an argument is provided in both default and parsed arguments, the parsed argument will take precedence.
        """

        # Arguments can come in different formats:
        # 1. Flag with value: ['-cn', 'ppo'] -> becomes "-cn ppo"
        # 2. Overriding a config value: ['foo.bar=value'] -> becomes "foo.bar=value"
        # 3. Appending a config value: ['+foo.bar=value'] -> becomes "+foo.bar=value"
        # 4. Appending or overriding a config value: ['++foo.bar=value'] -> becomes "++foo.bar=value"
        # 5. Removing a config value: ['~foo.bar'] or ['~foo.bar=value'] -> becomes "~foo.bar" or "~foo.bar=value"
        # We only care about 2., 3., 4. and 5.
        def get_key(arg: str) -> Optional[str]:
            for sep in ("=",):
                if sep in arg:
                    return arg.split(sep, 1)[0]
            return None

        parsed_keys = {get_key(arg) for arg in parsed_args if get_key(arg)}
        merged = [arg for arg in default_args if get_key(arg) not in parsed_keys]
        return parsed_args + merged

    def merge_arguments(
        self,
        default_global_args: Optional[List[str]] = None,
        default_stage_args: Optional[Dict[str, List[str]]] = None,
        global_args: List[str] = [],
        stage_args: Dict[str, List[str]] = {},
        selected_stages: List[str] = [],
    ) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
        """Merge default and parsed arguments.

        If an argument is provided in both default and parsed arguments, the parsed argument will take precedence.

        Args:
            default_global_args: Default arguments to be passed to all stages
            default_stage_args: Default arguments for specific stages
            global_args: Parsed arguments to be passed to all stages
            stage_args: Parsed arguments for specific stages
            selected_stages: Stages to run

        Returns:
            A tuple containing:
                - Merged global arguments
                - Merged stage arguments
                - List of stages to run
        """
        if default_global_args is not None:
            global_args = self.merge_args(default_global_args, global_args)
        if default_stage_args is not None:
            for stage in default_stage_args.keys():
                stage_args[stage] = self.merge_args(default_stage_args[stage], stage_args[stage])
        return global_args, stage_args, selected_stages

    def print_run_info(
        self, global_args: List[str], stage_args: Dict[str, List[str]], selected_stages: List[str]
    ) -> None:
        """Print information about the run configuration.

        Args:
            global_args: Arguments to be passed to all stages
            stage_args: Dictionary mapping stage names to their specific arguments
            selected_stages: List of stages to run
        """
        print(f"Running stages: {selected_stages}")
        print(f"Global args: {global_args}")
        print("Stage args:")
        for stage, args in stage_args.items():
            if stage in selected_stages:
                print(f"  {stage}: {args}")

    def run(
        self,
        default_global_args: Optional[List[str]] = None,
        default_stage_args: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run the experiment.

        Args:
            default_global_args: Default arguments to be passed to all stages
            default_stage_args: Default arguments for specific stages
        """
        global_args, stage_args, selected_stages = self.parse_arguments(sys.argv[1:])
        global_args, stage_args, selected_stages = self.merge_arguments(
            default_global_args, default_stage_args, global_args, stage_args, selected_stages
        )
        display_banner()
        self.print_run_info(global_args, stage_args, selected_stages)
        self.run_stages(global_args, stage_args, selected_stages)
