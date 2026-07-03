"""
Behavioral Cloning experiment runner.

This script runs a behavioral cloning experiment to train a policy by learning from expert demonstrations.
The policy is trained to mimic expert behavior by learning from a dataset of expert trajectories.
"""

import coolname

from ifo.common.runner import Runner
from ifo.modules.bc.experiment import BCExperiment

if __name__ == "__main__":
    run_id = coolname.generate_slug(3)
    runner = Runner(BCExperiment)
    default_global_args = [f"run_id={run_id}"]
    runner.run(default_global_args=default_global_args)
