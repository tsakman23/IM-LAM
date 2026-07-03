"""
Dataset generation script for expert demonstrations.

This script generates a dataset of expert demonstrations by running rollouts with a pre-trained expert policy.
The generated dataset contains trajectories of expert behavior that can be used for imitation learning methods
like BC, ILPO, or LAPO. The dataset is automatically uploaded to Hugging Face for easy sharing and reuse.

Note: A trained expert policy must be available before running this script and specified by the `run_id` argument.
"""

from ifo.common.runner import Runner
from ifo.modules.dataset.experiment import DatasetExperiment

if __name__ == "__main__":
    runner = Runner(DatasetExperiment)
    runner.run()
