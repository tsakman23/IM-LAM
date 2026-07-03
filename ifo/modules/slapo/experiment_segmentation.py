import os

from omegaconf import DictConfig

from ifo.common.utils.hydra import hydra_main_multistage
from ifo.common.utils.utility import (
    Conditional,
    add_legacy_features_to_vector_wrapper,
)
from ifo.modules.slapo.experiment import SLAPOExperiment


class SLAPOSegmentationExperiment(SLAPOExperiment):
    """Class for SLAPO with predicted mask from segmentation model experiment.

    Note: This experiment additionally executes stage_0, which is used to train a UNet-based binary segmentation model
          on ground-truth masks.
    """

    stages = ["stage_0", "stage_1", "stage_2", "stage_3"]
    config = {
        "stage_0": "slapo_segmentation_default_stage_0",
        "stage_1": "slapo_segmentation_default_stage_1",
        "stage_2": "slapo_segmentation_default_stage_2",
        "stage_3": "slapo_default_stage_3",
    }


@hydra_main_multistage(
    version_base=None, config_path=f"{os.getcwd()}/experiments/configs", config_name=SLAPOSegmentationExperiment.config
)
def main(cfg: DictConfig, stage: str):
    experiment = SLAPOSegmentationExperiment()
    add_legacy_features_to_vector_wrapper()
    getattr(experiment, stage)(cfg)


if __name__ == "__main__":
    main()
