import os

import hydra
from gymnasium import spaces
from omegaconf import DictConfig

from datasets import Array2D, Array3D, Array4D, Array5D, Features, Image, Sequence, Value
from ifo.common.experiment import Experiment
from ifo.common.utils.hydra import hydra_main_multistage
from ifo.common.utils.utility import Conditional, add_legacy_features_to_vector_wrapper


class DatasetExperiment(Experiment):
    """Class for dataset generation experiment.

    Generates a dataset for Hugging Face (https://huggingface.co/datasets/).
    """

    stages = ["stage_1"]
    config = {
        "stage_1": "ppo_default",
    }

    def _create_action_space_feature(self, action_space: spaces.Space):
        """Create a Features object for the action space.

        Args:
            action_space (spaces.Space): The action space to create a Features object for.

        Returns:
            Features: A Features object for the action space.
        """
        dim_to_feature = {
            1: lambda shape, dtype: Sequence(Value(dtype=dtype), length=shape[0]),
            2: Array2D,
            3: Array3D,
            4: Array4D,
            5: Array5D,
        }
        if isinstance(action_space, spaces.Box):
            shape = action_space.shape
            return dim_to_feature[len(shape)](shape, "float32")
        elif isinstance(action_space, spaces.Discrete):
            return Value(dtype="int32")
        elif isinstance(action_space, spaces.MultiDiscrete):
            shape = action_space.shape
            return dim_to_feature[len(shape)](shape, "int32")
        elif isinstance(action_space, spaces.Dict):
            return Features({k: self._create_action_space_feature(v) for k, v in action_space.spaces.items()})
        else:
            raise ValueError(f"Unsupported action space: {action_space}")

    def stage_1(self, cfg: DictConfig) -> None:
        """Generate dataset."""
        self._print_config(cfg)

        # Check for Hugging Face config entry needed for dataset generation.
        if "huggingface_dataset" not in cfg:
            raise ValueError("Hugging Face config entry needed for dataset generation.")

        # Initialize Fabric.
        cfg.fabric.devices = 1  # Disable multi-GPU training.
        fabric = self._configure_fabric(cfg)

        # Configure environment.
        env = hydra.utils.instantiate(cfg.env, num_envs=1, preserve_original=True)
        action_space = env.single_action_space
        observation_space = env.single_observation_space
        original_observation_space = env.get_wrapper_attr("original_single_observation_space")
        original_action_space = env.get_wrapper_attr("original_single_action_space")
        del env

        # Create features for the dataset.
        assert isinstance(
            original_observation_space, spaces.Box
        ), "Only image observations are supported for dataset generation."
        features = Features(
            observation=Image(mode="RGB"),
            action=self._create_action_space_feature(original_action_space),
            reward=Value(dtype="float32"),
            terminated=Value(dtype="bool"),
            truncated=Value(dtype="bool"),
        )

        # Initialize model.
        net = hydra.utils.instantiate(cfg.net, _partial_=True)(
            observation_space=observation_space, action_space=action_space
        )
        module = hydra.utils.instantiate(cfg.module, net=net, _recursive_=False)

        # Initialize recorder and start recording.
        recorder = hydra.utils.instantiate(
            cfg.huggingface_dataset.recorder,
            fabric=fabric,
            compile=cfg.trainer.compile,
            random_seed=cfg.trainer.random_seed,
            _partial_=True,
        )
        recorder = recorder(dataset_features=features)

        # To create a dataset, we need to load the checkpoint.
        assert cfg.trainer.checkpoint_path is not None, "Checkpoint path is required for dataset creation."

        recorder.create_dataset(
            model=module,
            env_cfg=cfg.env,
            checkpoint=cfg.trainer.checkpoint_path,
        )


@hydra_main_multistage(
    version_base=None, config_path=f"{os.getcwd()}/experiments/configs", config_name=DatasetExperiment.config
)
def main(cfg: DictConfig, stage: str):
    experiment = DatasetExperiment()
    add_legacy_features_to_vector_wrapper()
    getattr(experiment, stage)(cfg)


if __name__ == "__main__":
    main()
