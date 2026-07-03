from lightning_utilities.core.imports import RequirementCache

_IS_MINIGRID_AVAILABLE = RequirementCache("minigrid==2.3.1")
_IS_PROCGEN_AVAILABLE = RequirementCache("procgen==0.10.7")
_IS_VIZDOOM_AVAILABLE = RequirementCache("vizdoom==1.2.4")
_IS_MINERL_AVAILABLE = RequirementCache("minerl @ git+https://github.com/minerllabs/minerl@v1.0.2")
_IS_DM_CONTROL_AVAILABLE = RequirementCache("dm_control==1.0.37")
_IS_ATARI_AVAILABLE = RequirementCache("ale-py==0.11.2")
_IS_METAWORLD_AVAILABLE = RequirementCache("metaworld==3.0.0")
