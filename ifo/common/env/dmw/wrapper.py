import glob
import os
from typing import Any, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from PIL import Image

SKY_SEGMENTATION_INDEXES = [-2, 5]

# for compatability with Distracting Control Suite
DAVIS17_TRAINING_VIDEOS = [
    "bear",
    "bmx-bumps",
    "boat",
    "boxing-fisheye",
    "breakdance-flare",
    "bus",
    "car-turn",
    "cat-girl",
    "classic-car",
    "color-run",
    "crossing",
    "dance-jump",
    "dancing",
    "disc-jockey",
    "dog-agility",
    "dog-gooses",
    "dogs-scale",
    "drift-turn",
    "drone",
    "elephant",
    "flamingo",
    "hike",
    "hockey",
    "horsejump-low",
    "kid-football",
    "kite-walk",
    "koala",
    "lady-running",
    "lindy-hop",
    "longboard",
    "lucia",
    "mallard-fly",
    "mallard-water",
    "miami-surf",
    "motocross-bumps",
    "motorbike",
    "night-race",
    "paragliding",
    "planes-water",
    "rallye",
    "rhino",
    "rollerblade",
    "schoolgirls",
    "scooter-board",
    "scooter-gray",
    "sheep",
    "skate-park",
    "snowboard",
    "soccerball",
    "stroller",
    "stunt",
    "surf",
    "swing",
    "tennis",
    "tractor-sand",
    "train",
    "tuk-tuk",
    "upside-down",
    "varanus-cage",
    "walking",
]

DAVIS17_VALIDATION_VIDEOS = [
    "bike-packing",
    "blackswan",
    "bmx-trees",
    "breakdance",
    "camel",
    "car-roundabout",
    "car-shadow",
    "cows",
    "dance-twirl",
    "dog",
    "dogs-jump",
    "drift-chicane",
    "drift-straight",
    "goat",
    "gold-fish",
    "horsejump-high",
    "india",
    "judo",
    "kite-surf",
    "lab-coat",
    "libby",
    "loading",
    "mbike-trick",
    "motocross-jump",
    "paragliding-launch",
    "parkour",
    "pigs",
    "scooter-black",
    "shooting",
    "soapbox",
]


def imread(filename: str) -> np.ndarray:
    """Read an image file and return it as a numpy array.

    Args:
        filename: Path to the image file to read.

    Returns:
        The image as a numpy array with shape (H, W, C).
    """
    img = Image.open(filename)
    img_np = np.asarray(img)
    return img_np


def resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an image to the target height and width.

    If the image already matches the target dimensions, it is returned unchanged.

    Args:
        image: Source image array with shape (H, W, ...).
        height: Desired output height in pixels.
        width: Desired output width in pixels.

    Returns:
        The resized image as a uint8 numpy array.
    """
    image_height, image_width = image.shape[:2]
    if image_height != height or image_width != width:
        image = np.asarray(
            Image.fromarray(image).resize(size=(width, height)),
            dtype=np.uint8,
        )
    return image


def listdir(dir_path: str, filetype: str = "jpg", sort: bool = True) -> list[str]:
    """List filenames of a given type in a directory.

    Args:
        dir_path: Path to the directory to scan.
        filetype: File extension to filter by (without the leading dot).
        sort: Whether to return the filenames in sorted order.

    Returns:
        A list of matching filenames (basenames only, not full paths).
    """
    pattern = os.path.join(dir_path, f"*.{filetype}")
    paths = glob.glob(pattern, recursive=True)
    paths = list(map(lambda p: os.path.basename(p), paths))
    if sort:
        return sorted(paths)
    return paths


def blend(image: np.ndarray, background: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend an image with a background.

    Computes ``alpha * image + (1 - alpha) * background``.

    Args:
        image: Foreground image array.
        background: Background image array (must be broadcastable to *image*).
        alpha: Blending weight for the foreground (0.0 = fully background,
            1.0 = fully foreground).

    Returns:
        The blended image as a uint8 numpy array.
    """
    return (
        alpha * image.astype(np.float32) + (1.0 - alpha) * background.astype(np.float32)
    ).astype(np.uint8)


class MujocoRendererSegm(MujocoRenderer):
    """MujocoRenderer subclass that supports segmentation rendering."""

    def render(
        self, render_mode: Optional[str], segmentation: bool = False
    ) -> Optional[np.ndarray]:
        """Render the scene, optionally producing a segmentation map.

        Args:
            render_mode: One of ``"rgb_array"``, ``"depth_array"``, or ``"human"``.
            segmentation: If ``True``, return a segmentation map instead of an
                RGB image (only used for ``"rgb_array"`` / ``"depth_array"``).

        Returns:
            A numpy array for off-screen modes, or ``None`` for ``"human"`` mode.
        """
        viewer = self._get_viewer(render_mode=render_mode)

        if render_mode in ["rgb_array", "depth_array"]:
            return viewer.render(
                render_mode=render_mode,
                camera_id=self.camera_id,
                segmentation=segmentation,
            )
        elif render_mode == "human":
            return viewer.render()


class DistractingMetaworldWrapper(gym.Wrapper):
    """Gymnasium wrapper that adds visual distractors to Metaworld environments.

    Background regions of the rendered image (identified via segmentation) are
    replaced with frames from DAVIS-2017 videos, producing visually
    distracting observations for robust policy learning.
    """

    def __init__(
        self,
        env: gym.Env,
        dataset_path: str,
        dataset_videos: str | list[str] | None = "train",
        dynamic: bool = True,
        blend_alpha: float = 0.0,
        # for vanilla setup & debugging
        disable_distractors: bool = False,
    ) -> None:
        """Initialise the distracting wrapper.

        Args:
            env: The base Metaworld Gymnasium environment to wrap.
            dataset_path: Root directory containing DAVIS-2017 video folders.
            dataset_videos: Which video split to use. ``"train"`` /
                ``"training"`` for the training set, ``"val"`` / ``"validation"``
                for the validation set, a list of video names, or ``None`` to
                use every sub-directory found under *dataset_path*.
            dynamic: If ``True``, step through video frames over time; if
                ``False``, use a single randomly chosen static frame.
            blend_alpha: Alpha weight for blending the original pixels with the
                background (0.0 = full background replacement,
                1.0 = original pixels only).
            disable_distractors: If ``True``, skip all distractor logic
                (useful for vanilla baselines and debugging).
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            dataset_path=dataset_path,
            dataset_videos=dataset_videos,
            dynamic=dynamic,
            blend_alpha=blend_alpha,
        )
        gym.Wrapper.__init__(self, env)

        self.blend_alpha = blend_alpha
        self._dynamic = dynamic
        self._render_mode = env.render_mode
        self._height = env.unwrapped.height
        self._width = env.unwrapped.width
        self._camera_id = env.unwrapped.camera_name
        self.disable_distractors = disable_distractors
        self._rot_180 = self._camera_id not in ["behindGripper", "gripperPOV"]

        self.unwrapped.mujoco_renderer = MujocoRendererSegm(
            env.unwrapped.model,
            env.unwrapped.data,
            camera_name=self._camera_id,
            height=self._height,
            width=self._width,
        )
        # see: https://github.com/Farama-Foundation/Metaworld/pull/370
        # WARN: for some reason this may lead to change of goal on second state after the reset!
        self.unwrapped.seeded_rand_vec = True
        self.unwrapped._freeze_rand_vec = False

        self._images = None
        self._curr_frame = None
        self._direction = None

        if not self.disable_distractors:
            # Use all videos if no specific ones were passed.
            if not dataset_videos:
                dataset_videos = sorted(listdir(dataset_path))
            elif dataset_videos in ["train", "training"]:
                dataset_videos = DAVIS17_TRAINING_VIDEOS
            elif dataset_videos in ["val", "validation"]:
                dataset_videos = DAVIS17_VALIDATION_VIDEOS

            # Get complete paths for all video directories.
            self._video_paths = [
                os.path.join(dataset_path, subdir) for subdir in dataset_videos
            ]
            self.num_videos = len(self._video_paths)
            assert self.num_videos > 0

    def __reset_background(self) -> None:
        """Select a new random video and load its frames as background images."""
        # Randomly pick a video and load all images.
        video_idx = int(self.env.np_random.integers(0, self.num_videos))
        video_path = self._video_paths[video_idx]
        file_names = listdir(video_path)

        if not self._dynamic:
            # Randomly pick a single static frame.
            fn_idx = int(self.env.np_random.integers(0, len(file_names)))
            file_names = [file_names[fn_idx]]

        images = [imread(os.path.join(video_path, fn)) for fn in file_names]

        # Resize images to match render size.
        self._images = [
            resize_image(img, self._height, self._width) for img in images
        ]

        # Pick a random starting point and stepping direction.
        self._curr_frame = int(self.env.np_random.integers(0, len(self._images)))
        self._direction = int(self.env.np_random.choice([-1, 1]))

    def __step_background(self) -> None:
        """Advance the background frame index, bouncing at sequence boundaries."""
        # Move forward / backward in the image sequence by updating the index.
        self._curr_frame += self._direction

        # Start moving forward if we are past the start of the images.
        if self._curr_frame <= 0:
            self._curr_frame = 0
            self._direction = abs(self._direction)
        # Start moving backwards if we are past the end of the images.
        if self._curr_frame >= len(self._images) - 1:
            self._curr_frame = len(self._images) - 1
            self._direction = -abs(self._direction)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and select a new distractor background.

        Args:
            seed: Optional random seed forwarded to the base environment.
            options: Optional reset options forwarded to the base environment.

        Returns:
            A ``(observation, info)`` tuple.
        """
        if not self.disable_distractors:
            # Backup random state in order to not alter environment random state.
            rng_state = self.env.np_random.bit_generator.state
            self.__reset_background()
            self.env.np_random.bit_generator.state = rng_state
        obs, info = self.env.reset(seed=seed, options=options)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take one environment step, advancing the distractor background if dynamic.

        Args:
            action: Action to execute in the environment.

        Returns:
            A ``(observation, reward, terminated, truncated, info)`` tuple.
        """
        if not self.disable_distractors and self._dynamic:
            self.__step_background()
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def _render(self, segmentation: bool = False) -> np.ndarray:
        """Render an off-screen image via the custom MujocoRendererSegm.

        Args:
            segmentation: If ``True``, return a segmentation map instead of
                an RGB image.

        Returns:
            The rendered image or segmentation array.
        """
        return self.unwrapped.mujoco_renderer.render(
            render_mode="rgb_array", segmentation=segmentation
        )

    def render(self) -> np.ndarray:
        """Render the current frame with distractor backgrounds composited in.

        Background pixels (identified by segmentation) are replaced with the
        current video frame, blended according to ``blend_alpha``.

        Returns:
            The composited RGB image as a uint8 numpy array of shape
            ``(height, width, 3)``.
        """
        img_array = self._render(segmentation=False)
        img_array = np.rot90(img_array, 2) if self._rot_180 else img_array
        if not self.disable_distractors:
            segm = self._render(segmentation=True).sum(axis=2)
            segm = np.rot90(segm, 2) if self._rot_180 else segm

            background_array = self._images[self._curr_frame]
            mask = sum([segm == segm_id for segm_id in SKY_SEGMENTATION_INDEXES])
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(bool)
            img_array[mask] = blend(
                img_array[mask], background_array[mask], alpha=self.blend_alpha
            )
        return img_array


class SegmentationMetaworldWrapper(gym.Wrapper):
    """Gymnasium wrapper that modifies the rendering of Metaworld environments to output a segmentation map.

    The robot arm is isolated in the segmentation map, while the background is black.
    """

    def __init__(
        self,
        env: gym.Env,
        object_body_names: Optional[list[str]] = None,
        object_geom_ids: Optional[list[int]] = None,
    ) -> None:
        """Initialise the segmentation wrapper.

        Args:
            env: The base Metaworld Gymnasium environment to wrap.
            object_body_names: Optional explicit list of MuJoCo body names whose
                geoms make up the manipulated object. Overrides the automatic
                movable-body heuristic used by :meth:`_get_object_geom_ids`.
            object_geom_ids: Optional explicit list of geom IDs for the
                manipulated object. Takes precedence over everything else.
        """
        super().__init__(env)
        self._render_mode = env.render_mode
        self._height = env.unwrapped.height
        self._width = env.unwrapped.width
        self._camera_id = env.unwrapped.camera_name
        self._rot_180 = self._camera_id not in ["behindGripper", "gripperPOV"]

        self.unwrapped.mujoco_renderer = MujocoRendererSegm(
            env.unwrapped.model,
            env.unwrapped.data,
            camera_name=self._camera_id,
            height=self._height,
            width=self._width,
        )
        # see: https://github.com/Farama-Foundation/Metaworld/pull/370
        # WARN: for some reason this may lead to change of goal on second state after the reset!
        self.unwrapped.seeded_rand_vec = True
        self.unwrapped._freeze_rand_vec = False


        # Pre-compute geom IDs belonging to the robot arm's kinematic chain
        # so that render() can isolate only the arm in the segmentation image.
        self._robot_geom_ids = self._get_robot_geom_ids()

        # Pre-compute geom IDs of the manipulated object(s) for the object mask.
        self._object_geom_ids = self._get_object_geom_ids(
            object_body_names=object_body_names, object_geom_ids=object_geom_ids
        )

    def _get_robot_body_ids(self) -> set[int]:
        """Body IDs of the robot arm's kinematic chain.

        Traverses the MuJoCo body tree starting from the robot's root body
        (the first child of the world body whose name contains ``'base'``) and
        collects every body in that subtree. Manipulated objects and static
        scene bodies (table, walls, fixtures) hang off the world body rather
        than off the robot, so they are naturally excluded.

        Returns:
            The set of body IDs belonging to the robot arm.
        """
        model = self.unwrapped.model

        # Build a parent -> children map for the body tree.
        children: dict[int, list[int]] = {i: [] for i in range(model.nbody)}
        for i in range(1, model.nbody):
            children[int(model.body_parentid[i])].append(i)

        # Find the robot root body (child of world body 0 named 'base').
        robot_root = None
        for child_id in children[0]:
            if "base" in model.body(child_id).name.lower():
                robot_root = child_id
                break
        # Fallback: first child of world that has its own children.
        if robot_root is None:
            for child_id in children[0]:
                if children[child_id]:
                    robot_root = child_id
                    break

        # BFS to collect all body IDs in the robot's kinematic chain.
        robot_body_ids: set[int] = set()
        if robot_root is not None:
            queue = [robot_root]
            while queue:
                body_id = queue.pop(0)
                robot_body_ids.add(body_id)
                queue.extend(children[body_id])

        return robot_body_ids

    def _get_robot_geom_ids(self) -> list[int]:
        """Geom IDs attached to the robot arm's kinematic chain.

        Returns:
            A sorted list of integer geom IDs belonging to the robot arm.
        """
        model = self.unwrapped.model
        robot_body_ids = self._get_robot_body_ids()
        return sorted(
            gid for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) in robot_body_ids
        )

    def _is_movable_wrt_world(self, body_id: int) -> bool:
        """Whether any joint lies on the path from ``body_id`` up to the world.

        Static scene bodies (table, walls, fixtures) are welded to the world and
        have no such joint; manipulated objects (free-joint pucks/pegs, hinged
        doors, sliding windows) do.
        """
        model = self.unwrapped.model
        b = int(body_id)
        while b != 0:
            if int(model.body_jntnum[b]) > 0:
                return True
            b = int(model.body_parentid[b])
        return False

    def _get_object_geom_ids(
        self,
        object_body_names: Optional[list[str]] = None,
        object_geom_ids: Optional[list[int]] = None,
    ) -> list[int]:
        """Identify geom IDs belonging to the manipulated object(s).

        Resolution order:
          1. explicit ``object_geom_ids`` if provided;
          2. geoms of the bodies named in ``object_body_names`` if provided;
          3. otherwise the automatic heuristic: every body that is not part of
             the robot chain and is movable relative to the world (see
             :meth:`_is_movable_wrt_world`). This selects the manipulated object
             and excludes the robot, table, walls, and other static fixtures.

        Returns:
            A sorted list of integer geom IDs belonging to the object(s).
        """
        model = self.unwrapped.model

        if object_geom_ids is not None:
            return sorted(int(g) for g in object_geom_ids)

        if object_body_names is not None:
            wanted = set(object_body_names)
            body_ids = {b for b in range(model.nbody) if model.body(b).name in wanted}
        else:
            robot_body_ids = self._get_robot_body_ids()
            body_ids = {
                b
                for b in range(1, model.nbody)
                if b not in robot_body_ids and self._is_movable_wrt_world(b)
            }

        return sorted(
            gid for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) in body_ids
        )

    def _render(self, segmentation: bool = False) -> np.ndarray:
        """Render an off-screen image via the custom MujocoRendererSegm.

        Args:
            segmentation: If ``True``, return a segmentation map instead of
                an RGB image.

        Returns:
            The rendered image or segmentation array.
        """
        return self.unwrapped.mujoco_renderer.render(
            render_mode="rgb_array", segmentation=segmentation
        )

    def render(self) -> np.ndarray:
        """Render a segmentation image of only the robot arm.

        Uses the pre-computed robot geom IDs (from the MuJoCo body tree) to
        isolate the arm. All non-arm pixels (table, sky, objects) are black.

        Returns:
            A binary segmentation image as a uint8 numpy array of shape
            ``(height, width, 3)`` where robot arm pixels are white (255)
            and all other pixels are black (0).
        """
        # segm shape: (H, W, 2) — channel 0 = object type, channel 1 = object ID
        segm = self._render(segmentation=True)
        segm = np.rot90(segm, 2) if self._rot_180 else segm

        # Match pixels whose geom ID (channel 1) belongs to the robot arm.
        geom_ids = segm[:, :, 1]
        arm_mask = np.isin(geom_ids, self._robot_geom_ids)

        # Morphological closing to fill isolated black pixels (artefacts)
        # inside the arm mask without changing the overall silhouette.
        #arm_mask = binary_closing(arm_mask, structure=np.ones((3, 3)))

        # Build a 3-channel binary segmentation image.
        img = np.zeros((segm.shape[0], segm.shape[1], 3), dtype=np.uint8)
        img[arm_mask] = 255

        return img

    def render_masks(self) -> dict[str, np.ndarray]:
        """Render binary agent and object segmentation masks in a single pass.

        Both masks come from one segmentation render, so they are mutually
        exclusive and occlusion-correct: each pixel is labelled by the
        front-most geom only. The agent mask is identical to what :meth:`render`
        produces. The object mask additionally requires the pixel's object type
        to be a geom, so that rendered sites (e.g. goal markers) cannot leak in.

        Returns:
            A dict with keys ``"agent"`` and ``"object"``, each a 3-channel
            uint8 array of shape ``(height, width, 3)`` where present pixels are
            255 and all others 0.
        """
        # segm shape: (H, W, 2) — channel 0 = object type, channel 1 = object ID.
        segm = self._render(segmentation=True)
        segm = np.rot90(segm, 2) if self._rot_180 else segm
        obj_type = segm[:, :, 0]
        geom_ids = segm[:, :, 1]

        agent_mask = np.isin(geom_ids, self._robot_geom_ids)
        is_geom = obj_type == mujoco.mjtObj.mjOBJ_GEOM
        object_mask = np.isin(geom_ids, self._object_geom_ids) & is_geom

        return {
            "agent": self._binary_to_image(agent_mask),
            "object": self._binary_to_image(object_mask),
        }

    @staticmethod
    def _binary_to_image(mask: np.ndarray) -> np.ndarray:
        """Convert a boolean (H, W) mask to a 3-channel uint8 image (255/0)."""
        img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        img[mask] = 255
        return img

