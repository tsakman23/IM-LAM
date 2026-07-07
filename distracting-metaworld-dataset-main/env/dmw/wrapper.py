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
    """MujocoRenderer subclass that supports segmentation rendering.

    Overrides the default rendering path for segmentation to work around a
    numpy 2.x incompatibility in gymnasium's ``OffScreenViewer``, where
    ``uint8 * (2**8)`` raises ``OverflowError`` instead of promoting the
    dtype (see gymnasium ``mujoco_rendering.py``).
    """

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

        if render_mode in ["rgb_array", "depth_array"] and segmentation:
            return self._render_segmentation(viewer, render_mode)
        elif render_mode in ["rgb_array", "depth_array"]:
            return viewer.render(
                render_mode=render_mode,
                camera_id=self.camera_id,
                segmentation=False,
            )
        elif render_mode == "human":
            return viewer.render()

    def _render_segmentation(self, viewer, render_mode: str) -> np.ndarray:
        """Render a segmentation map, fixing the uint8 overflow in gymnasium.

        This replicates the logic of ``OffScreenViewer.render`` with
        ``segmentation=True`` but casts the raw pixel data to ``int32``
        before computing segmentation IDs so that the ``* (2**8)`` and
        ``* (2**16)`` multiplications do not overflow on numpy >= 2.0.
        """
        # --- camera setup (same as OffScreenViewer.render) ---
        if self.camera_id is not None:
            if self.camera_id == -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = self.camera_id

        mujoco.mjv_updateScene(
            viewer.model, viewer.data, viewer.vopt, viewer.pert,
            viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.scn,
        )

        # Enable segmentation flags
        viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        viewer.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        for marker_params in viewer._markers:
            viewer._add_marker_to_scene(marker_params)

        mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.con)

        for gridpos, (text1, text2) in viewer._overlays.items():
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150, gridpos,
                viewer.viewport, text1.encode(), text2.encode(), viewer.con,
            )

        # Disable segmentation flags
        viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
        viewer.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

        # Read raw pixels
        rgb_arr = np.zeros(
            3 * viewer.viewport.width * viewer.viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            viewer.viewport.width * viewer.viewport.height, dtype=np.float32
        )
        mujoco.mjr_readPixels(rgb_arr, depth_arr, viewer.viewport, viewer.con)

        if render_mode == "depth_array":
            depth_img = depth_arr.reshape(
                (viewer.viewport.height, viewer.viewport.width)
            )
            viewer._markers.clear()
            return depth_img[::-1, :]

        rgb_img = rgb_arr.reshape(
            (viewer.viewport.height, viewer.viewport.width, 3)
        )
        rgb_img = rgb_img[::-1, :]

        # FIX: cast to int32 before multiplication to avoid uint8 overflow
        seg_img = (
            rgb_img[:, :, 0].astype(np.int32)
            + rgb_img[:, :, 1].astype(np.int32) * (2**8)
            + rgb_img[:, :, 2].astype(np.int32) * (2**16)
        )
        seg_img[seg_img >= (viewer.scn.ngeom + 1)] = 0
        seg_ids = np.full(
            (viewer.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32
        )

        for i in range(viewer.scn.ngeom):
            geom = viewer.scn.geoms[i]
            if geom.segid != -1:
                seg_ids[geom.segid + 1, 0] = geom.objtype
                seg_ids[geom.segid + 1, 1] = geom.objid

        viewer._markers.clear()
        return seg_ids[seg_img]


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

        # Dedicated RNG for background selection – completely separate from the
        # environment's physics np_random so that seeding never affects physics.
        self._bg_rng = np.random.default_rng()

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
        """Select a new random video and load its frames as background images.

        Uses the dedicated ``_bg_rng`` (never the env's physics np_random) so
        that background selection does not disturb the physics random state.
        """
        # Randomly pick a video and load all images.
        video_idx = int(self._bg_rng.integers(0, self.num_videos))
        video_path = self._video_paths[video_idx]
        file_names = listdir(video_path)

        if not self._dynamic:
            # Randomly pick a single static frame.
            fn_idx = int(self._bg_rng.integers(0, len(file_names)))
            file_names = [file_names[fn_idx]]

        images = [imread(os.path.join(video_path, fn)) for fn in file_names]

        # Resize images to match render size.
        self._images = [
            resize_image(img, self._height, self._width) for img in images
        ]

        # Pick a random starting point and stepping direction.
        self._curr_frame = int(self._bg_rng.integers(0, len(self._images)))
        self._direction = int(self._bg_rng.choice([-1, 1]))

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
            # Seed the dedicated background RNG deterministically from the
            # episode seed (offset to avoid correlation with physics seed).
            # This is completely decoupled from the env's physics np_random,
            # so vanilla / distracting / segmentation envs all end up with
            # identical physics state after reset(seed=seed).
            if seed is not None:
                self._bg_rng = np.random.default_rng(seed + 0xDEAD_BEEF)
            self.__reset_background()

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
        self.unwrapped.seeded_rand_vec = True
        self.unwrapped._freeze_rand_vec = False

        self._robot_geom_ids = self._get_robot_geom_ids()
        self._object_geom_ids = self._get_object_geom_ids(
            object_body_names=object_body_names, object_geom_ids=object_geom_ids
        )

    def _get_robot_geom_ids(self) -> list[int]:
        """Identify geom IDs belonging to the robot arm's kinematic chain.

        Traverses the MuJoCo body tree starting from the robot's root body
        (the first child of the world body whose name contains ``'base'``)
        and collects all geom IDs attached to the robot's bodies.

        Returns:
            A sorted list of integer geom IDs belonging to the robot arm.
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

        # BFS to collect all body IDs in the robot's kinematic chain,
        # pruning subtrees rooted at task-object bodies.
        robot_body_ids: set[int] = set()
        if robot_root is not None:
            queue = [robot_root]
            while queue:
                body_id = queue.pop(0)
                robot_body_ids.add(body_id)
                queue.extend(children[body_id])

        # Map robot body IDs to their geom IDs.
        robot_geom_ids = sorted(
            gid for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) in robot_body_ids
        )
        return robot_geom_ids

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

    def _get_robot_body_ids(self) -> set[int]:
        model = self.unwrapped.model
        children: dict[int, list[int]] = {i: [] for i in range(model.nbody)}
        for i in range(1, model.nbody):
            children[int(model.body_parentid[i])].append(i)
        robot_root = None
        for child_id in children[0]:
            if "base" in model.body(child_id).name.lower():
                robot_root = child_id
                break
        if robot_root is None:
            for child_id in children[0]:
                if children[child_id]:
                    robot_root = child_id
                    break
        robot_body_ids: set[int] = set()
        if robot_root is not None:
            queue = [robot_root]
            while queue:
                body_id = queue.pop(0)
                robot_body_ids.add(body_id)
                queue.extend(children[body_id])
        return robot_body_ids

    def _is_movable_wrt_world(self, body_id: int) -> bool:
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
        model = self.unwrapped.model
        if object_geom_ids is not None:
            return sorted(int(g) for g in object_geom_ids)
        if object_body_names is not None:
            wanted = set(object_body_names)
            body_ids = {b for b in range(model.nbody) if model.body(b).name in wanted}
        else:
            robot_body_ids = self._get_robot_body_ids()
            body_ids = {
                b for b in range(1, model.nbody)
                if b not in robot_body_ids and self._is_movable_wrt_world(b)
            }
        return sorted(
            gid for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) in body_ids
        )

    @staticmethod
    def _binary_to_image(mask: np.ndarray) -> np.ndarray:
        img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        img[mask] = 255
        return img

    def render_masks(self) -> dict[str, np.ndarray]:
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

