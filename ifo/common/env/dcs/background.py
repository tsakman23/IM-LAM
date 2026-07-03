# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A wrapper for dm_control environments which applies color distractions."""

from ifo.common.utils.imports import _IS_DM_CONTROL_AVAILABLE

if not _IS_DM_CONTROL_AVAILABLE:
    raise ModuleNotFoundError(_IS_DM_CONTROL_AVAILABLE)

import collections
import glob
import os

import numpy as np

# import tensorflow as tf
from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control
from PIL import Image

# for compatability with old experiments
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
SKY_TEXTURE_INDEX = 0
Texture = collections.namedtuple("Texture", ("size", "address", "textures"))


def _get_texture_buffer(model):
    """Return the underlying Mujoco texture buffer for the given model.

    Different Mujoco / dm_control versions expose different attributes for
    texture data (e.g. `tex_rgb`, `tex_rgba`, `tex_data`). This helper picks
    whichever is available so that the rest of the code can remain agnostic.
    """
    if hasattr(model, "tex_rgb"):
        return model.tex_rgb
    if hasattr(model, "tex_rgba"):
        return model.tex_rgba
    if hasattr(model, "tex_data"):
        return model.tex_data
    raise AttributeError("Mujoco model has no texture buffer attribute among "
                         "`tex_rgb`, `tex_rgba`, or `tex_data`.")


def imread(filename):
    img = Image.open(filename)
    img_np = np.asarray(img)
    return img_np


def size_and_flatten(image, ref_height, ref_width):
    # Resize image if necessary and flatten the result.
    image_height, image_width = image.shape[:2]

    if image_height != ref_height or image_width != ref_width:
        # image = tf.cast(tf.image.resize(image, [ref_height, ref_width]), tf.uint8)
        image = np.asarray(Image.fromarray(image).resize(size=(ref_width, ref_height)), dtype=np.uint8)

    return image.flatten(order="K")
    # return tf.reshape(image, [-1]).numpy()


def blend_to_background(alpha, image, background):
    if alpha == 1.0:
        return image
    elif alpha == 0.0:
        return background
    else:
        return (alpha * image.astype(np.float32) + (1.0 - alpha) * background.astype(np.float32)).astype(np.uint8)


def listdir(dir_path, filetype="jpg", sort=True):
    dir_path = os.path.join(dir_path, f"*.{filetype}")
    paths = glob.glob(dir_path, recursive=True)
    paths = list(map(lambda p: os.path.basename(p), paths))

    if sort:
        return sorted(paths)
    return paths


class DistractingBackgroundEnv(control.Environment):
    """Environment wrapper for background visual distraction.

    **NOTE**: This wrapper should be applied BEFORE the pixel wrapper to make sure
    the background image changes are applied before rendering occurs.
    """

    def __init__(
        self,
        env,
        dataset_path=None,
        dataset_videos=None,
        video_alpha=1.0,
        ground_plane_alpha=1.0,
        num_videos=None,
        dynamic=False,
        seed=None,
        shuffle_buffer_size=None,
    ):
        if not 0 <= video_alpha <= 1:
            raise ValueError("`video_alpha` must be in the range [0, 1]")

        self._env = env
        self._video_alpha = video_alpha
        self._ground_plane_alpha = ground_plane_alpha
        self._random_state = np.random.RandomState(seed=seed)
        self._dynamic = dynamic
        self._shuffle_buffer_size = shuffle_buffer_size
        self._background = None
        self._current_img_index = 0

        if not dataset_path or num_videos == 0:
            # Allow running the wrapper without backgrounds to still set the ground
            # plane alpha value.
            self._video_paths = []
        else:
            # Use all videos if no specific ones were passed.
            if not dataset_videos:
                # dataset_videos = sorted(tf.io.gfile.listdir(dataset_path))
                dataset_videos = sorted(listdir(dataset_path))

            # Replace video placeholders 'train'/'val' with the list of videos.
            elif dataset_videos in ["train", "training"]:
                dataset_videos = DAVIS17_TRAINING_VIDEOS
            elif dataset_videos in ["val", "validation"]:
                dataset_videos = DAVIS17_VALIDATION_VIDEOS

            # Get complete paths for all videos.
            video_paths = [os.path.join(dataset_path, subdir) for subdir in dataset_videos]

            # Optionally use only the first num_paths many paths.
            if num_videos is not None:
                if num_videos > len(video_paths) or num_videos < 0:
                    raise ValueError(
                        f"`num_bakground_paths` is {num_videos} but "
                        "should not be larger than the number of available "
                        f"background paths ({len(video_paths)}) and at "
                        "least 0."
                    )
                video_paths = video_paths[:num_videos]

            self._video_paths = video_paths

    def reset(self):
        """Reset the background state."""
        time_step = self._env.reset()
        self._reset_background()
        return time_step

    def _reset_background(self):
        # Make grid semi-transparent.
        if self._ground_plane_alpha is not None:
            self._env.physics.named.model.mat_rgba["grid", "a"] = self._ground_plane_alpha

        # For some reason the height of the skybox is set to 4800 by default,
        # which does not work with new textures.
        self._env.physics.model.tex_height[SKY_TEXTURE_INDEX] = 800

        # Set the sky texture reference.
        physics_model = self._env.physics.model
        sky_height = physics_model.tex_height[SKY_TEXTURE_INDEX]
        sky_width = physics_model.tex_width[SKY_TEXTURE_INDEX]
        nchannel = int(physics_model.tex_nchannel[SKY_TEXTURE_INDEX])
        sky_size = sky_height * sky_width * nchannel
        sky_address = physics_model.tex_adr[SKY_TEXTURE_INDEX]

        # Grab the full underlying texture (all channels) as float32.
        texture_buffer = _get_texture_buffer(physics_model)
        sky_texture_all = texture_buffer[sky_address : sky_address + sky_size].astype(np.float32)

        # Work in RGB space for blending; keep any extra channels (e.g. alpha)
        # untouched and restore them later when writing back.
        n_rgb = min(3, nchannel)  # typically 3 or 4 channels.
        sky_texture_all_reshaped = sky_texture_all.reshape(-1, nchannel)
        sky_texture_rgb = sky_texture_all_reshaped[:, :n_rgb].reshape(-1)

        if self._video_paths:
            if self._shuffle_buffer_size:
                # Shuffle images from all videos together to get background frames.
                file_names = [
                    os.path.join(path, fn)
                    for path in self._video_paths
                    for fn in listdir(path)
                    # for fn in tf.io.gfile.listdir(path)
                ]
                self._random_state.shuffle(file_names)
                # Load only the first n images for performance reasons.
                file_names = file_names[: self._shuffle_buffer_size]
                images = [imread(fn) for fn in file_names]
            else:
                # Randomly pick a video and load all images.
                video_path = self._random_state.choice(self._video_paths)
                # file_names = tf.io.gfile.listdir(video_path)
                file_names = listdir(video_path)

                if not self._dynamic:
                    # Randomly pick a single static frame.
                    file_names = [self._random_state.choice(file_names)]
                images = [imread(os.path.join(video_path, fn)) for fn in file_names]

            # Pick a random starting point and steping direction.
            self._current_img_index = self._random_state.choice(len(images))
            self._step_direction = self._random_state.choice([-1, 1])

            # Prepare images in the texture format by resizing and flattening.

            # Generate image textures.
            texturized_images = []
            for image in images:
                image_flattened = size_and_flatten(image, sky_height, sky_width)

                # Ensure the image has the same number of RGB channels we
                # operate on (n_rgb). If the source image is RGB this is a
                # no-op; if it has a different number of channels we fall back
                # to taking / repeating the first channel(s).
                image_flattened = image_flattened.reshape(-1, 3)
                if n_rgb == 1:
                    image_rgb = image_flattened[:, :1]
                elif n_rgb == 2:
                    image_rgb = image_flattened[:, :2]
                else:
                    image_rgb = image_flattened[:, :3]
                image_rgb = image_rgb.reshape(-1)

                base_rgb = sky_texture_rgb
                if base_rgb.size != image_rgb.size:
                    raise ValueError(
                        "Background image and sky texture must have matching "
                        "RGB sizes. Got image size "
                        f"{image_rgb.size}, sky size {base_rgb.size}."
                    )

                new_rgb = blend_to_background(self._video_alpha, image_rgb, base_rgb)

                # Reconstruct full texture (including any extra channels) from
                # the blended RGB and the original non-RGB channels.
                if nchannel > n_rgb:
                    extra_channels = sky_texture_all_reshaped[:, n_rgb:]
                    merged = np.concatenate(
                        [
                            new_rgb.reshape(-1, n_rgb),
                            extra_channels,
                        ],
                        axis=1,
                    )
                    new_texture_full = merged.reshape(-1).astype(np.uint8)
                else:
                    new_texture_full = new_rgb.astype(np.uint8)

                texturized_images.append(new_texture_full)

        else:
            self._current_img_index = 0
            # If no video backgrounds are provided, keep the original texture.
            texturized_images = [sky_texture_all.astype(np.uint8)]

        self._background = Texture(sky_size, sky_address, texturized_images)
        self._apply()

    def step(self, action):
        time_step = self._env.step(action)

        if time_step.first():
            self._reset_background()
            return time_step

        if self._dynamic and self._video_paths:
            # Move forward / backward in the image sequence by updating the index.
            self._current_img_index += self._step_direction

            # Start moving forward if we are past the start of the images.
            if self._current_img_index <= 0:
                self._current_img_index = 0
                self._step_direction = abs(self._step_direction)
            # Start moving backwards if we are past the end of the images.
            if self._current_img_index >= len(self._background.textures):
                self._current_img_index = len(self._background.textures) - 1
                self._step_direction = -abs(self._step_direction)

            self._apply()
        return time_step

    def _apply(self):
        """Apply the background texture to the physics."""

        if self._background:
            start = self._background.address
            end = self._background.address + self._background.size
            texture = self._background.textures[self._current_img_index]

            physics_model = self._env.physics.model
            texture_buffer = _get_texture_buffer(physics_model)
            texture_buffer[start:end] = texture

            # Upload the new texture to the GPU. Note: we need to make sure that the
            # OpenGL context belonging to this Physics instance is the current one.
            with self._env.physics.contexts.gl.make_current() as ctx:
                ctx.call(
                    mjbindings.mjlib.mjr_uploadTexture,
                    self._env.physics.model.ptr,
                    self._env.physics.contexts.mujoco.ptr,
                    SKY_TEXTURE_INDEX,
                )

    # Forward property and method calls to self._env.
    def __getattr__(self, attr):
        if hasattr(self._env, attr):
            return getattr(self._env, attr)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))
