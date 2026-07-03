import asyncio
import json
import os
import random
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from copy import deepcopy
from queue import Empty, Queue
from threading import Barrier, Thread
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import aiohttp
import cv2
import numpy as np
import pyarrow as pa
import torch
from datasets.formatting import TorchFormatter
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import IterableDataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from ifo.common.transforms.data import DictTransform


class MineRLBasaltDataset(IterableDataset):
    """Expert trajectory dataset collected for the MineRL BASALT challenge.

    Link to challenge: https://www.aicrowd.com/challenges/submission-2022-minerl-basalt-competition
    Link to baseline repository: https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline/tree/main
    Link to dataset repository: https://github.com/openai/Video-Pre-Training

    Note: Each task needs at least 400GB of disk space. That means that all tasks together need at least 1.6TB of disk
    space to store the data.
    """

    index_file_ids = {
        "FindCave": "find-cave-Jul-28.json",
        "MakeWaterfall": "waterfall-Jul-28.json",
        "MakeVillageAnimalPen": "pen-animals-Jul-28.json",
        "BuildVillageHouse": "build-house-Jul-28.json",
        "Survival": "all_6xx_Jun_29.json",
        "SurvivalEarlyGameData": "all_7xx_Apr_6.json",
        "BuildHouseFromScratch": "all_9xx_jun_29.json",
        "BuildHouseFromRandomStartMaterials": "all_8xx_jun_29.json",
        "ObtainDiamondPickaxe": "all_10xx_Jun_29.json",
    }
    _base_url = "https://openaipublic.blob.core.windows.net/minecraft-rl/"
    _dir_index_files = "index_files"
    _dir_tasks = "tasks"

    # Action that does nothing in the environment.
    noop_action = {
        "ESC": 0,
        "back": 0,
        "drop": 0,
        "forward": 0,
        "hotbar.1": 0,
        "hotbar.2": 0,
        "hotbar.3": 0,
        "hotbar.4": 0,
        "hotbar.5": 0,
        "hotbar.6": 0,
        "hotbar.7": 0,
        "hotbar.8": 0,
        "hotbar.9": 0,
        "inventory": 0,
        "jump": 0,
        "left": 0,
        "right": 0,
        "sneak": 0,
        "sprint": 0,
        "swapHands": 0,
        "camera": [0, 0],
        "attack": 0,
        "use": 0,
        "pickItem": 0,
    }

    # Mapping of MineRL actions to keyboard buttons.
    keyboard_button_mapping = {
        "key.keyboard.escape": "ESC",
        "key.keyboard.s": "back",
        "key.keyboard.q": "drop",
        "key.keyboard.w": "forward",
        "key.keyboard.1": "hotbar.1",
        "key.keyboard.2": "hotbar.2",
        "key.keyboard.3": "hotbar.3",
        "key.keyboard.4": "hotbar.4",
        "key.keyboard.5": "hotbar.5",
        "key.keyboard.6": "hotbar.6",
        "key.keyboard.7": "hotbar.7",
        "key.keyboard.8": "hotbar.8",
        "key.keyboard.9": "hotbar.9",
        "key.keyboard.e": "inventory",
        "key.keyboard.space": "jump",
        "key.keyboard.a": "left",
        "key.keyboard.d": "right",
        "key.keyboard.left.shift": "sneak",
        "key.keyboard.left.control": "sprint",
        "key.keyboard.f": "swapHands",
    }

    # Maps mouse movement (dx, dy) to camera rotation in degrees.
    camera_scaler = 360.0 / 2400.0

    # Original recording resolution of the MineRL dataset.
    minerec_original_height_px = 720

    # Video resolution and frame rate of the MineRL dataset.
    video_resolution = (640, 360)
    frame_rate = 20

    # Path to the cursor image file.
    cursor_file = os.path.join("resources", "mouse_cursor_white_16x16.png")

    # Number of maximum connections for downloading data.
    # High enough for high speed, but low enough to not get blocked by OpenAI's servers.
    max_connections = 500

    def __init__(
        self,
        task_names: Optional[Union[List[str], str]] = None,
        split: str = "train",
        cache_dir: Optional[str] = None,
        num_proc: int = 8,
        block_size: int = 1,
        transform: Optional[DictTransform] = None,
        num_parallel_videos: int = 32,
        shuffle: bool = True,
        shuffle_seed: int = 1337,
        tensor_layout: str = "BTCHW",
        queue_timeout: float = 600.0,
        queue_size: int = 100,
        limit_num_tasks: Optional[int] = None,
        retry_download: Optional[bool] = False,
    ) -> None:
        """Initialize MineRLBasaltDataset.

        Args:
            task_names (Union[List[str], str], optional): List of names of the tasks that are suppposed to be loaded.
                If None, all tasks are loaded.
            split (str, optional): Split of the dataset to load. Defaults to "train".
            cache_dir (str, optional): Directory to read/write data. Defaults to
                "{WORKDIR}/cache/minerl_basalt_dataset".
            num_proc (int, optional): Number of processes to use for converting the dataset.
            block_size (int, optional): Sequence length of the dataset samples. Defaults to 1.
            transform (Callable, optional): Transform to apply to the dataset. Requires DictTransform!
            num_parallel_videos (int, optional): Number of videos to process in parallel. Defaults to 32.
                This allows to sample from multiple videos at the same time, which increases batch diversity.
                This should be set to `batch_size / num_workers` for decorellated samples.
            shuffle (bool, optional): If True, shuffle the videos before sampling. Defaults to True.
            shuffle_seed (int, optional): Seed for the random shuffle of tasks. Defaults to 1337.
            tensor_layout (str, optional): Format of the output tensors. Defaults to "BTCHW". Can also be "BTHWC".
            queue_timeout (float, optional): Timeout for the queue in seconds. Defaults to 100.0 seconds.
            queue_size (int, optional): Size of the sample queue. Defaults to 100 examples.
            limit_num_tasks (int, optional): Limit the number of tasks to load. Defaults to None and loads all tasks.
            retry_download (str, optional): Flag that decides whether downloads previously marked as failed in the metadata
                dictionary are tried to re-downloaded each time the Dataloader is called.
        """
        # If no tasks are provided, all tasks are loaded.
        if not task_names:
            task_names = self.index_file_ids.keys()
        # If a single task name is provided, convert it to a list.
        elif isinstance(task_names, str):
            task_names = [task_names]
        assert all(
            task_name in self.index_file_ids for task_name in task_names
        ), f"Task names {task_names} must be subset of {self.index_file_ids.keys()}"

        # If no cache directory is provided, use the current directory for caching.
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "cache")
        self.dataset_dir = os.path.join(cache_dir, "minerl_basalt_dataset")
        self.split = split
        self.num_proc = num_proc
        self.block_size = block_size
        self.num_parallel_videos = num_parallel_videos
        self.queue_size = queue_size
        self.queue_timeout = queue_timeout
        self.task_ids = {task_name: i for i, task_name in enumerate(self.index_file_ids.keys())}
        self.task_one_hot = torch.eye(len(self.task_ids))
        self.shuffle = shuffle
        self.length = 0
        self.limit_num_tasks = limit_num_tasks
        self.retry_download = retry_download
        self.formatter = TorchFormatter()

        # Check which tensor layout to use for observations.
        self.layout = {"BTCHW": self.permute, "BTHWC": self.identity}
        assert tensor_layout in self.layout.keys(), f"Tensor layout {tensor_layout} not supported."
        self.transform_layout = self.layout[tensor_layout]
        # Split up the transformations, so we can use them on single data points.
        self.transform_observation = self._get_transform(transform, "observation")
        self.transform_action = self._get_transform(transform, "action")
        self.transform_task = self._get_transform(transform, "task")
        self.schema = self._get_schema()

        self._setup_dir_structure(task_names)

        self.metadata_dict = {}
        self.tasks = []
        for task_name in task_names:
            print(f"###### Downloading and Preprocessing data for task {task_name} ######")
            index_file_id = self.index_file_ids[task_name]
            index_file_save_path = os.path.join(self.index_file_dir, index_file_id)
            task_save_dir = os.path.join(self.task_data_dir, task_name)
            self._download_data(index_file_id, index_file_save_path, task_save_dir)
            self._preprocess_data(index_file_save_path, task_save_dir)
            self._create_sampling_structure(task_name)

        # Set nice random seed for deterministic shuffling in multiprocessing.
        self.generator = random.Random(shuffle_seed)
        if self.shuffle:
            self.generator.shuffle(self.tasks)

    def _get_schema(self) -> pa.Schema:
        """Get the schema for the dataset.

        Returns:
            pa.Schema: The schema for the dataset.
        """
        schema = []
        for key in self.noop_action.keys():
            if key == "camera":
                schema.append((key, pa.list_(pa.float32())))
            else:
                schema.append((key, pa.int8()))
        return pa.schema(schema)

    def _get_transform(self, transforms: Optional[DictTransform], key: str) -> Callable:
        """Get the transform for a specific key.

        If no transforms are provided, the identity function is returned.

        Args:
            transforms (DictTransform): Dictionary of transformations.
            key (str): Key for the transformation.

        Returns:
            Callable: The transformation function.
        """
        if transforms:
            return transforms.transform.get(key, self.identity)
        return self.identity

    def permute(self, x: Tensor) -> Tensor:
        """Permute the image from HWC to CHW.

        Args:
            x (Tensor): Image tensor.

        Returns:
            Tensor: Permuted image tensor.
        """
        return x.permute(2, 0, 1)

    def identity(self, x: Tensor) -> Tensor:
        """Identity function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return x

    def _setup_dir_structure(self, task_names: List[str]) -> None:
        """Create the base structure of the dataset directory.

        The directory has the following structure and folders are only created if not yet existent:

        Folder structure:
             self.dataset_dir/
             ├── index_files/
             │   ├── ...
             ├── tasks/
             │   ├── taskName1/
             │   |   ├── test/
             │   |   |   ├── ...
             │   |   |   ├── ...
             │   |   ├── train/
             │   |   |   ├── ...
             │   |   |   ├── ...
             │   └── taskName2/
             │   |   ├── ...

         Args:
             task_names (List[str]): Tasks for which dataset directory is set up.
        """
        self.index_file_dir = os.path.join(self.dataset_dir, self._dir_index_files)
        self.task_data_dir = os.path.join(self.dataset_dir, self._dir_tasks)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.index_file_dir, exist_ok=True)
        os.makedirs(self.task_data_dir, exist_ok=True)

        for task_name in task_names:
            os.makedirs(os.path.join(self.task_data_dir, task_name), exist_ok=True)
            os.makedirs(os.path.join(self.task_data_dir, task_name, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.task_data_dir, task_name, "test"), exist_ok=True)

    def _download_data(self, index_file_id: str, index_file_save_path: str, task_save_dir: str) -> None:
        """Download video and action files for provided tasks.

        Downloads the video and action files required for the specified task. This method checks if an index file
        containing the URLs of all relevant video and action files for the given task exists in the cache directory.
        If the index file is missing, it downloads the index file. It then downloads any video or action files listed
        in the index that are not already present in the cache directory. Progress is logged in `metadata.json`.

        Args
            index_file_id (str): Identifier used to download index file from URL (e.g. "find-cave-Jul-28.json").
            index_file_save_path (str): Path to save index file to.
            task_save_dir (str): Directory to store video and action files to.

        Raises:
            Exception: If the downloading process encounters an error.
        """
        # Download index file if it does not already exist.
        if not os.path.exists(index_file_save_path):
            index_file_url = os.path.join(self._base_url, "snapshots", index_file_id)
            urlretrieve(index_file_url, index_file_save_path)
            self._preprocess_index_file(index_file_id, index_file_save_path)

        # Download video and action files for the specified task and log progress in metadata.json if the download is
        # interrupted.
        video_save_paths, action_save_paths = self._filter_downloaded_files(
            index_file_save_path, task_save_dir, "download_is_sucess"
        )

        if len(video_save_paths) + len(action_save_paths) > 0:
            try:
                asyncio.run(self._download_video_and_action_files(video_save_paths, action_save_paths))
            except Exception as e:
                print(f"Download failed {e}")
            finally:
                with open(os.path.join(task_save_dir, "metadata.json"), "w") as f:
                    json.dump(self.metadata_dict, f)

    def _preprocess_index_file(self, index_file_id: str, index_file_save_path: str) -> None:
        """Make sure that contractor demonstrations before version 6.13 are filtered from the index file

        Args
            index_file_id (str): Identifier used to download index file from URL (e.g. "find-cave-Jul-28.json").
            index_file_save_path (str): Path to save index file to.
        """
        if index_file_id == "all_6xx_Jun_29.json":
            with open(index_file_save_path, "r") as f:
                index_file_dict = json.load(f)
            index_file_dict["relpaths"] = [x for x in index_file_dict["relpaths"] if "data/6.13" in x]
            with open(index_file_save_path, "w") as f:
                json.dump(index_file_dict, f)

    async def _download_video_and_action_files(self, video_save_paths: List[str], action_save_paths: List[str]) -> None:
        """Concurrently download video and action files from provided urls to provided paths.

        Args:
            video_save_paths: save paths of video files to be downloaded
            action_save_paths: save paths of action files to be downloaded
        """
        connector = aiohttp.TCPConnector(limit_per_host=50, limit=self.max_connections)  # Limit connections
        async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(3600)) as session:
            download_tasks = []

            with (
                tqdm_asyncio(total=len(video_save_paths), desc="Downloading Videos", position=0) as video_pbar,
                tqdm_asyncio(total=len(action_save_paths), desc="Downloading JSONs", position=1) as json_pbar,
            ):
                for video_save_path in video_save_paths:
                    download_tasks.append(self._download_file(session, video_save_path, video_pbar))

                for json_save_path in action_save_paths:
                    download_tasks.append(self._download_file(session, json_save_path, json_pbar))

                # Wait for all downloads to complete
                await asyncio.gather(*download_tasks)

    async def _download_file(self, session: aiohttp.ClientSession, save_path: str, pbar: tqdm_asyncio) -> None:
        """Asynchronously downloads a file from the given URL and saves it to the specified path.

        Extracts the URL of the file to be downloaded from the save_path and downloads it into
        the same save_path. If the download fails, this is logged to the metadata.json file

        Args:
            session (aiohttp.ClientSession): The aiohttp session for the request.
            save_path (str): Local path where the video will be saved.
            pbar (tqdm_asyncio): Progress bar to update.

        """
        url = os.path.join(self.base_url, save_path.split("/")[-1])

        try:
            async with session.get(url) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Write the content to a file in chunks
                with open(save_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB
                        f.write(chunk)

            pbar.update(1)
            self.metadata_dict[save_path] = {
                "download_is_sucess": True,
                "preprocess_is_sucess": False,
            }

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            self.metadata_dict[save_path] = {
                "download_is_sucess": False,
                "preprocess_is_sucess": False,
            }
            pbar.update(1)

    def _handle_preprocess_keyboard_interrupt(self, futures: List[Future]) -> None:
        """Handle a KeyboardInterrupt during the multiprocessing preprocessing step.

        This method identifies which preprocessing futures are still running after a KeyboardInterrupt
        has been detected. This is necessary as only futures that are done have been handled correctly
        at this point, meaning their information was used to overwrite the metadata dictionary. Running
        futures can not be canceled and will therefore delete the corresponding video and action files.
        Thus it is necessary to wait for the termination of all running futures and to overwrite their
        information into the metadata dict. Pending futures are canceled.

        Args:
            futures (List[Future]): All futures from the ProcessPoolExcecutor
        """
        futures_to_handle = []
        for future in futures:
            if future.running():
                futures_to_handle.append(future)
            else:
                future.cancel()

        # Wait for running futures to terminate and handle their output (overwrite metadata dict)
        for future in as_completed(futures_to_handle):
            try:
                # Check if worker finished or was terminated prematurely.
                if result := future.result():
                    for key, value in result.items():
                        self.metadata_dict[key].update(value)
            except Exception as e:
                print(f"Error retrieving result from a finished future: {e}")

    def _preprocess_data(self, index_file_save_path: str, task_save_dir: str) -> None:
        """Preprocesses the downloaded video and action files for the specified task.

        This method identifies downloaded files that still need preprocessing by comparing them against
        entries in the `metadata.json` file. It then applies the necessary preprocessing steps to these files
        and logs the progress to avoid redundant processing.

        Args
            index_file_id (str): Identifier used to download index file from URL (e.g. "find-cave-Jul-28.json").
            index_file_save_path (str): Path to index file.
            task_save_dir (str): Directory of video and action files for specifiy task.

        Raises:
            Exception: If the preprocessing process encounters an error.
        """
        video_save_paths, action_save_paths = self._filter_downloaded_files(
            index_file_save_path, task_save_dir, "preprocess_is_sucess"
        )

        if len(video_save_paths) + len(action_save_paths) > 0:
            self._prepare_cursor_image()
            try:
                with tqdm(total=len(video_save_paths), desc="Preprocessing files") as pbar:
                    with ProcessPoolExecutor(max_workers=self.num_proc) as ex:
                        futures = [
                            ex.submit(self._preprocess_action_and_video_file, ap, vp)
                            for ap, vp in zip(action_save_paths, video_save_paths)
                        ]

                        try:
                            for future in as_completed(futures):
                                for key, value in future.result().items():
                                    self.metadata_dict[key].update(value)
                                pbar.update(1)

                        except KeyboardInterrupt:
                            print("KeyboadInterrupt received, waiting for all running futures to finish ...")
                            self._handle_preprocess_keyboard_interrupt(futures)
                            raise  # Re-raise the KeyboardInterrupt

            except Exception as e:
                print(f"Preprocessing failed {e}")

            finally:
                with open(os.path.join(task_save_dir, "metadata.json"), "w") as f:
                    json.dump(self.metadata_dict, f)
                ex.shutdown(cancel_futures=True)

    def _filter_downloaded_files(
        self, path_to_index_file: str, task_save_dir: str, check_type: str
    ) -> Tuple[List[str], List[str]]:
        """
        Identifies video and action files that still need to be downloaded or preprocessed, avoiding redundant work.

        This method compares the expected file save paths against the `metadata.json` file in `task_save_dir` to determine
        which files have already been downloaded or preprocessed.
        - If `check_type` is "preprocess_is_sucess", the method returns files that have been downloaded but not yet preprocessed.
        - If `check_type` is "download_is_sucess":
          - By default, it filters out files that have already been tried to be downloaded.
          - If `self.retry_download` is enabled, it only filters out successfully downloaded files, allowing failed downloads to be retried.

        The method ensures that video and action files are handled separately, as one may be processed while the other is not.
        It compares full save paths instead of filenames, preserving directory structure information.

        Args:
            path_to_index_file (str): Path to the index file containing base URLs and relative file paths.
            task_save_dir (str): Directory where task-related data is stored.
            check_type (str): Specifies whether to filter based on download or preprocessing status.
                             Must be one of ["download_is_sucess", "preprocess_is_sucess"].

        Returns:
            Tuple[List[str], List[str]]:
                - List of `.mp4` video file paths that still require processing.
                - List of `.jsonl` action file paths that still require processing.
        """

        video_save_paths, action_save_paths = self._create_save_paths_with_split(path_to_index_file, task_save_dir)

        if os.path.exists(f"{task_save_dir}/metadata.json"):
            with open(os.path.join(task_save_dir, "metadata.json"), "r") as f:
                self.metadata_dict = json.load(f)  # Load existing metadata dict (to add or overwrite possible changes).

        if check_type == "preprocess_is_sucess":
            files_to_preprocess = [
                key
                for key, value in self.metadata_dict.items()
                if value["download_is_sucess"] and not value["preprocess_is_sucess"]
            ]
            video_save_paths = [el for el in video_save_paths if el in files_to_preprocess]
            action_save_paths = [el for el in action_save_paths if el in files_to_preprocess]

        elif check_type == "download_is_sucess":
            if self.retry_download:
                ignore_files = [key for key, value in self.metadata_dict.items() if value["download_is_sucess"]]
            else:
                ignore_files = self.metadata_dict.keys()
            video_save_paths = [el for el in video_save_paths if el not in ignore_files]
            action_save_paths = [el for el in action_save_paths if el not in ignore_files]

        return video_save_paths, action_save_paths

    def _create_save_paths_with_split(self, path_to_index_file: str, task_save_dir: str) -> Tuple[List[str], List[str]]:
        """Creates lists containing save paths of all videos and action_dict lists to be downloaded.

        In order to create the save paths, each video and action file is assigned a label "test" or
        "train". Each save path has the format: tasks/<task_name>/<train-or-test>/<file_name>.
        The train-test split ratio is 80:20.

        Args:
            path_to_index_file (str): Path to index file that is supposed to be loaded.
            task_save_dir (str): Path to the directory where the tasks data is saved.

        Returns:
            List[str]: Save paths of all the mp4 video files.
            List[str]: Save paths of all the jsonl action files.
        """
        with open(path_to_index_file, "r") as f:
            index_file_dict = json.load(f)

        base_url = index_file_dict["basedir"]
        rel_paths = index_file_dict["relpaths"]
        self.base_url = os.path.dirname(os.path.join(base_url, rel_paths[0]))

        if self.limit_num_tasks:
            rel_paths = rel_paths[: self.limit_num_tasks]

        # Assign test or train label to each video (and corresponding action data file).
        total = len(rel_paths)
        train_test_assignments = ["train"] * int(total * 0.8) + ["test"] * (total - int(total * 0.8))

        # Create the paths to save videos and action files
        video_save_paths = [
            os.path.join(task_save_dir, assignment, os.path.basename(rel_path.replace("/data", "")))
            for rel_path, assignment in zip(rel_paths, train_test_assignments)
        ]
        action_save_paths = [save_path.replace("mp4", "jsonl") for save_path in video_save_paths]
        return video_save_paths, action_save_paths

    def _prepare_cursor_image(self) -> None:
        """Load cursor image in order to add it to specific frames in the prepocessing step."""
        cursor_image = cv2.imread(self.cursor_file, cv2.IMREAD_UNCHANGED)
        # Assume 16x16
        self.cursor_image = cursor_image[:16, :16, :]
        self.cursor_alpha = cursor_image[:, :, 3:] / 255.0
        self.cursor_image = self.cursor_image[:, :, :3]

    def _preprocess_video_file(self, action_path: str, video_path: str, metadata_dict: Dict[str, Any]) -> None:
        """Preprocess video file.

        Preprocessing steps:
            - Add cursor image to frames where GUI is open.

        Args:
            action_path (str): Path to action file.
            video_path (str): Path to video file.
            metadata_dict (Dict[str, Any]): Metadata dictionary to update.
        """
        preprocessed_video_path = video_path.replace(".mp4", "_preprocessed")
        video = cv2.VideoCapture(video_path)

        # Check if video is corrupted.
        if not video.isOpened():
            print(f"Video file {video_path} is corrupted and will be ignored.")
            metadata_dict[video_path]["preprocess_is_sucess"] = True
            metadata_dict[video_path]["corrupted"] = True
            metadata_dict[video_path]["n_frames"] = 0
            return
        # Read in action jsonl file.
        json_data = self.load_jsonl(action_path)

        # Preprocess video file.
        n_frames = 0
        for i, action_dict in enumerate(json_data):
            ret, frame = video.read()
            if ret:
                # Add cursor image to frame if the GUI is open.
                if action_dict["isGuiOpen"]:
                    camera_scaling_factor = frame.shape[0] / self.minerec_original_height_px
                    cursor_x = int(action_dict["mouse"]["x"] * camera_scaling_factor)
                    cursor_y = int(action_dict["mouse"]["y"] * camera_scaling_factor)
                    self._composite_images_with_alpha(frame, self.cursor_image, self.cursor_alpha, cursor_x, cursor_y)

                frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                cv2.imwrite(f"{preprocessed_video_path}/{i}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                n_frames += 1
            else:
                print(f"Could not read frame from video {video_path}")
                print(f"Skipping the rest of the video {video_path}")
                break

        video.release()

        metadata_dict[video_path]["preprocess_is_sucess"] = True
        metadata_dict[video_path]["corrupted"] = False
        metadata_dict[video_path]["n_frames"] = n_frames

    def _preprocess_action_and_video_file(self, action_path: str, video_path: str) -> Dict[str, Any]:
        """Preprocess action and video files.

        Args:
            action_path (str): Path to action file.
            video_path (str): Path to video file.

        Returns:
            Dict[str, Any]: Metadata dictionary containing information about preprocessing success.
        """
        metadata_dict = {action_path: {}, video_path: {}}
        os.makedirs(action_path.replace(".jsonl", "_preprocessed"), exist_ok=True)

        try:
            try:
                self._preprocess_action_file(action_path, metadata_dict)
            except Exception as e:
                print(f"Failed preprocessing action file {action_path}: {e}")
                metadata_dict[action_path]["preprocess_is_sucess"] = False
            try:
                self._preprocess_video_file(action_path, video_path, metadata_dict)
            except Exception as e:
                print(f"Failed preprocessing video file {video_path}: {e}")
                metadata_dict[video_path]["preprocess_is_sucess"] = False

            os.remove(action_path)
            os.remove(video_path)
            return metadata_dict

        except KeyboardInterrupt:
            pass  # allows for KeyboardInterrupt to be handled correctly for each subprocess

    def _json_action_to_env_action(self, json_action: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a json action into a MineRL action.

        Args:
            json_action (Dict[str, Any]): JSON action dictionary.

        Returns:
            Dict[str, Any]: MineRL action dictionary.
        """
        env_action = deepcopy(self.noop_action)
        keyboard_keys = json_action["keyboard"]["keys"]
        for key in keyboard_keys:
            # You can have keys that we do not use, so just skip them
            # NOTE in original training code, ESC was removed and replaced with
            #      "inventory" action if GUI was open.
            #      Not doing it here, as BASALT uses ESC to quit the game.
            if key in self.keyboard_button_mapping:
                env_action[self.keyboard_button_mapping[key]] = 1

        mouse = json_action["mouse"]
        camera_action = env_action["camera"]
        if mouse["dx"] != 0 or mouse["dy"] != 0:
            camera_action[0] = mouse["dy"] * self.camera_scaler
            camera_action[1] = mouse["dx"] * self.camera_scaler
        else:
            if abs(camera_action[0]) > 180:
                camera_action[0] = 0
            if abs(camera_action[1]) > 180:
                camera_action[1] = 0

        mouse_buttons = mouse["buttons"]
        if 0 in mouse_buttons:
            env_action["attack"] = 1
        if 1 in mouse_buttons:
            env_action["use"] = 1
        if 2 in mouse_buttons:
            env_action["pickItem"] = 1

        return env_action

    def _composite_images_with_alpha(self, image1, image2, alpha, x, y):
        """
        Draw image2 over image1 at location x,y, using alpha as the opacity for image2.

        Modifies image1 in-place
        """
        ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
        cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
        if ch == 0 or cw == 0:
            return
        alpha = alpha[:ch, :cw]
        image1[y : y + ch, x : x + cw, :] = (
            image1[y : y + ch, x : x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha
        ).astype(np.uint8)

    def _create_sampling_structure(self, task_name: str) -> None:
        """Create list from which __get_item__ method samples.

        Loops over all preprocessed train or test video files that are available for a specific task,
        gets the frame count (N) for each video file and adds the file path N times to the list.
        This way a list is created that contains an element for each frame of each video, making it
        suitable for frame-wise sampling.

        Args:
            task_name (str): task which is supposed to be added to list
        """
        prefix_len = len(self.dataset_dir) + 1
        all_video_paths = [key for key in self.metadata_dict.keys() if key.endswith(".mp4")]
        split_video_paths = [key for key in all_video_paths if self.split in key[prefix_len:]]
        for video_path in split_video_paths:
            metadata = self.metadata_dict[video_path]
            if metadata["preprocess_is_sucess"] and not metadata["corrupted"]:
                task_path = video_path.replace(".mp4", "_preprocessed")
                if metadata["n_frames"] >= self.block_size:
                    self.length += metadata["n_frames"] + 1 - self.block_size
                    self.tasks.append((task_name, task_path, metadata["n_frames"]))

    def load_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load a JSONL file.

        Args:
            jsonl_path (str): Path to the JSONL file.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the JSONL data.
        """
        with open(jsonl_path) as json_file:
            json_lines = json_file.readlines()
            json_data = "[" + ",".join(json_lines) + "]"
            json_data = json.loads(json_data)
        return json_data

    def _preprocess_action_file(self, action_path: str, metadata_dict: Dict[str, Any]) -> None:
        """Preprocess action file.

        Preprocessing steps:
            - Remove actions where attack is stuck down from the beginning.
              In some recordings, the game seems to start with attack always down from the beginning, which is stuck
              down until player actually presses attack.
            - Update hotbar actions when hotbar selection changes based on scrollwheel.
              The scrollwheel is an allowed way to change items, but this is not captured by the recorder. We work
              around this by keeping track of the selected hotbar item and updating "hotbar.#" actions when hotbar
              selection changes.

        Args:
            action_path (str): Path to action file.
            metadata_dict (Dict[str, Any]): Metadata dictionary to update.
        """
        preprocessed_action = []
        attack_is_stuck = False
        last_hotbar = 0

        # Read in action jsonl file.
        json_data = self.load_jsonl(action_path)

        # Preprocess action file.
        for i, action_dict in enumerate(json_data):
            # Check if attack is stuck down from the beginning.
            if i == 0:
                if action_dict["mouse"]["newButtons"] == [0]:
                    attack_is_stuck = True
            # Check if we press attack down, then it might not be stuck.
            elif attack_is_stuck:
                if 0 in action_dict["mouse"]["newButtons"]:
                    attack_is_stuck = False

            # If still stuck, remove the action.
            if attack_is_stuck:
                action_dict["mouse"]["buttons"] = [button for button in action_dict["mouse"]["buttons"] if button != 0]

            # Convert json action to env action.
            action = self._json_action_to_env_action(action_dict)

            # Capture hotbar selection changes.
            current_hotbar = action_dict["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar

            # Append preprocessed action to list.
            preprocessed_action.append(action)

        # Save as PyArrow table.
        table = pa.Table.from_pylist(preprocessed_action, schema=self.schema)
        with pa.ipc.new_file(action_path.replace(".jsonl", "_preprocessed/actions.arrow"), self.schema) as writer:
            writer.write(table)
        metadata_dict[action_path]["preprocess_is_sucess"] = True

    def _get_worker_task_queue(self) -> Queue:
        """Initialize the worker task queue.

        This method sets up the worker-specific task queue based on the worker ID and the total number of workers.
        It essentially distributes the tasks among the workers.

        Returns:
            Queue: The worker task queue.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        else:
            self.worker_id = 0
            self.num_workers = 1

        worker_task_queue = [task for i, task in enumerate(self.tasks) if i % self.num_workers == self.worker_id]

        if self.shuffle:
            self.generator.shuffle(worker_task_queue)

        shuffled_worker_task_queue = Queue()
        for task in worker_task_queue:
            shuffled_worker_task_queue.put(task)

        return shuffled_worker_task_queue

    def _set_up_workers(self, task_queue: Queue) -> Queue:
        """Set up the worker threads.

        Args:
            task_queue (Queue): The worker task queue to pass to the workers.

        Returns:
            Queue: The buffer queue to store the samples.
        """
        # Fill up worker task queue with the maximum number of parallel videos.
        available_workers = min(self.num_parallel_videos, task_queue.qsize())
        if available_workers < self.num_parallel_videos:
            print(f"Number of parallel videos {self.num_parallel_videos} exceeds available videos {available_workers}.")

        barrier = Barrier(available_workers)
        buffer = Queue(maxsize=self.queue_size)
        workers = [
            Thread(target=self._get_worker, args=(task_queue, buffer, barrier, id)) for id in range(available_workers)
        ]

        # Start workers.
        for worker in workers:
            worker.start()

        return buffer

    def _get_worker(self, task_queue: Queue, buffer: Queue, barrier: Barrier, worker_id: int) -> None:
        """Worker function to process the tasks in task_queue and fill the buffer with samples.

        Args:
            task_queue (Queue): The tasks to process.
            buffer (Queue): The buffer to store the samples.
            barrier (Barrier): The barrier to synchronize the workers after finishing.
            worker_id (int): The id of the worker.
        """
        while True:
            try:
                task_name, task_path, num_samples = task_queue.get_nowait()
            except Empty:
                break

            # Open action file:
            with pa.ipc.open_file(f"{task_path}/actions.arrow") as reader:
                actions_dict = self.formatter.format_batch(reader.read_all())
                num_samples = actions_dict["forward"].shape[0]
                actions = TensorDict(actions_dict, batch_size=(num_samples,))
            actions = self.transform_action(actions)

            # Get one-hot task encoding and reshape it to the block size.
            task = self.transform_task(self.task_one_hot[self.task_ids[task_name]])[None, :].expand(self.block_size, -1)
            # Fill up sample buffer until we can yield a block of samples and continue until the end of the video.
            video_buffer = []
            for i in range(num_samples):
                frame = torch.from_numpy(cv2.imread(f"{task_path}/{i}.jpg", flags=cv2.IMREAD_COLOR_RGB))
                frame = self.transform_observation(self.transform_layout(frame))
                video_buffer.append(frame)
                if len(video_buffer) == self.block_size:
                    video = torch.stack(video_buffer)
                    video_buffer.pop(0)
                    sample = TensorDict(
                        {"observation": video, "action": actions[i + 1 - self.block_size : i + 1], "task": task},
                        batch_size=(self.block_size,),
                    )
                    buffer.put(sample, timeout=self.queue_timeout)

        # Wait for all workers to finish and signal the end of the buffer.
        barrier.wait()
        if worker_id == 0:
            buffer.put(None)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.length

    def __iter__(self) -> Iterator[TensorDict]:
        """Initialize the worker task queue and return an iterator over the dataset.

        This method sets up the worker-specific task queue based on the worker ID and the total number of workers.
        It then returns an iterator that yields samples from the dataset.

        Returns:
            Iterator[TensorDict]: An iterator over the dataset.
        """
        worker_task_queue = self._get_worker_task_queue()
        buffer = self._set_up_workers(worker_task_queue)
        return self.__next__(buffer)

    def __next__(self, buffer: Queue) -> Iterator[TensorDict]:
        """Yields samples from the dataset.

        Yields:
            Iterator[TensorDict]: An iterator over the dataset.
        """
        while True:
            try:
                sample = buffer.get(timeout=self.queue_timeout)
            except Empty:
                print("Warning: Could not get sample from buffer.")
                break
            # If the buffer is empty, signal the end of the dataset.
            if sample is not None:
                yield sample
            else:
                break
