"""SAM2 video mask propagation for two entities (agent + object) per episode.

Pure helpers (`empty_masks`, `mask_tensor_to_pil`) are unit-tested; the
`Sam2DualPredictor` model wrapper is covered by GPU integration tests.
"""
import numpy as np
import PIL.Image


def empty_masks(num_frames: int, height: int, width: int) -> list:
    """A list of ``num_frames`` all-black single-channel masks."""
    return [
        PIL.Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L")
        for _ in range(num_frames)
    ]


def mask_tensor_to_pil(mask) -> PIL.Image.Image:
    """Convert a binary mask (torch tensor or ndarray) to a mode-``L`` PIL image.

    Leading singleton dims (e.g. SAM2's ``(1, 1, H, W)`` per object) are squeezed;
    values are binarized to 0/255.
    """
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    arr = np.asarray(mask)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    arr = (arr > 0).astype(np.uint8) * 255
    return PIL.Image.fromarray(arr, mode="L")


class Sam2DualPredictor:
    """SAM2 video predictor that tracks the agent (obj_id=1, fixed frame-0 box)
    and, when detected, the object (obj_id=2, GDINO frame-0 box) in a single
    propagation pass. Uses the transformers-v5 video API. Loads the checkpoint
    lazily so importing this module stays cheap for unit tests.
    """

    def __init__(self, model_id: str, device: str, dtype):
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.device = device
        self.dtype = dtype
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id).to(device, dtype=dtype).eval()

    def process_episode(self, frames: list, agent_box: list, object_box) -> dict:
        """Return ``{"agent": [PIL.L, ...], "object": [PIL.L, ...]}``, one mask per
        frame. If ``object_box`` is None, every object mask is empty. On CUDA OOM,
        returns all-empty masks for both entities (matching the authors' guard)."""
        import torch

        try:
            return self._process_impl(frames, agent_box, object_box)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            w, h = frames[0].size
            n = len(frames)
            return {"agent": empty_masks(n, h, w), "object": empty_masks(n, h, w)}

    def _process_impl(self, frames: list, agent_box: list, object_box) -> dict:
        import torch

        session = self.processor.init_video_session(
            video=frames, inference_device=self.device
        )
        # v5 API: all objects in ONE add_inputs call (obj_with_new_inputs is
        # replaced, not appended). input_boxes shape (1, num_objs, 4). Then
        # propagate from frame 0 - no separate model(frame_idx=0) call.
        obj_ids = [1] if object_box is None else [1, 2]
        boxes = [agent_box] if object_box is None else [agent_box, object_box]
        self.processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=obj_ids, input_boxes=[boxes]
        )

        segments = {}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.dtype):
            for out in self.model.propagate_in_video_iterator(session, start_frame_idx=0):
                masks = self.processor.post_process_masks(
                    [out.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=True,
                )[0]  # (num_objs, 1, H, W)
                segments[out.frame_idx] = masks

        w, h = frames[0].size
        agent_out, object_out = [], []
        for i in range(len(frames)):
            masks = segments.get(i)
            if masks is None:
                agent_out.append(PIL.Image.fromarray(np.zeros((h, w), np.uint8), "L"))
                object_out.append(PIL.Image.fromarray(np.zeros((h, w), np.uint8), "L"))
                continue
            agent_out.append(mask_tensor_to_pil(masks[0]))
            if object_box is not None:
                object_out.append(mask_tensor_to_pil(masks[1]))
            else:
                object_out.append(PIL.Image.fromarray(np.zeros((h, w), np.uint8), "L"))
        return {"agent": agent_out, "object": object_out}
