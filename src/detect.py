#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

from typing import Tuple

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


class Detect(object):
    """YOLOv5 tflite detect model."""

    def __init__(
        self,
        model_file: str,
        conf_thr: float,
    ):
        # Load model to memory.
        self.interpreter = Interpreter(model_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter
        _, self.width, self.height, _ = self.interpreter.get_input_details()[0]["shape"]
        self.output_details = self.interpreter.get_output_details()
        self.conf_thr = conf_thr

    def detect(self, img: np.ndarray, box_type="xywh") -> Tuple[np.ndarray]:
        """Detect objects.
        Returns:
           Tuple[np.ndarray]: The shape of each element is (25500, 4) (25500,) (25500,).
        """
        img = self.preprocess(img)
        output_data = self._detect(img)
        boxes, scores, class_idx = self.postprocess(output_data, box_type)
        return boxes, scores, class_idx

    def _detect(self, img: np.ndarray):
        """Inference."""
        self.interpreter.set_tensor(0, img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])  # get tensor  x(1, 25200, 7)
        return output_data

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess."""
        # Resize
        img = cv2.resize(img, (self.height, self.width))
        # BGR -> RGB
        img = img[:, :, [2, 1, 0]]
        # Normalize
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def postprocess(self, output_data_batch: Tuple[np.ndarray, ...], box_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Improved Postprocess: Filter by confidence first, then run argmax on fewer items.
        """
        # Assuming output_data_batch is a tuple/list containing one primary output tensor
        # or that it's directly the output tensor.
        # The original code did output_data = output_data[0], so we'll stick to that pattern.
        output_data = output_data_batch[0]  # Shape: (num_proposals, 4 + 1 + num_classes)

        if output_data.shape[0] == 0: # Handle cases with no proposals
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        # --- 1. Extract raw data ---
        # (N, 4) for boxes (xywh)
        # (N,) for confidences
        # (N, num_classes) for class scores
        raw_boxes_xywh = output_data[..., :4]
        raw_confidences = output_data[..., 4]    # Shape (N,), no need to squeeze later
        raw_class_scores = output_data[..., 5:] # Shape (N, num_classes)

        # --- 2. Filter by confidence threshold ---
        # Create a boolean mask for proposals above the confidence threshold
        # This is generally faster than np.where for direct indexing.
        conf_filter_mask = raw_confidences > self.conf_thr

        # Apply this mask to get filtered data
        # These will be empty arrays if no proposals pass the threshold
        filtered_boxes_xywh = raw_boxes_xywh[conf_filter_mask]
        filtered_confidences = raw_confidences[conf_filter_mask]
        filtered_class_scores = raw_class_scores[conf_filter_mask]

        # --- 3. Handle case where no proposals pass the confidence threshold ---
        num_filtered_proposals = filtered_boxes_xywh.shape[0]
        if num_filtered_proposals == 0:
            return (
                np.empty((0, 4), dtype=raw_boxes_xywh.dtype),
                np.empty((0,), dtype=raw_confidences.dtype),
                np.empty((0,), dtype=np.float32), # For consistency with original cls dtype
            )

        # --- 4. Determine class IDs for filtered proposals ---
        # Argmax on the (potentially much) smaller, filtered class scores array
        # Resulting shape: (num_filtered_proposals,)
        filtered_cls_ids = np.argmax(filtered_class_scores, axis=1).astype(np.float32)

        # --- 5. Convert box format if necessary ---
        if box_type == "xyxy":
            # xywh -> xyxy
            final_boxes = self.to_xyxy(filtered_boxes_xywh)
        else:
            final_boxes = filtered_boxes_xywh # Already xywh

        return final_boxes, filtered_confidences, filtered_cls_ids

    def to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Covert xywh to xyxy."""
        # (x, y) is cordinate fo the center of the box.
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        boxes = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        # [4, n] -> [n, 4]
        boxes = boxes.transpose((1, 0))
        return boxes