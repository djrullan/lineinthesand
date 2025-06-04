#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

from typing import Tuple

import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


class Detect(object):
    """YOLOv5 tflite detect model.

    This class dynamically handles both fully integer quantized (INT8)
    and float (FP16/FP32) TFLite models for input and output processing.
    """

    def __init__(
        self,
        model_file: str,
        conf_thr: float,
    ):
        """Initializes the Detect class.

        Args:
            model_file (str): Path to the TFLite model file.
            conf_thr (float): Confidence threshold for object detection.
        """

        # Load model to memory.
        self.interpreter = Interpreter(model_file)
        self.interpreter.allocate_tensors()

        # Get input and output tensor details
        self.input_details = self.interpreter.get_input_details()[0]
        # Extract input shape (batch, height, width, channels)
        _, self.height, self.width, _ = self.input_details["shape"]
        self.output_details = self.interpreter.get_output_details()[0]  # Assuming a single output tensor

        # Determine if the model is quantized based on input/output dtypes
        self.is_quantized_input = self.input_details['dtype'] == np.int8
        self.is_quantized_output = self.output_details['dtype'] == np.int8

        # Initialize quantization parameters. They will only be used if the model is quantized.
        self.input_scale = 1.0
        self.input_zero_point = 0
        self.output_scale = 1.0
        self.output_zero_point = 0

        if self.is_quantized_input:
            self.input_scale = self.input_details['quantization_parameters']['scales'][0]
            self.input_zero_point = self.input_details['quantization_parameters']['zero_points'][0]
        
        if self.is_quantized_output:
            self.output_scale = self.output_details['quantization_parameters']['scales'][0]
            self.output_zero_point = self.output_details['quantization_parameters']['zero_points'][0]

        self.input_dtype = self.input_details['dtype']
        self.conf_thr = conf_thr
        # print(f"CONF THRESHOLD IS {self.conf_thr}")
        # print(f"Model input type: {self.input_dtype}, Quantized: {self.is_quantized_input}")
        # print(f"Model output type: {self.output_details['dtype']}, Quantized: {self.is_quantized_output}")


    def detect(self, img: np.ndarray, box_type="xywh") -> Tuple[np.ndarray]:
        """Detect objects in the given image.

        Args:
            img (np.ndarray): The input image (BGR format, typically 0-255).
            box_type (str, optional): Desired bounding box format ("xywh" or "xyxy").
                                      Defaults to "xywh".

        Returns:
            Tuple[np.ndarray]: A tuple containing:
                - boxes (np.ndarray): Bounding box coordinates.
                - scores (np.ndarray): Confidence scores for each detection.
                - class_idx (np.ndarray): Class indices for each detection.
                The shape of each element is (num_detections, 4), (num_detections,), (num_detections,).
        """
        # Preprocess the image (resize, BGR->RGB, normalize, and optionally quantize)
        processed_img = self.preprocess(img)
        # Perform inference
        output_data = self._detect(processed_img)
        # Postprocess the output (optionally de-quantize, filter, convert box format)
        boxes, scores, class_idx = self.postprocess(output_data, box_type)
        return boxes, scores, class_idx

    def _detect(self, img: np.ndarray) -> np.ndarray:
        """Performs inference on the preprocessed image.

        Args:
            img (np.ndarray): The preprocessed image tensor (e.g., float32 or int8).

        Returns:
            np.ndarray: The raw output tensor from the TFLite model (e.g., float32 or int8).
        """
        # Set the input tensor for the interpreter
        self.interpreter.set_tensor(self.input_details['index'], img)
        # Invoke the interpreter to run inference
        self.interpreter.invoke()
        # Get the raw output tensor from the model
        output_data = self.interpreter.get_tensor(self.output_details["index"])
        return output_data

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocesses the input image for the TFLite model.

        Steps:
        1. Resize the image to the model's expected input dimensions.
        2. Convert BGR (OpenCV default) to RGB.
        3. Normalize pixel values from 0-255 to 0-1 (float32).
        4. (Conditionally) Quantize the float32 values to the model's expected int8 input range
           using the input tensor's scale and zero_point, if the model is quantized.
        5. Add a batch dimension.

        Args:
            img (np.ndarray): The input image in BGR format (e.g., from cv2.imread).

        Returns:
            np.ndarray: The preprocessed image tensor, ready for inference.
        """
        # Resize to the model's expected input size
        img = cv2.resize(img, (self.width, self.height))
        # Convert BGR (OpenCV default) to RGB
        img = img[:, :, [2, 1, 0]]
        # Normalize pixel values from 0-255 to 0-1 (float32)
        img = img.astype(np.float32) / 255.0

        if self.is_quantized_input:
            # Quantize for INT8 model
            img = (img / self.input_scale) + self.input_zero_point
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        # Cast to the model's expected input data type (e.g., np.int8 or np.float32)
        return img.astype(self.input_dtype)

    def postprocess(self, output_data: np.ndarray, box_type: str) -> Tuple[np.ndarray]:
        """Postprocesses the raw output from the TFLite model.

        Steps:
        1. (Conditionally) De-quantize the int8 output data back to float32, if the model is quantized.
        2. Remove the batch dimension.
        3. Extract bounding boxes, confidence scores, and class probabilities.
        4. Apply confidence thresholding.
        5. Convert box format if requested (xywh to xyxy).

        Args:
            output_data (np.ndarray): The raw output tensor from the TFLite model (e.g., float32 or int8).
            box_type (str): Desired bounding box format ("xywh" or "xyxy").

        Returns:
            Tuple[np.ndarray]: A tuple containing filtered boxes, scores, and class indices.
        """
        # Store original raw output data if quantized, for debugging
        raw_output_for_debug = output_data.copy() if self.is_quantized_output else None

        if self.is_quantized_output:
            # De-quantize the output data from int8 back to float32
            output_data = (output_data - self.output_zero_point) * self.output_scale

        # Remove the batch dimension (e.g., (1, 25200, 7) -> (25200, 7))
        output_data = output_data[0]
        if raw_output_for_debug is not None:
             raw_output_for_debug = raw_output_for_debug[0] # Also remove batch dim for consistency

        # Extract bounding box coordinates (xywh), confidence, and class probabilities
        boxes = output_data[..., :4]
        conf = output_data[..., 4] # Changed from 4:5 to 4 for direct array
        cls_probs = output_data[..., 5:]  # Class probabilities for each class

        # Get the index of the class with the highest probability
        # Add epsilon to cls_probs before argmax if they are all zero to avoid argmax breaking on all-zero slice
        # (though np.argmax handles it, it's good practice for general case)
        cls = np.argmax(cls_probs, axis=1).astype(np.float32)
        max_cls_probs = np.max(cls_probs, axis=1) # Get the highest probability for each detection

        # --- START: Added/Adjusted Print Statements for Debugging ---
        # print(f"\n--- Post-processing Debug (Before Threshold) ---")
        # print(f"Total detections (raw output): {len(conf)}")
        # print(f"De-quantized overall confidences: min={conf.min():.4f}, max={conf.max():.4f}, mean={conf.mean():.4f}")

        # Detailed inspection of class probabilities (before argmax)
        # if cls_probs.size > 0:
        #     # print(f"De-quantized class probabilities (overall): min={cls_probs.min():.4f}, max={cls_probs.max():.4f}, mean={cls_probs.mean():.4f}")
        #     # Pick a few samples to inspect
        #     # Find an index where objectness conf is higher, to see its class probs
        #     strong_conf_indices = np.where(conf > 0.1)[0] # Example: where objectness is reasonably high
        #     if len(strong_conf_indices) > 0:
        #         # print(f"Sample de-quantized class probabilities for high objectness conf detections (first 5):")
        #         for i in range(min(5, len(strong_conf_indices))):
        #             idx = strong_conf_indices[i]
        #             print(f"  Det {idx}: Objectness Conf={conf[idx]:.4f}, All Class Probs={cls_probs[idx]}")
        #             if raw_output_for_debug is not None:
        #                 print(f"    Raw INT8 Class Probs: {raw_output_for_debug[idx, 5:]}")
        #     else:
        #         print("No detections with high objectness confidence found to sample class probabilities.")

        # Find detections classified as human (class 0)
        human_idxs = np.where(cls == 0)[0]

        # if len(human_idxs) > 0:
        #     human_confs = conf[human_idxs]
        #     human_max_cls_probs = max_cls_probs[human_idxs]
        #     print(f"\n--- Human (Class 0) Detections ---")
        #     print(f"Number of human detections: {len(human_idxs)}")
        #     print(f"Human objectness confidences: min={human_confs.min():.4f}, max={human_confs.max():.4f}, mean={human_confs.mean():.4f}")
        #     print(f"Human highest class probabilities: min={human_max_cls_probs.min():.4f}, max={human_max_cls_probs.max():.4f}, mean={human_max_cls_probs.mean():.4f}")

        #     # Also check raw INT8 values for a few human detections
        #     if raw_output_for_debug is not None:
        #         print(f"Sample raw INT8 values for human (Class 0) detections (first 5):")
        #         for i in range(min(5, len(human_idxs))):
        #             idx = human_idxs[i]
        #             print(f"  Det {idx}: Objectness Conf={conf[idx]:.4f}, Class Probs (de-quantized)={cls_probs[idx]}, Raw INT8 Class Probs={raw_output_for_debug[idx, 5:]}")

        # else:
        #     print(f"\nNo detections identified as human (Class 0) before thresholding.")
        # print(f"--------------------------------------------------\n")
        # --- END: Added/Adjusted Print Statements ---

        # Convert bounding box format if requested
        if box_type == "xyxy":
            boxes = self.to_xyxy(boxes)

        # Filter detections by confidence threshold
        idxs = np.where(conf > self.conf_thr)
        boxes = boxes[idxs]
        cls = cls[idxs]
        conf = conf[idxs]

        return boxes, conf, cls

    def to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Converts bounding box format from xywh (center, width, height) to xyxy (top-left, bottom-right).

        Args:
            boxes (np.ndarray): Bounding boxes in xywh format (N, 4).

        Returns:
            np.ndarray: Bounding boxes in xyxy format (N, 4).
        """
        # (x, y) is the center coordinate of the box.
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        # Calculate top-left (x1, y1) and bottom-right (x2, y2) coordinates
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        # Stack the coordinates and transpose to get shape (N, 4)
        boxes = np.array([x1, y1, x2, y2]).transpose((1, 0))
        return boxes