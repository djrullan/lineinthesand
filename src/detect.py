#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino
from typing import Tuple
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

class Detect(object):
    """YOLOv5 tflite detect model with int8 quantization support."""
    
    def __init__(
        self,
        model_file: str,
        conf_thr: float,
    ):
        # Load model to memory.
        self.interpreter = Interpreter(model_file)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        _, self.width, self.height, _ = self.input_details[0]["shape"]
        
        # Check if model is quantized and get quantization parameters
        self._setup_quantization_params()
        
        self.conf_thr = conf_thr
    
    def _setup_quantization_params(self):
        """Setup quantization parameters for input and output tensors."""
        # Input quantization parameters
        input_quant_params = self.input_details[0].get('quantization_parameters', {})
        self.input_scale = input_quant_params.get('scales', [1.0])[0]
        self.input_zero_point = input_quant_params.get('zero_points', [0])[0]
        self.input_is_quantized = self.input_details[0]['dtype'] == np.uint8
        
        # Output quantization parameters
        output_quant_params = self.output_details[0].get('quantization_parameters', {})
        self.output_scale = output_quant_params.get('scales', [1.0])[0]
        self.output_zero_point = output_quant_params.get('zero_points', [0])[0]
        self.output_is_quantized = self.output_details[0]['dtype'] == np.uint8
        
        print(f"Input quantized: {self.input_is_quantized}")
        print(f"Output quantized: {self.output_is_quantized}")
        if self.input_is_quantized:
            print(f"Input scale: {self.input_scale}, zero_point: {self.input_zero_point}")
        if self.output_is_quantized:
            print(f"Output scale: {self.output_scale}, zero_point: {self.output_zero_point}")

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
        """Inference with quantization handling."""
        self.interpreter.set_tensor(self.input_details[0]["index"], img)
        self.interpreter.invoke()
        
        # Get raw output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        
        # Dequantize output if necessary
        if self.output_is_quantized:
            output_data = self._dequantize_output(output_data)
            
        return output_data

    def _quantize_input(self, data: np.ndarray) -> np.ndarray:
        """Quantize float32 input to uint8."""
        # Quantization formula: quantized_value = float_value / scale + zero_point
        quantized = np.round(data / self.input_scale + self.input_zero_point)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        return quantized

    def _dequantize_output(self, data: np.ndarray) -> np.ndarray:
        """Dequantize uint8 output to float32."""
        # Dequantization formula: float_value = (quantized_value - zero_point) * scale
        dequantized = (data.astype(np.float32) - self.output_zero_point) * self.output_scale
        return dequantized

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess with quantization support."""
        # Resize
        img = cv2.resize(img, (self.height, self.width))
        
        # BGR -> RGB
        img = img[:, :, [2, 1, 0]]
        
        # Normalize to [0, 1] range
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Handle quantization
        if self.input_is_quantized:
            # For quantized models, we need to quantize the normalized input
            img = img.astype(np.float32)
            img = self._quantize_input(img)
        else:
            img = img.astype(np.float32)
            
        return img

    def postprocess(self, output_data, box_type: str) -> Tuple[np.ndarray]:
        """Postprocess with proper data type handling."""
        output_data = output_data[0]
        
        # Ensure output_data is float32 for all operations
        output_data = output_data.astype(np.float32)
        
        # xywh
        boxes = output_data[..., :4]
        conf = output_data[..., 4:5]
        cls = np.argmax(output_data[..., 5:], axis=1).astype(np.float32).reshape(-1, 1)
        conf = np.squeeze(conf, axis=1)
        cls = np.squeeze(cls, axis=1)

        if box_type == "xyxy":
            # xywh -> xyxy
            boxes = self.to_xyxy(boxes)

        # Filter by confidence threshold.
        idxs = np.where(conf > self.conf_thr)
        boxes = boxes[idxs]
        cls = cls[idxs]
        conf = conf[idxs]

        return boxes, conf, cls

    def to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert xywh to xyxy."""
        # (x, y) is coordinate of the center of the box.
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        boxes = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        # [4, n] -> [n, 4]
        boxes = boxes.transpose((1, 0))
        return boxes

    def get_model_info(self):
        """Get model information for debugging."""
        print("=== Model Information ===")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input dtype: {self.input_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        print(f"Output dtype: {self.output_details[0]['dtype']}")
        print(f"Input quantized: {self.input_is_quantized}")
        print(f"Output quantized: {self.output_is_quantized}")