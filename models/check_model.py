import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="yolov5n_full_int_quant.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Get input tensor properties for quantization
input_scale = input_details['quantization_parameters']['scales'][0]
input_zero_point = input_details['quantization_parameters']['zero_points'][0]
input_dtype = input_details['dtype'] # Should be tf.int8 or np.int8

print(input_scale)
print(input_zero_point)
print(input_dtype)