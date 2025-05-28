import tensorflow as tf
import os
import numpy as np # Used for dtype comparison, though not strictly necessary for just printing

def get_tflite_model_data_types(model_path):
    """
    Inspects a TFLite model to determine the data types of its input and output tensors.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the model path is correct.")
        return

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Model: {model_path}")
        print("\n--- Input Tensor Data Types ---")
        for i, detail in enumerate(input_details):
            print(f"  Input {i} (name: {detail['name']}): {detail['dtype']}")
            if 'quantization_parameters' in detail and detail['quantization_parameters']:
                qp = detail['quantization_parameters']
                print(f"    Quantization: scale={qp.get('scales')}, zero_point={qp.get('zero_points')}")

        print("\n--- Output Tensor Data Types ---")
        for i, detail in enumerate(output_details):
            print(f"  Output {i} (name: {detail['name']}): {detail['dtype']}")
            if 'quantization_parameters' in detail and detail['quantization_parameters']:
                qp = detail['quantization_parameters']
                print(f"    Quantization: scale={qp.get('scales')}, zero_point={qp.get('zero_points')}")

    except Exception as e:
        print(f"An error occurred while interpreting the TFLite model: {e}")
        print("Please ensure you have TensorFlow installed (`pip install tensorflow`)")
        print("and that the .tflite file is a valid model.")

if __name__ == "__main__":
    # Define the exact path to your YOLOv5 TFLite model
    model_file_path = "./yolov5n-fp16.tflite" # As specified in your original script

    get_tflite_model_data_types(model_file_path)