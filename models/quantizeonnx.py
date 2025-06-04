import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def quantize_onnx_model_dynamic(model_path, quantized_model_path):
    """
    Quantizes an ONNX model using dynamic range quantization.

    Args:
        model_path (str): The file path to the input ONNX model.
        quantized_model_path (str): The file path where the quantized ONNX model will be saved.
    """
    try:
        print(f"Attempting to quantize model: {model_path}")
        print(f"Quantized model will be saved to: {quantized_model_path}\n")

        # Ensure the output directory exists
        output_dir = os.path.dirname(quantized_model_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Perform dynamic quantization
        # This method quantizes weights to int8 and activations to int8/uint8 dynamically at runtime.
        # It's generally a good starting point for quantization as it doesn't require a calibration dataset.
        quantize_dynamic(
            model_path,
            quantized_model_path,
            weight_type=QuantType.QUInt8, # Quantize weights to signed 8-bit integers
            per_channel=False              # Apply per-tensor quantization for weights
        )

        print(f"\nSuccessfully quantized model and saved to: {quantized_model_path}")

        # Optional: Verify the quantized model can be loaded
        try:
            quantized_model = onnx.load(quantized_model_path)
            print("Quantized model loaded successfully for verification.")
            print(f"Quantized model graph inputs: {[i.name for i in quantized_model.graph.input]}")
            print(f"Quantized model graph outputs: {[o.name for o in quantized_model.graph.output]}")
        except Exception as e:
            print(f"Warning: Could not load or verify the quantized model. Error: {e}")

    except FileNotFoundError:
        print(f"Error: Input model file not found at '{model_path}'")

# --- Example Usage ---
if __name__ == "__main__":
    # Define the path to your input ONNX model (from previous context)
    input_model_file = "yolo11n.onnx" 
    
    # Define the path for the output quantized ONNX model
    # It's good practice to save it with a different name, e.g., adding '_quantized'
    output_model_file = "yolo11n_quantized_dynamic.onnx" 
    
    quantize_onnx_model_dynamic(input_model_file, output_model_file)
    print("\nQuantization process complete.")

