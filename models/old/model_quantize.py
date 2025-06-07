import tensorflow as tf
import numpy as np
import os # For path checks

# Path to your SavedModel directory
saved_model_dir = "yolov5n_saved_model"

# Function to get model size
def get_model_size(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024) # Size in MB
    return 0

# Function to get number of parameters for a Keras model (from SavedModel)
def get_keras_model_parameters(saved_model_path):
    try:
        # Load the SavedModel as a Keras model
        model = tf.keras.models.load_model(saved_model_path)
        # Count trainable parameters
        trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
        # Count non-trainable parameters
        non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
        return trainable_params + non_trainable_params
    except Exception as e:
        print(f"Could not load SavedModel as Keras model or get parameters: {e}")
        return "N/A"


if not os.path.exists(saved_model_dir) or not os.path.isdir(saved_model_dir):
    print(f"Error: SavedModel directory not found at {saved_model_dir}")
    print("Please ensure the path is correct and it's a directory.")
else:
    print(f"Using SavedModel from: {saved_model_dir}")

    # --- Print Model Size and Parameters BEFORE Quantization ---
    # Get the size of the SavedModel directory (approximate)
    saved_model_size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(saved_model_dir) for filename in filenames) / (1024 * 1024)
    print(f"\n--- Original SavedModel Statistics ---")
    print(f"Original Model Size: {saved_model_size_mb:.2f} MB (directory size)")
    original_params = get_keras_model_parameters(saved_model_dir)
    print(f"Original Model Estimated Parameters: {original_params}")


    # --- Option 1: Dynamic Range Quantization (Recommended first step) ---
    print(f"\n--- Applying Dynamic Range Quantization ---")
    converter_dr = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter_dr.optimizations = [tf.lite.Optimize.DEFAULT] # This enables dynamic range quantization
    tflite_model_dr = converter_dr.convert()

    dr_output_path = "yolov5n_repeat.tflite"
    with open(dr_output_path, "wb") as f:
        f.write(tflite_model_dr)
    print(f"Dynamic Range Quantized TFLite model saved to: {dr_output_path}")

    # --- Print Model Size and Parameters AFTER Dynamic Range Quantization ---
    dr_quant_model_size = get_model_size(dr_output_path)
    print(f"\n--- Dynamic Range Quantized Model Statistics ---")
    print(f"Quantized Model Size: {dr_quant_model_size:.2f} MB")
    print(f"Quantized Model Parameters: Not directly available for TFLite, but typically reduced in effective memory footprint.")


    # --- Option 2: Full Integer Quantization (Requires Representative Dataset) ---
    # This requires a function that yields input data for calibration.
    # Replace this dummy data with actual preprocessed images from your dataset.
    def representative_dataset_gen():
        # Example: YOLOv5n typically expects (1, 640, 640, 3) float32 input in RGB format, normalized to 0-1
        # Adjust batch size (1 in this example) and resolution (640) as per your model's requirement.
        num_calibration_samples = 100 # Adjust based on your dataset size and desired calibration quality
        for _ in range(num_calibration_samples):
            # Create a dummy image (replace with loading and preprocessing a real image)
            dummy_image = np.random.rand(1, 640, 640, 3).astype(np.float32)
            # You might need to normalize it if your model was trained with normalized inputs
            # dummy_image = dummy_image / 255.0 # If model expects 0-1 range
            yield [dummy_image] # Yield a list of inputs, one for each input tensor

    # Keep Option 2 code, but don't execute for now as per request.
    # To enable, uncomment the following block:
    '''
    print(f"\n--- Applying Full Integer Quantization ---")
    converter_full_int = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter_full_int.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_full_int.representative_dataset = representative_dataset_gen

    # Optionally, specify input/output types to force full integer compatibility
    # This is often needed for integer-only hardware accelerators like Edge TPU
    # converter_full_int.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter_full_int.inference_input_type = tf.int8
    # converter_full_int.inference_output_type = tf.int8
    # Note: Forcing int8 I/O might require specific preprocessing for input and post-processing for output.

    try:
        tflite_model_full_int = converter_full_int.convert()
        full_int_output_path = "yolov5n_full_int_quant.tflite"
        with open(full_int_output_path, "wb") as f:
            f.write(tflite_model_full_int)
        print(f"Full Integer Quantized TFLite model saved to: {full_int_output_path}")

        # --- Print Model Size and Parameters AFTER Full Integer Quantization ---
        full_int_quant_model_size = get_model_size(full_int_output_path)
        print(f"\n--- Full Integer Quantized Model Statistics ---")
        print(f"Quantized Model Size: {full_int_quant_model_size:.2f} MB")
        print(f"Quantized Model Parameters: Not directly available for TFLite, but typically reduced in effective memory footprint.")

    except Exception as e:
        print(f"Error during Full Integer Quantization: {e}")
        print("This often happens if some operations in the model cannot be fully quantized,")
        print("or if the representative dataset is not correctly provided/formatted.")
    '''