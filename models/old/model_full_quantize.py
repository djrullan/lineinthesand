import tensorflow as tf
import numpy as np
import os
import cv2 # For video processing

# Path to your SavedModel directory
saved_model_dir = "../../yolov5/yolov5n_saved_model"

# Path to your representative dataset video
video_path = "video.mp4" # Make sure this path is correct

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

    dr_output_path = "yolov5n_dr_quant.tflite"
    with open(dr_output_path, "wb") as f:
        f.write(tflite_model_dr)
    print(f"Dynamic Range Quantized TFLite model saved to: {dr_output_path}")

    # --- Print Model Size and Parameters AFTER Dynamic Range Quantization ---
    dr_quant_model_size = get_model_size(dr_output_path)
    print(f"\n--- Dynamic Range Quantized Model Statistics ---")
    print(f"Quantized Model Size: {dr_quant_model_size:.2f} MB")
    print(f"Quantized Model Parameters: Not directly available for TFLite, but typically reduced in effective memory footprint.")

    # --- Option 2: Full Integer Quantization (Requires Representative Dataset) ---
    print(f"\n--- Applying Full Integer Quantization ---")

    # This requires a function that yields input data for calibration.
    # We will now process frames from video.mp4
    def representative_dataset_gen():
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        num_calibration_samples = 100 # Adjust based on your dataset size and desired calibration quality
        frames_processed = 0
        frame_skip_interval = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / num_calibration_samples)) # Adjust to sample evenly

        print(f"Processing video for representative dataset. Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        print(f"Sampling approximately {num_calibration_samples} frames with an interval of {frame_skip_interval}.")

        while frames_processed < num_calibration_samples:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            # Skip frames to get a diverse set of samples
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_skip_interval != 0 and frames_processed > 0:
                continue

            # Preprocessing: Resize and normalize
            # YOLOv5n typically expects (1, 640, 640, 3) float32 input in RGB format, normalized to 0-1
            # Adjust resolution (640) as per your model's requirement.
            try:
                # Convert BGR (OpenCV default) to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to the model's expected input size
                resized_frame = cv2.resize(rgb_frame, (640, 640))
                # Add batch dimension and normalize to 0-1
                input_tensor = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0

                yield [input_tensor] # Yield a list of inputs, one for each input tensor
                frames_processed += 1
                if frames_processed % 10 == 0:
                    print(f"Processed {frames_processed} calibration samples...")

            except Exception as e:
                print(f"Error processing frame: {e}")
                continue # Skip to the next frame

        cap.release()
        print(f"Finished processing video. Total {frames_processed} calibration samples generated.")


    converter_full_int = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter_full_int.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_full_int.representative_dataset = representative_dataset_gen

    # Optionally, specify input/output types to force full integer compatibility
    # This is often needed for integer-only hardware accelerators like Edge TPU
    # Uncomment the following lines if you specifically target int8 inference
    converter_full_int.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_full_int.inference_input_type = tf.int8
    converter_full_int.inference_output_type = tf.int8
    # Note: Forcing int8 I/O might require specific preprocessing for input and post-processing for output
    # during actual inference, converting to int8/uint8 and scaling.

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
        print("If you encounter errors, try removing `converter_full_int.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`")
        print("and related `inference_input_type`/`inference_output_type` lines to allow for float fallback.")