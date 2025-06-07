import tensorflow as tf
import cv2
import numpy as np

def representative_dataset_gen():

        cap = cv2.VideoCapture("video.mp4")
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        num_calibration_samples = 500 # Adjust based on your dataset size and desired calibration quality
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
        
#----------------------------------------------------------------

saved_model_dir = "yolov5n_saved_model"

representative_dataset_gen()
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#--------------------edit below

# do nothing
# tflite_quant_model = converter.convert()
# path = "yolov11n_nothing.tflite"

# full integer; dont force int8
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# path = "yolov11n_full_int.tflite"

# #float16 quant 
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_quant_model = converter.convert()
# path = "yolov5n_float16_quantized.tflite"

# #dynamic range we used before
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quant_model = converter.convert()
# path = "yolov5n_dynamic.tflite"

with open(path, "wb") as f:
        f.write(tflite_quant_model)
print(f"TFLite model saved to: {path}")

