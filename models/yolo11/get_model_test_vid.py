from ultralytics import YOLO
import cv2
import time
from collections import deque
import numpy as np

# --- Configuration ---
# model_name = "yolov11n"  # Using a base name for easier file management
model = YOLO("yolo11n.pt")
################ EDIT BELOW ########
format_str = 'tflite'
imgsz = (480, 640)
half = False
int8 = False
batch = 1
device = 'cpu'
data = 'coco8.yaml'
fraction = 1
################ EDIT ABOVE ########

# --- Model Export ---
# Define the exported model path string based on configuration
exported_model_filename = f"yolo11n_{format_str}_model"
print(f"think model name is: {exported_model_filename}")

model.export(
    format=format_str,
    imgsz=imgsz,
    half=half,
    # int8=int8,
    batch=batch,
    device=device,
    # data=data,
    # fraction=fraction
)

# --- Model and Video Setup ---
# Load the exported model
print(f"Loading exported model: {exported_model_filename}")
model = YOLO(exported_model_filename)

input_video_source = "video.mp4"
output_video_path = f"annotated_video_{exported_model_filename}.mp4"

cap = cv2.VideoCapture(input_video_source)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
source_fps = cap.get(cv2.CAP_PROP_FPS)

# --- NEW: Data storage for two-pass processing ---
# These lists will store the data from the first pass
inference_times = []
original_frames = []
inference_results = []

# --- Pass 1: Inference and Timing ---
print(f"\nStarting Pass 1: Inference on {input_video_source}")
frame_count = 0
start_time_inference_loop = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break # End of video

    frame_count += 1
    # Process only every 10th frame, as in the original script
    if frame_count % 10 != 0:
        continue

    # Store the original frame for later annotation
    original_frames.append(frame)

    # Time ONLY the inference step
    start_inference = time.time()
    results = model(frame, verbose=False)
    end_inference = time.time()
    
    # Store the results and the time taken
    inference_results.append(results[0]) # Store the first (and only) result object
    inference_times.append(end_inference - start_inference)
    
    print(f"Processed frame {frame_count} | Inference Time: {end_inference - start_inference:.4f}s")

end_time_inference_loop = time.time()
cap.release()
print("--- Pass 1 Finished ---")


# --- Pass 2: Plotting and Writing Video ---
print(f"\nStarting Pass 2: Plotting results and writing to {output_video_path}")
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (frame_width, frame_height))

# Now, iterate through the stored data to create the video
for i, frame_to_annotate in enumerate(original_frames):
    results_for_frame = inference_results[i]
    annotated_frame = results_for_frame.plot()
    video_writer.write(annotated_frame)

video_writer.release()
print("--- Pass 2 Finished ---")
print(f"\nAnnotated video saved to: {output_video_path}")


# --- Performance Calculation and Logging ---
total_inference_time = sum(inference_times)
frames_actually_processed = len(inference_times)

avg_inference_time = 0
inference_fps = 0
if frames_actually_processed > 0:
    avg_inference_time = total_inference_time / frames_actually_processed
    inference_fps = 1 / avg_inference_time

total_processing_time = end_time_inference_loop - start_time_inference_loop

print("\n--- Performance Summary ---")
print(f"Frames processed: {frames_actually_processed}")
print(f"Total inference time: {total_inference_time:.2f} seconds")
print(f"Average inference time per frame: {avg_inference_time:.4f} seconds")
print(f"Inference FPS (1 / avg_inference_time): {inference_fps:.2f}")
print(f"Total time for inference loop (incl. reading frames): {total_processing_time:.2f} seconds")

# --- Log to file ---
with open('model_quant_logs.txt', 'a') as log_file:
    now = time.localtime()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", now)

    log_file.write(f"--- Log Entry: {timestamp} ---\n")
    log_file.write(f"Model File: {exported_model_filename}\n")
    log_file.write(f"Format: {format_str}\n")
    log_file.write(f"Image Size: {imgsz}\n")
    log_file.write(f"Half Precision (FP16): {half}\n")
    log_file.write(f"INT8 Quantization: {int8}\n")
    log_file.write(f"Batch Size: {batch}\n")
    log_file.write(f"Device: {device}\n")
    log_file.write(f"Dataset for INT8: {data}\n")
    log_file.write(f"INT8 Calibration Fraction: {fraction}\n")
    log_file.write("--- Performance ---\n")
    log_file.write(f"Frames Processed: {frames_actually_processed}\n")
    log_file.write(f"Average Inference Time: {avg_inference_time:.4f} seconds\n")
    log_file.write(f"Inference FPS: {inference_fps:.2f}\n")
    log_file.write(f"Total Inference Loop Time: {total_processing_time:.4f} seconds\n")
    log_file.write("--- End of Log Entry ---\n\n")

print("\nLog information has been successfully written to model_quant_logs.txt")