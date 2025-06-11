import cv2
from ultralytics import NAS
import os
import torch
import time # Import the time module for FPS calculation

# --- Configuration ---
VIDEO_PATH = "video.mp4"
OUTPUT_VIDEO_PATH = "annotated_video.mp4"
ONNX_MODEL_NAME = "yolo_nas_s.onnx"
FRAME_SKIP = 10  # Process every Nth frame
CONFIDENCE_THRESHOLD = 0.25 # Minimum confidence for a detection to be considered
IOU_THRESHOLD = 0.7       # Intersection over Union threshold for Non-Maximum Suppression

# --- 1. Load the YOLO-NAS model ---
print("Loading YOLO-NAS-s model...")
try:
    model = NAS("yolo_nas_s.pt")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'yolo_nas_s.pt' is available or check your internet connection for download.")
    exit()

# --- 2. Export the model to ONNX ---
if not os.path.exists(ONNX_MODEL_NAME):
    print(f"Exporting model to ONNX: {ONNX_MODEL_NAME}...")
    try:
        # Note: 'imgsz' (image size) is important for ONNX export.
        # Default is usually 640. You might adjust based on your model's training.
        model.export(format="onnx", filename=ONNX_MODEL_NAME, imgsz=640)
        print(f"Model exported to {ONNX_MODEL_NAME}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        print("ONNX export failed. The script will continue using the original PyTorch model.")
else:
    print(f"ONNX model already exists: {ONNX_MODEL_NAME}. Skipping export.")

# --- 3. Set up video processing ---
print(f"Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}. Please check the path and filename.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height} pixels, {fps} FPS (input video), {total_frames} frames.")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For MP4, 'mp4v' or 'XVID' often works
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    print(f"Error: Could not create output video file {OUTPUT_VIDEO_PATH}.")
    print("Ensure you have the necessary codecs installed or try a different output format (e.g., '.avi' with 'XVID').")
    cap.release()
    exit()

# --- 4. Process frames ---
frame_count = 0
processed_inference_frames = 0 # Counter for frames where inference was actually run
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for inference: {device}")

print("Starting video processing...")
start_time = time.time() # Record the start time

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break

    annotated_frame = frame.copy() # Initialize annotated_frame with the original frame

    # Check if this is a frame to process
    if frame_count % FRAME_SKIP == 0:
        print(f"Processing frame {frame_count+1}/{total_frames} (running inference)...")
        processed_inference_frames += 1

        # Run inference on the current frame
        results = model.predict(
            source=frame,
            show=False,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=device
        )

        # Annotate the frame with detection results
        if results:
            annotated_frame = results[0].plot()

    # Write the (potentially annotated) frame to the output video
    out.write(annotated_frame)

    frame_count += 1

    # Optional: Display processing progress if you want a real-time window
    # cv2.imshow("Annotated Video", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
    #     break

end_time = time.time() # Record the end time
elapsed_time = end_time - start_time

print(f"Finished processing. Read {frame_count} frames. Ran inference on {processed_inference_frames} frames.")

# --- 5. Calculate and print FPS ---
if elapsed_time > 0:
    overall_processing_fps = frame_count / elapsed_time
    inference_run_fps = processed_inference_frames / elapsed_time
    print(f"\nOverall Processing FPS (reading, processing, writing): {overall_processing_fps:.2f} FPS")
    print(f"Inference Run FPS (how often inference was called): {inference_run_fps:.2f} FPS")
else:
    print("\nProcessing completed too quickly to calculate meaningful FPS.")

# --- 6. Clean up ---
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Annotated video saved successfully to {OUTPUT_VIDEO_PATH}")