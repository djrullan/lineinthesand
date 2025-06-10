import argparse
import time
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

# --- NEW: Import Picamera2 for live mode ---
try:
    from picamera2 import Picamera2
    pi_camera_available = True
except ImportError:
    pi_camera_available = False
    print("Warning: picamera2 library not found. --live mode will be unavailable.")


def process_video_file(model, input_path, output_path):
    """
    Original two-pass logic for processing a video file.
    This is ideal for benchmarking and creating a high-quality annotated video.
    """
    print(f"--- Starting File Processing Mode ---")
    print(f"Input video: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    # --- Data storage for two-pass processing ---
    inference_times = []
    original_frames = []
    inference_results = []

    # --- Pass 1: Inference and Timing ---
    print(f"\nStarting Pass 1: Inference on {input_path}")
    frame_count = 0
    start_time_inference_loop = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1
        # Process only every 10th frame to save time
        if frame_count % 10 != 0:
            continue

        original_frames.append(frame)
        start_inference = time.time()
        results = model(frame, verbose=False)
        end_inference = time.time()

        inference_results.append(results[0])
        inference_times.append(end_inference - start_inference)
        print(f"Processed frame {frame_count} | Inference Time: {end_inference - start_inference:.4f}s")

    end_time_inference_loop = time.time()
    cap.release()
    print("--- Pass 1 Finished ---")

    # --- Pass 2: Plotting and Writing Video ---
    print(f"\nStarting Pass 2: Plotting results and writing to {output_path}")
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (frame_width, frame_height))

    for i, frame_to_annotate in enumerate(original_frames):
        results_for_frame = inference_results[i]
        annotated_frame = results_for_frame.plot()
        video_writer.write(annotated_frame)

    video_writer.release()
    print("--- Pass 2 Finished ---")
    print(f"\nAnnotated video saved to: {output_path}")

    # --- Performance Calculation and Logging ---
    total_inference_time = sum(inference_times)
    frames_actually_processed = len(inference_times)
    avg_inference_time = total_inference_time / frames_actually_processed if frames_actually_processed > 0 else 0
    inference_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
    total_processing_time = end_time_inference_loop - start_time_inference_loop

    print("\n--- Performance Summary ---")
    print(f"Frames processed: {frames_actually_processed}")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Average inference time per frame: {avg_inference_time:.4f} seconds")
    print(f"Inference FPS (based on pure inference): {inference_fps:.2f}")
    print(f"Total time for inference loop (incl. I/O): {total_processing_time:.2f} seconds")


def process_live_feed(model):
    """
    NEW: Single-pass logic for processing a live Picamera2 feed.
    This captures, infers, annotates, and displays in a loop for real-time performance.
    """
    if not pi_camera_available:
        print("Error: picamera2 library is required for --live mode but is not installed.")
        return

    print(f"--- Starting Live Processing Mode ---")
    print("Press 'q' in the display window to quit.")
    
    # --- Camera Setup ---
    picam2 = Picamera2()
    # Configure for a preview size that's manageable for real-time inference
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1) # Allow camera to warm up

    # --- Real-time Performance Tracking ---
    fps_deque = deque(maxlen=30) # Store last 30 frame times for smooth FPS calculation
    
    while True:
        start_frame_time = time.time()
        
        # Capture frame and convert color from RGB (picamera2) to BGR (OpenCV)
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # --- Inference ---
        results = model(frame_bgr, verbose=False)
        
        # --- Annotation and Display ---
        annotated_frame = results[0].plot()

        # Calculate and display FPS
        end_frame_time = time.time()
        frame_time = end_frame_time - start_frame_time
        fps_deque.append(frame_time)
        avg_fps = len(fps_deque) / sum(fps_deque)

        print(f"instant fps: {1/frame_time:.2f}")
        # Put FPS text on the frame

        cv2.imshow("Live Inference - Press 'q' to quit", annotated_frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    picam2.stop()
    cv2.destroyAllWindows()
    print("--- Live Mode Stopped ---")


if __name__ == '__main__':
    # --- NEW: Argument Parsing ---
    parser = argparse.ArgumentParser(description="YOLO Object Detection on Video or Live Feed.")
    parser.add_argument(
        '--live', 
        action='store_true', 
        help="Use live video from Picamera2 instead of a video file."
    )
    args = parser.parse_args()

    # --- Model Loading (common to both modes) ---
    exported_model_filename = "/home/djrul/lineinthesand/models/yolo11/yolo11n_fp16_openvino_model"
    print(f"Loading exported model: {exported_model_filename}")
    try:
        model = YOLO(exported_model_filename)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # --- Main Logic Branch ---
    if args.live:
        process_live_feed(model)
    else:
        # These are only needed for file mode
        input_video_source = "models/video.mp4"
        output_video_path = f"annotated_video_{exported_model_filename.split('/')[-1]}.mp4"
        process_video_file(model, input_video_source, output_video_path)

    # Note: The original logging section is omitted as it was commented out and specific
    # to the file processing metrics. You can add logging within each function if needed.