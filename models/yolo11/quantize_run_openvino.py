import cv2
import nncf
import openvino as ov
import numpy as np
import time

# --- 1. Setup and Configuration ---

# -- Model and Video Settings
# This should be the path to the XML file of your OpenVINO model
MODEL_PATH = "yolo11n_openvino_model/yolo11n.xml"
# This is the video you want to process
INPUT_VIDEO_PATH = "video.mp4"
# A detailed name for the output file
OUTPUT_VIDEO_FILENAME = "yolo11n_quantized_annotated_video.mp4"

# -- Model Input Dimensions
# The exact HxW dimensions your model expects
TARGET_H, TARGET_W = 480, 640

# -- Post-Processing Settings
# Confidence threshold: ignore detections with a confidence score lower than this
CONF_THRESHOLD = 0.5
# Non-Maximum Suppression (NMS) threshold: a higher value allows for more overlapping boxes
NMS_THRESHOLD = 0.5


# --- 2. Post-Processing and Drawing ---

def post_process(prediction: np.ndarray, original_shape: tuple, resized_shape: tuple) -> list:
    """
    Decodes the raw YOLO model output, applies NMS, and scales bounding boxes.

    Args:
        prediction (np.ndarray): The raw output tensor from the YOLO model.
        original_shape (tuple): The (height, width) of the original video frame.
        resized_shape (tuple): The (height, width) the frame was resized to for the model.

    Returns:
        list: A list of final detections [x1, y1, x2, y2, score, class_id].
    """
    # The output of YOLOv8 is [1, 84, 8400], where 84 = 4 (box) + 80 (classes).
    # We transpose it to [1, 8400, 84] to process detections easily.
    output = prediction.transpose(0, 2, 1)[0]  # Shape: [8400, 84]

    boxes = []
    class_ids = []
    scores = []

    for row in output:
        # The first 4 values are box coordinates (center_x, center_y, width, height)
        # The rest are class scores.
        xc, yc, w, h = row[:4]
        
        # Find the class with the highest score
        class_id = np.argmax(row[4:])
        score = row[4 + class_id]

        if score > CONF_THRESHOLD:
            # Convert from center-width-height to top-left-bottom-right coordinates
            x1 = (xc - w / 2)
            y1 = (yc - h / 2)
            x2 = (xc + w / 2)
            y2 = (yc + h / 2)
            
            boxes.append([x1, y1, x2, y2])
            class_ids.append(class_id)
            scores.append(float(score))

    # Apply Non-Maximum Suppression to filter out overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
    if len(indices) == 0:
        return []

    # Scaling factors to convert coordinates from model space to original video space
    orig_h, orig_w = original_shape
    resized_h, resized_w = resized_shape
    x_scale = orig_w / resized_w
    y_scale = orig_h / resized_h
    
    final_detections = []
    for i in indices.flatten():
        x1, y1, x2, y2 = boxes[i]
        # Scale the coordinates back to the original frame size
        final_detections.append([
            int(x1 * x_scale), int(y1 * y_scale),
            int(x2 * x_scale), int(y2 * y_scale),
            scores[i],
            class_ids[i]
        ])

    return final_detections

def draw_boxes(frame, detections):
    """Draws bounding boxes on a frame."""
    for x1, y1, x2, y2, score, class_id in detections:
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Create the label text
        label = f"Class {class_id}: {score:.2f}"
        # Draw a filled rectangle for the label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        # Put the label text on the background
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


# --- 3. Quantization Pipeline ---
ov_core = ov.Core()
model = ov_core.read_model(MODEL_PATH)

def video_frame_generator(video_path: str, max_frames: int = 300):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open video file: {video_path}")
    frame_count = 0
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret: break
            resized_frame = cv2.resize(frame, (TARGET_W, TARGET_H))
            transposed_frame = resized_frame.transpose((2, 0, 1))
            input_tensor = np.expand_dims(transposed_frame, 0).astype(np.float32)
            yield input_tensor
            frame_count += 1
    finally:
        cap.release()
        print(f"Calibration data generator finished after {frame_count} frames.")

calibration_dataset = nncf.Dataset(video_frame_generator(INPUT_VIDEO_PATH))

print("Starting model quantization...")
quantized_model = nncf.quantize(model=model, calibration_dataset=calibration_dataset, subset_size=300)
print("Model quantization complete.")

# --- 4. Compile Model and Setup Video I/O ---
print("Compiling quantized model for CPU...")
compiled_quantized_model = ov_core.compile_model(quantized_model, device_name="CPU")
print("Model compiled successfully.")

# Setup video capture for inference
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {INPUT_VIDEO_PATH}")

# Get video properties for the output writer
original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Setup the video writer to save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME, fourcc, fps, (original_w, original_h))
print(f"Ready to process video. Output will be saved to '{OUTPUT_VIDEO_FILENAME}'")


# --- 5. Run Inference and Save Video ---
frame_number = 0
latest_detections = [] # Store the last known detections

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # We run inference only on every 10th frame to save computation
    if frame_number % 10 == 0:
        # Preprocess the frame for the model
        resized_frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        transposed_frame = resized_frame.transpose((2, 0, 1))
        input_tensor = np.expand_dims(transposed_frame, 0)

        start_time = time.perf_counter()
        # The output from the compiled model is a dictionary
        results_dict = compiled_quantized_model([input_tensor])
        # Get the raw prediction tensor (adjust key if necessary)
        raw_prediction = next(iter(results_dict.values()))
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000
        print(f"Frame {frame_number}: Inference took {processing_time_ms:.2f} ms")

        # Post-process the output to get clean bounding boxes
        latest_detections = post_process(
            prediction=raw_prediction,
            original_shape=(original_h, original_w),
            resized_shape=(TARGET_H, TARGET_W)
        )

    # Draw the latest detections on the current frame
    # This ensures frames between inference runs still have boxes on them
    if latest_detections:
        frame = draw_boxes(frame, latest_detections)

    # Write the frame (with or without boxes) to the output video file
    video_writer.write(frame)
    frame_number += 1

# --- 6. Cleanup ---
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"\nProcessing complete. Annotated video saved as '{OUTPUT_VIDEO_FILENAME}'")