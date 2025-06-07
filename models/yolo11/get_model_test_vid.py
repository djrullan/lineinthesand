from ultralytics import YOLO
import cv2
import time


# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to NCNN format
# model.export(format="ncnn")
model.export(format="ncnn", imgsz=(480,640), device="cpu", int8=True)  # creates '/yolo11n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("./yolo11n_ncnn_model")

input_video_source = "video.mp4"
output_video_path = "annotated_video.mp4"

cap = cv2.VideoCapture(input_video_source)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
source_fps = cap.get(cv2.CAP_PROP_FPS)

video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (frame_width, frame_height))

frames_processed = 0
start_time_processing = time.time()

print(f"Starting video processing for: {input_video_source}")

while True:
    ret, frame = cap.read()
    frames_processed += 1
    if frames_processed % 10 != 0:
        continue
    if not ret:
        break

    results = ncnn_model(frame, verbose=False)
    
    annotated_frame = results[0].plot()

    video_writer.write(annotated_frame)
    
    print(frames_processed)
    

end_time_processing = time.time()

print(f"Finished processing. Annotated video saved to: {output_video_path}")

cap.release()
video_writer.release()

total_processing_time = end_time_processing - start_time_processing

calculated_fps = 0.0
if frames_processed > 0 and total_processing_time > 0:
    calculated_fps = frames_processed / total_processing_time

print(f"Total frames processed: {frames_processed}")
print(f"Total time taken: {total_processing_time:.2f} seconds")
print(f"Frames per second (FPS): {calculated_fps:.2f}")