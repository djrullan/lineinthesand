from ultralytics import YOLO
import cv2
import time

model = YOLO("yolo11n.pt")
################ EDIT BELOW ########
format_str = 'ncnn'
imgsz = (480,640) #jeep this here
keras = False
optimize = False
half = False
int8 = False
dynamic = False
nms = False
batch = 1
device = None
data = 'coco8.yaml'
fraction = 1.0
################ EDIT ABOVE ########

model.export(
    format=format_str,
    imgsz=imgsz,
    keras=keras,
    optimize=optimize,
    half=half,
    int8=int8,
    dynamic=dynamic,
    nms=nms,
    batch=batch,
    device=device,
    data=data,
    fraction=fraction
)

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

print(f"Total frames processed: {200}")
print(f"Total time taken: {total_processing_time:.2f} seconds")

with open('model_quant_logs.txt', 'a') as log_file:
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write the log data to the file
    log_file.write(f"--- Log Entry: {timestamp} ---\n")
    log_file.write(f"Format: {format_str}\n")
    log_file.write(f"Image Size: {imgsz}\n")
    log_file.write(f"Keras: {keras}\n")
    log_file.write(f"Optimize: {optimize}\n")
    log_file.write(f"Half Precision (FP16): {half}\n")
    log_file.write(f"INT8 Quantization: {int8}\n")
    log_file.write(f"Dynamic Axes: {dynamic}\n")
    log_file.write(f"Simplify ONNX: {simplify}\n")
    log_file.write(f"ONNX Opset: {opset}\n")
    log_file.write(f"TensorRT Workspace (GiB): {workspace}\n")
    log_file.write(f"Add NMS: {nms}\n")
    log_file.write(f"Batch Size: {batch}\n")
    log_file.write(f"Device: {device}\n")
    log_file.write(f"Dataset for INT8: {data}\n")
    log_file.write(f"INT8 Calibration Fraction: {fraction}\n")
    log_file.write(f"Total Processing Time: {total_processing_time:.4f} seconds\n")
    log_file.write("--- End of Log Entry ---\n\n")

print("Log information has been successfully written to model_quant_logs.txt")