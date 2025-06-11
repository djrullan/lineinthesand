import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
import threading
from flask import Flask, Response

# --- Import Picamera2 for live mode ---
try:
    from picamera2 import Picamera2
    pi_camera_available = True
except ImportError:
    pi_camera_available = False
    print("Warning: picamera2 library not found. Live mode will be unavailable.")

# --- Global variables for sharing data between threads ---
output_frame = None
lock = threading.Lock()

# --- Flask App Initialization (No changes here) ---
app = Flask(__name__)

# --- Polygon Drawing Functions (No changes here) ---
polygon_points = []
current_mouse_pos = (0, 0)
# (Your mouse_callback_drag and draw_polygon_on_frame functions go here, unchanged)
def mouse_callback_drag(event, x, y, flags, param):
    global polygon_points, current_mouse_pos
    current_mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"Point added: ({x}, {y}). Press 'd' when done, 'c' to clear.")

def draw_polygon_on_frame(frame):
    # This function requires a local display (monitor or VNC) to work
    global polygon_points, current_mouse_pos
    polygon_points = []
    clone = frame.copy()
    window_name = "Draw Polygon - Click points, press 'd' when done, 'c' to clear"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback_drag)
    print("\n--- Polygon Drawing Mode (Requires Local Display) ---")
    print("Click to define polygon corners. A line will follow your cursor.")
    print("Press 'd' for done, 'c' for clear, 'q' to quit.")
    while True:
        temp_frame = clone.copy()
        if len(polygon_points) > 0:
            for point in polygon_points:
                cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
        if len(polygon_points) > 1:
            cv2.polylines(temp_frame, [np.array(polygon_points)], isClosed=False, color=(0, 255, 0), thickness=2)
        if len(polygon_points) > 0:
            cv2.line(temp_frame, polygon_points[-1], current_mouse_pos, (255, 255, 0), 2)
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            if len(polygon_points) < 3:
                print("Error: Polygon needs at least 3 points.")
                polygon_points = []
            else:
                break
        elif key == ord('c'):
            polygon_points = []
            print("Points cleared.")
        elif key == ord('q'):
            polygon_points = []
            break
    cv2.destroyWindow(window_name)
    return np.array(polygon_points, dtype=np.int32) if len(polygon_points) > 2 else None

# --- Flask Routes (No changes here) ---
@app.route("/")
def index():
    # ... (unchanged)
    return """
    <html><head><title>Live Video Stream</title></head>
    <body><h1>Live Inference Stream</h1><img src="/video_feed" width="640" height="480"></body></html>
    """

def generate_frames():
    # ... (unchanged)
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # ... (unchanged)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# --- NEW: Streamer Abstraction Classes ---
class LiveStreamer:
    """A wrapper for Picamera2 to provide a consistent interface."""
    def __init__(self):
        if not pi_camera_available:
            raise RuntimeError("Picamera2 library not found, cannot use live stream.")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (640, 480)}, controls={"FrameRate": 30})
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.0)
        print("Live camera stream started.")

    def get_frame(self):
        frame = self.picam2.capture_array()
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def stop(self):
        self.picam2.stop()
        print("Live camera stream stopped.")
    
    def reset(self):
        # Not applicable for live streams, but included for interface consistency.
        pass

class FileStreamer:
    """A wrapper for cv2.VideoCapture to provide a consistent interface."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {path}")
        self.path = path
        print(f"File stream started for: {self.path}")

    def get_frame(self):
        return self.cap.read()

    def stop(self):
        self.cap.release()
        print(f"File stream stopped for: {self.path}")

    def reset(self):
        # Resets the video to the first frame.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- RENAMED & REFACTORED: Main Processing Function ---
def run_inference_stream(model, streamer, track_mode=False, save_video=False):
    global output_frame, lock
    video_writer = None

    try:
        # --- Start Flask Server (works for both modes) ---
        flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False))
        flask_thread.daemon = True
        flask_thread.start()
        print("Flask server started. Open http://<your_pi_ip>:5000 to view.")

        polygon = None
        if track_mode:
            ret, first_frame = streamer.get_frame()
            if not ret:
                print("Could not get the first frame to draw polygon.")
                return
            polygon = draw_polygon_on_frame(first_frame)
            if polygon is None:
                print("No polygon drawn. Exiting.")
                return
            people_counter = 0
            counted_track_ids = set()
            # Reset the stream to the beginning if it's a file
            streamer.reset()

        # --- Main Inference Loop (Now Generic) ---
        print("\nStarting main inference loop. Press Ctrl+C to stop.")
        while True:
            start_frame_time = time.time()
            ret, frame_bgr = streamer.get_frame()
            if not ret:
                print("End of video stream.")
                break # Exit loop when video ends or stream fails
            
            # --- Perform Inference and Annotation (this logic is unchanged) ---
            if track_mode:
                results = model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", classes=0, verbose=False)
                annotated_frame = results[0].plot()
                cv2.polylines(annotated_frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        detection_point = (float(center_x), float(center_y))
                        cv2.circle(annotated_frame, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
                        if track_id not in counted_track_ids and cv2.pointPolygonTest(polygon, detection_point, False) >= 0:
                            people_counter += 1
                            print(f"SECURITY ALERT: PERSON INSIDE ZONE")
                            counted_track_ids.add(track_id)
                cv2.putText(annotated_frame, f"Inside Count: {people_counter}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            else:
                results = model(frame_bgr, verbose=False)
                annotated_frame = results[0].plot()
            
            # Share frame with Flask and save to video
            with lock:
                output_frame = annotated_frame.copy()
            if save_video:
                if video_writer is None:
                    h, w, _ = annotated_frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_filename = "processed_video.mp4"
                    video_writer = cv2.VideoWriter(output_filename, fourcc, 20.0, (w, h))
                    print(f"Video saving enabled. Output: {output_filename}")
                video_writer.write(annotated_frame)
            
            # Print performance
            frame_time = time.time() - start_frame_time
            if frame_time > 0:
                print(f"Main Thread FPS: {1.0 / frame_time:.2f}")

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down gracefully...")
    finally:
        print("\nReleasing resources...")
        streamer.stop()
        if video_writer:
            video_writer.release()
            print(f"Video file finalized and saved to {output_filename}.")
        print("Shutdown complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Object Tracking with Web Stream.")
    # --- NEW: --input argument added ---
    parser.add_argument('--input', type=str, help="Path to a video file. If not provided, uses live camera.")
    parser.add_argument('--track', action='store_true', help="Enable person counting with a user-drawn polygon.")
    parser.add_argument('--save-video', action='store_true', help="Save the output to a video file.")
    args = parser.parse_args()

    # --- Load Model ---
    exported_model_filename = "/home/djrul/lineinthesand/models/yolo11/yolo11n_fp16_openvino_model"
    print(f"Loading exported model: {exported_model_filename}")
    try:
        model = YOLO(exported_model_filename)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # --- UPDATED: Create the correct streamer based on input ---
    streamer = None
    try:
        if args.input:
            streamer = FileStreamer(args.input)
        else:
            streamer = LiveStreamer()
        
        run_inference_stream(model, streamer, track_mode=args.track, save_video=args.save_video)

    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error initializing stream: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")