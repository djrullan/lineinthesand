#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino
# Substantial modifications for static frame boundary selection & performance optimization
#
# Refactored for Low-Latency Edge Inference

import argparse
import os
import time
import threading
import queue
from typing import Dict, Tuple, List
import cProfile
import pstats
import io

import cv2
import numpy as np
from picamera2 import Picamera2

# Import Flask for HTTP streaming
from flask import Flask, Response
import logging

# Disable default Flask logging to keep terminal cleaner
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Assuming these are in relative paths or Python path
from detect import Detect
from tracker import Tracker
from utils import direction_config

# --- Globals for Mouse Interaction (Setup Phase) ---
_boundary_points: List[Tuple[int, int]] = []
_freehand_points: List[Tuple[int, int]] = []
_drawing: bool = False
_all_freehand_segments: List[List[Tuple[int, int]]] = []

# --- Globals for Frame Sharing & Thread Synchronization ---
_latest_processed_frame: np.ndarray = None
_frame_lock = threading.Lock()
_new_frame_condition = threading.Condition(_frame_lock) # Condition variable for signaling new frames

# --- Flask App for Video Streaming ---
app = Flask(__name__)

def generate_mjpeg_stream():
    """Waits for a new frame and streams it as MJPEG."""
    while True:
        with _new_frame_condition:
            # Wait until the main thread signals that a new frame is ready
            _new_frame_condition.wait()
            if _latest_processed_frame is None:
                continue

            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', _latest_processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Endpoint to provide the MJPEG video feed."""
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Mouse Callback Functions (Setup Phase) ---
def _mouse_callback(event, x, y, flags, param):
    global _boundary_points
    if event == cv2.EVENT_LBUTTONDOWN and len(_boundary_points) < 2:
        _boundary_points.append((x, y))
        cv2.circle(param['frame'], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(param['window_name'], param['frame'])

def _freehand_mouse(event, x, y, flags, param):
    global _freehand_points, _drawing, _all_freehand_segments
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _freehand_points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        if not _freehand_points or np.linalg.norm(np.array(_freehand_points[-1]) - np.array((x, y))) > 2:
            _freehand_points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP and _drawing:
        _drawing = False
        if _freehand_points:
            _all_freehand_segments.append(list(_freehand_points))
        _freehand_points = []

# --- Static Frame Acquisition (Setup Phase) ---
def _get_static_frame_for_boundary(src_path: str, live_mode: bool, camera_device_index: int) -> np.ndarray:
    if live_mode:
        with Picamera2(camera_num=camera_device_index) as picam2:
            print("Initializing temporary Picamera2 for single frame capture...")
            config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
            picam2.configure(config)
            picam2.start()
            time.sleep(0.5)
            frame = picam2.capture_array()
            if frame is None:
                raise RuntimeError("Failed to capture frame from Picamera2 for boundary definition.")
            print("Temporary Picamera2 instance released.")
            return frame
    else:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file for boundary definition: {src_path}")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read a frame from the video file: {src_path}")
        return frame

# --- Boundary Definition Functions (Setup Phase) ---
def _get_freehand_boundary_on_static_frame(src_path: str, live_mode: bool, camera_device_index: int, win_name: str) -> list:
    global _freehand_points, _drawing, _all_freehand_segments
    base_frame = _get_static_frame_for_boundary(src_path, live_mode, camera_device_index)
    _freehand_points, _drawing, _all_freehand_segments = [], False, []

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _freehand_mouse)
    print(f"Draw on window '{win_name}'. Press ENTER to finish, ESC to cancel.")

    while True:
        display_frame = base_frame.copy()
        for segment in _all_freehand_segments:
            if len(segment) > 1:
                cv2.polylines(display_frame, [np.array(segment, dtype=np.int32)], False, (0, 255, 0), 2)
        if _drawing and len(_freehand_points) > 1:
            cv2.polylines(display_frame, [np.array(_freehand_points, dtype=np.int32)], False, (0, 255, 255), 2)

        cv2.imshow(win_name, display_frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 13: break
        if key == 27: _all_freehand_segments = []; break

    cv2.destroyWindow(win_name)
    return [point for segment in _all_freehand_segments for point in segment]

def _get_interactive_boundary(src: str, use_box: bool, live: bool, device: int) -> Tuple[list, list]:
    global _boundary_points
    _boundary_points = []
    frame = _get_static_frame_for_boundary(src, live, device)
    window_name = "Define Boundary - Click 2 Points"
    clone = frame.copy()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_callback, {'frame': clone, 'window_name': window_name})

    print(f"Click two points to define the {'box' if use_box else 'line'}. Press ENTER when done.")
    while len(_boundary_points) < 2:
        cv2.imshow(window_name, clone)
        if cv2.waitKey(1) & 0xFF == 13: break

    cv2.destroyWindow(window_name)
    if len(_boundary_points) < 2: return None, None
    p1, p2 = _boundary_points[0], _boundary_points[1]
    return ([p1, p2] if use_box else None, None if use_box else [p1, p2])


# --- Video Stream Classes ---
class BaseStream:
    def __init__(self):
        self._frame_count = 0

    def next(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    @property
    def frame_count(self):
        return self._frame_count

class CameraStream(BaseStream):
    def __init__(self, device_index=0, resolution=(640, 480), framerate=60):
        super().__init__()
        print(f"Initializing CameraStream with Picamera2 (device {device_index})....")
        self.picam2 = Picamera2(camera_num=device_index)
        video_config = self.picam2.create_video_configuration(
            main={"size": resolution, "format": "BGR888"},
            controls={"FrameRate": framerate, "NoiseReductionMode": 2} # Added noise reduction
        )
        self.picam2.configure(video_config)
        self.picam2.start()
        self.framerate = framerate
        print(f"Picamera2 stream started at {resolution[0]}x{resolution[1]} @ {framerate}fps.")

    def next(self) -> Tuple[bool, np.ndarray]:
        frame = self.picam2.capture_array()
        if frame is not None:
            self._frame_count += 1
            return True, frame
        return False, None

    def release(self):
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            print("CameraStream's Picamera2 instance released.")

class VideoStream(BaseStream):
    def __init__(self, src_path):
        super().__init__()
        self.cap = cv2.VideoCapture(src_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {src_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"VideoStream initialized for {src_path}, {self.total_frames} frames at {self.fps:.2f} FPS.")

    def next(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.cap.read()
        if ret:
            self._frame_count += 1
        return ret, frame

    def release(self):
        if self.cap: self.cap.release()

# --- Core Inference Function ---
def _detect_person(detect_obj: Detect, frame: np.ndarray, confidence_thresh: float, iou_thresh: float) -> np.ndarray:
    """Optimized detection function."""
    boxes, scores, class_idx = detect_obj.detect(frame)
    if boxes is None:
        return np.empty((0, 5))

    # Perform NMS. It returns indices of boxes to keep.
    nms_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence_thresh, iou_thresh)

    if nms_indices is None or len(nms_indices) == 0:
        return np.empty((0, 5))

    # Flatten indices and filter for persons (class_id == 0)
    person_mask = (class_idx[nms_indices] == 0)
    if not np.any(person_mask):
        return np.empty((0, 5))
        
    person_indices = nms_indices[person_mask]
    
    # Select boxes and scores for persons
    person_boxes = boxes[person_indices]
    person_scores = scores[person_indices]

    # Convert boxes to xyxy format and scale
    H, W = frame.shape[:2]
    person_boxes = detect_obj.to_xyxy(person_boxes) * np.array([W, H, W, H])
    
    # Concatenate for tracker
    return np.concatenate([person_boxes.astype(int), person_scores.reshape(-1, 1)], axis=1)

# --- Video Writer Thread ---
def video_writer_thread_func(frame_queue: queue.Queue, stop_event: threading.Event, output_path: str, fourcc_code: int, fps: float, frame_size: Tuple[int, int]):
    """Consumes frames from a queue and writes them to a video file."""
    print(f"Video writer thread started. Saving to: {output_path}")
    writer = cv2.VideoWriter(output_path, fourcc_code, fps, frame_size)
    if not writer.isOpened():
        print(f"Error: VideoWriter could not be opened at {output_path}")
        return

    while not stop_event.is_set():
        try:
            # A small timeout prevents the thread from blocking indefinitely
            frame = frame_queue.get(timeout=1.0)
            writer.write(frame)
        except queue.Empty:
            continue # If queue is empty, check stop_event again

    # After stop signal, empty the rest of the queue
    while not frame_queue.empty():
        try:
            writer.write(frame_queue.get_nowait())
        except queue.Empty:
            break
            
    writer.release()
    print("Video writer thread finished.")

# --- Helper to draw ROI ---
def draw_roi(frame: np.ndarray, polygon: list, box: list, border: list):
    """Draws the defined Region of Interest on the frame."""
    if polygon:
        cv2.polylines(frame, [np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)], isClosed=True, color=(255, 0, 255), thickness=2)
    elif box:
        cv2.rectangle(frame, box[0], box[1], (255, 0, 0), 2)
    elif border:
        cv2.line(frame, border[0], border[1], (0, 0, 255), 2)
    return frame

# --- Main Application Logic ---
def main(args: argparse.Namespace):
    global _latest_processed_frame

    # --- Profiling Setup ---
    profiler = cProfile.Profile()
    
    # --- ROI and Tracker Setup ---
    polygon, box, border = None, args.box_coords, args.border_coords
    if args.interactive_freehand:
        polygon = _get_freehand_boundary_on_static_frame(args.src, args.live, args.device, "Draw Freehand ROI")
    elif args.interactive_boundary:
        box, border = _get_interactive_boundary(args.src, args.use_box, args.live, args.device)

    tracker = Tracker(
        directions={k: direction_config.get(v) for k, v in args.directions.items()},
        box=box, border=border, polygon=polygon,
        use_box=bool(box)
    )
    detect_obj = Detect(args.model, args.confidence)
    
    # --- Stream Initialization ---
    stream = CameraStream(args.device) if args.live else VideoStream(args.src)
    
    # --- Output Setup ---
    if not os.path.exists(args.dest): os.makedirs(args.dest, exist_ok=True)
    
    # Get first frame to determine properties
    is_running, frame = stream.next()
    if not is_running:
        print("Failed to get first frame. Exiting.")
        stream.release()
        return

    H, W = frame.shape[:2]
    output_fps = stream.framerate if args.live else stream.fps
    fourcc = cv2.VideoWriter_fourcc(*{"mp4": "MP4V", "avi": "XVID"}.get(args.video_fmt, "XVID"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    video_name = 'live' if args.live else os.path.basename(args.src).split('.')[0]
    out_path = os.path.join(args.dest, f"{video_name}_{ts}.{args.video_fmt}")

    # --- Start Worker Threads ---
    stop_event = threading.Event()
    # Video Saving Thread
    frame_queue = queue.Queue(maxsize=int(output_fps * 2)) # Buffer for ~2 seconds
    writer_thread = threading.Thread(target=video_writer_thread_func, args=(frame_queue, stop_event, out_path, fourcc, output_fps, (W, H)))
    writer_thread.daemon = True
    writer_thread.start()
    
    # Flask Streaming Thread
    print(f"Starting Flask streaming server: http://{args.flask_host}:{args.flask_port}/video_feed")
    flask_thread = threading.Thread(target=app.run, kwargs={'host': args.flask_host, 'port': args.flask_port}, daemon=True)
    flask_thread.start()

    # --- Main Processing Loop ---
    profiler.enable()
    start_time = time.monotonic()
    last_print_time = time.monotonic() # Add this line
    
    try:
        # Process the first frame which was already fetched
        dets = _detect_person(detect_obj, frame, args.confidence, args.iou_threshold)
        processed_frame = tracker.update(frame, dets)
        processed_frame = draw_roi(processed_frame, polygon, box, border)
        frame_queue.put_nowait(processed_frame)
        
        with _new_frame_condition:
            _latest_processed_frame = processed_frame.copy()
            _new_frame_condition.notify_all()

        # Loop over remaining frames
        while True:
            loop_start_time = time.monotonic() # Add this line
            is_running, frame = stream.next()
            if not is_running:
                break
            
            # --- Inference and Tracking ---
            dets = _detect_person(detect_obj, frame, args.confidence, args.iou_threshold)
            processed_frame = tracker.update(frame, dets)
            
            # --- Visualization and Output ---
            processed_frame = draw_roi(processed_frame, polygon, box, border)

            # Push to worker threads
            try:
                frame_queue.put_nowait(processed_frame)
            except queue.Full:
                pass # Drop frame if writer is backed up

            with _new_frame_condition:
                _latest_processed_frame = processed_frame.copy()
                _new_frame_condition.notify_all()
                
                
            loop_time = time.monotonic() - loop_start_time
            instant_fps = 1.0 / loop_time
            # Print FPS stats every second to avoid cluttering the terminal
            print(f"Instant FPS: {instant_fps:.2f}")

    except KeyboardInterrupt:
        print("\nInterruption detected. Shutting down.")
    finally:
        # --- Shutdown ---
        profiler.disable()
        end_time = time.monotonic()
        
        stop_event.set()
        writer_thread.join(timeout=5.0) # Wait for writer to finish
        stream.release()
        cv2.destroyAllWindows()
        
        # --- Performance Report ---
        total_time = end_time - start_time
        processed_frames = stream.frame_count
        avg_fps = processed_frames / total_time
        
        print("\n--- Performance Summary ---")
        print(f"Processed {processed_frames} frames in {total_time:.2f} seconds.")
        print(f"Average FPS: {avg_fps:.2f}")
        print("---------------------------\n")

        # Print detailed profiling stats
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(15) # Print top 15 cumulative time functions
        print("--- Profiling Report (Top 15) ---")
        print(s.getvalue())
        print("---------------------------------")
        print("Application finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Human Detection and Tracking for Edge Devices.")
    parser.add_argument("--src", default="./data/TownCentreXVID.mp4", help="Path to video source.")
    parser.add_argument("--dest", default="./outputs/", help="Path to output directory.")
    parser.add_argument("--model", default="./models/yolov5n6_float16.tflite", help="Path to TFLite model file.")
    parser.add_argument("--video-fmt", choices=["mp4", "avi"], default="mp4", help="Format of output video.")
    parser.add_argument("--confidence", type=float, default=0.4, help="Confidence threshold for detection.")
    parser.add_argument("--iou-threshold", type=float, default=0.4, help="IoU threshold for NMS.")
    parser.add_argument("--directions", default={"total": None}, type=eval, help="Tracker directions config.")
    
    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument("--interactive-freehand", action="store_true", help="Draw a freehand ROI.")
    roi_group.add_argument("--interactive-boundary", action="store_true", help="Draw a line/box ROI.")
    
    parser.add_argument("--use-box", action="store_true", help="If --interactive-boundary, draw a box (else a line).")
    parser.add_argument("--box-coords", type=eval, default=None, help="Predefined box ROI as [(x1,y1),(x2,y2)].")
    parser.add_argument("--border-coords", type=eval, default=None, help="Predefined line ROI as [(x1,y1),(x2,y2)].")
    
    parser.add_argument('--live', action='store_true', help='Use live camera feed (Picamera2).')
    parser.add_argument('--device', type=int, default=0, help='Camera device index for live mode.')
    
    parser.add_argument('--flask-host', type=str, default='0.0.0.0', help='Host IP for the Flask streaming server.')
    parser.add_argument('--flask-port', type=int, default=5000, help='Port for the Flask streaming server.')

    args = parser.parse_args()

    # Input validation
    if not args.live and not os.path.exists(args.src):
        print(f"Error: Source video file '{args.src}' not found. Use --live for camera feed.")
        exit(1)
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        exit(1)

    main(args)