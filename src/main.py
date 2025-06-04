#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino
# Substantial modifications for static frame boundary selection

import argparse
import os
import time
import threading
import queue
from typing import Dict, Tuple, List

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

# --- Globals for Mouse Interaction ---
_boundary_points: List[Tuple[int, int]] = [] # For line/box boundary

_freehand_points: List[Tuple[int, int]] = []        # Holds points for the segment currently being drawn
_drawing: bool = False      # True if mouse button is down and drawing a segment
_all_freehand_segments: List[List[Tuple[int, int]]] = [] # Holds all completed freehand segments

_picam2_stream_instance: Picamera2 = None # Global Picamera2 instance for main streaming

# --- Global for frame sharing with Flask ---
_latest_processed_frame: np.ndarray = None
_frame_lock = threading.Lock() # To safely update/read _latest_processed_frame

# --- Flask App for Video Streaming ---
app = Flask(__name__)

def generate_mjpeg_stream():
    """Generates an MJPEG stream from the latest processed frame."""
    while True:
        with _frame_lock:
            if _latest_processed_frame is not None:
                # Encode the frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', _latest_processed_frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # Adjust sleep to control stream FPS (e.g., ~30 FPS)

@app.route('/video_feed')
def video_feed():
    """Endpoint to provide the MJPEG video feed."""
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Mouse Callback for Line/Box Boundary ---
def _mouse_callback(event, x, y, flags, param):
    global _boundary_points
    if event == cv2.EVENT_LBUTTONDOWN and len(_boundary_points) < 2:
        _boundary_points.append((x, y))
        cv2.circle(param['frame'], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(param['window_name'], param['frame'])

# --- Mouse Callback for Freehand Drawing (works on static or live frame data) ---
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
        _freehand_points.append((x, y))
        if _freehand_points: # Save if any points were captured
            _all_freehand_segments.append(list(_freehand_points))
        _freehand_points = []

# --- Function to Get a Single Static Frame for Boundary Definition ---
def _get_static_frame_for_boundary(src_path: str, live_mode: bool, camera_device_index: int) -> np.ndarray:
    frame = None
    if live_mode:
        picam2_temp = None
        print("Initializing temporary Picamera2 for single frame capture...")
        try:
            picam2_temp = Picamera2(camera_num=camera_device_index)
            config = picam2_temp.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
            picam2_temp.configure(config)
            picam2_temp.start()
            time.sleep(0.5) 
            frame = picam2_temp.capture_array()
            if frame is None:
                raise RuntimeError("Failed to capture frame from Picamera2 for boundary definition.")
        finally:
            if picam2_temp:
                if picam2_temp.started:
                    picam2_temp.stop()
                picam2_temp.close() 
            print("Temporary Picamera2 instance for frame capture released.")
    else: 
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file for boundary definition: {src_path}")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame from video file for boundary definition: {src_path}")
    
    if frame is None: 
        raise RuntimeError("Could not obtain any frame for boundary drawing.")
    return frame

# --- Redesigned Function to Draw Freehand Boundary on a Single Static Frame ---
def _get_freehand_boundary_on_static_frame(src_path: str, live_mode: bool, camera_device_index: int, win_name: str) -> list:
    global _freehand_points, _drawing, _all_freehand_segments

    try:
        base_frame = _get_static_frame_for_boundary(src_path, live_mode, camera_device_index)
    except Exception as e:
        print(f"Error acquiring frame for freehand boundary: {e}")
        return [] 

    _freehand_points = []
    _drawing = False
    _all_freehand_segments = []

    cv2.namedWindow(win_name)
    cv2.waitKey(1) 
    cv2.setMouseCallback(win_name, _freehand_mouse)

    print(f"Draw your freehand boundary on the static frame in window '{win_name}'.")
    print("Each drag creates a segment. Press ENTER to finish, ESC to cancel.")

    while True:
        display_frame = base_frame.copy()

        for segment in _all_freehand_segments:
            if len(segment) > 1:
                cv2.polylines(display_frame, [np.array(segment, dtype=np.int32)], False, (0, 255, 0), 2)
            elif len(segment) == 1:
                cv2.circle(display_frame, segment[0], 3, (0, 255, 0), -1)
        
        if _drawing and len(_freehand_points) > 0:
            cv2.circle(display_frame, _freehand_points[0], 3, (0, 0, 255), -1)
            if len(_freehand_points) > 1:
                cv2.polylines(display_frame, [np.array(_freehand_points, dtype=np.int32)], False, (0, 255, 255), 2)

        cv2.imshow(win_name, display_frame)
        key = cv2.waitKey(20) & 0xFF 
        
        if key == 13:   # ENTER
            break
        elif key == 27:   # ESC
            _all_freehand_segments = [] 
            print("Freehand drawing cancelled.")
            break
            
    cv2.destroyWindow(win_name)
    flattened_points = [point for segment in _all_freehand_segments for point in segment]
    return flattened_points

# --- Function to Get Interactive Line/Box Boundary on a Single Static Frame ---
def _get_interactive_boundary(src: str, use_box: bool, live: bool, device: int) -> Tuple[list, list]:
    global _boundary_points
    _boundary_points = [] 

    try:
        frame = _get_static_frame_for_boundary(src, live, device)
    except Exception as e:
        print(f"Error acquiring frame for interactive boundary: {e}")
        return None, None

    window_name = "Define Boundary - Click 2 Points"
    clone = frame.copy()
    cv2.namedWindow(window_name)
    cv2.waitKey(1) 
    cv2.setMouseCallback(window_name, _mouse_callback, {'frame': clone, 'window_name': window_name})
    cv2.imshow(window_name, clone)
    
    print(f"Click two points to define the {'box' if use_box else 'line'}. Press ENTER after selecting.")
    while len(_boundary_points) < 2:
        if cv2.waitKey(1) & 0xFF == 13:
            break
    
    cv2.destroyWindow(window_name)

    if len(_boundary_points) < 2:
        print("Less than two points selected. Boundary might be incomplete.")
        return None, None
    p1, p2 = _boundary_points[0], _boundary_points[1]
    return ([p1, p2] if use_box else None, None if use_box else [p1, p2])


# --- CameraStream Class for Main Video Processing ---
class CameraStream:
    def __init__(self, device_index=0, resolution=(640, 480), framerate=60):
        global _picam2_stream_instance
        self.resolution = resolution
        self.framerate = framerate
        print(f"Initializing CameraStream with Picamera2 (device {device_index})....")
        
        if _picam2_stream_instance is None:
            _picam2_stream_instance = Picamera2(camera_num=device_index)
            video_config = _picam2_stream_instance.create_video_configuration(
                main={"size": self.resolution, "format": "BGR888"}, 
                controls={"FrameRate": self.framerate}
            )
            _picam2_stream_instance.configure(video_config)
            _picam2_stream_instance.start()
            print(f"Picamera2 stream started at {self.resolution[0]}x{self.resolution[1]} @ {self.framerate}fps.")
        else:
            print("Picamera2 instance already running. Re-using existing instance.")
        
        self.picam2 = _picam2_stream_instance
        self._frame_count = 0

    def next(self):
        if self.picam2 is None or not self.picam2.started:
            print("Error: Picamera2 stream is not active or not initialized.")
            return False, None
        
        frame = self.picam2.capture_array() 
        if frame is not None:
            self._frame_count += 1
            return True, frame
        return False, None

    def release(self):
        global _picam2_stream_instance
        if _picam2_stream_instance is not None:
            print("Releasing CameraStream's Picamera2 instance.")
            if _picam2_stream_instance.started:
                _picam2_stream_instance.stop()
            _picam2_stream_instance.close()
            _picam2_stream_instance = None
            print("CameraStream's Picamera2 instance released.")

# --- VideoStream Class for Main Video Processing from File ---
class VideoStream:
    def __init__(self, src_path):
        self.cap = cv2.VideoCapture(src_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {src_path}")
        self.src_path = src_path
        self._frame_count = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"VideoStream initialized for {src_path}, {self.total_frames} frames at {self.fps} FPS.")

    def next(self):
        ret, frame = self.cap.read()
        if ret:
            self._frame_count += 1
            return True, frame
        return False, None

    def release(self):
        if self.cap:
            self.cap.release()
            print(f"VideoStream for {self.src_path} released.")

    def __len__(self): # Video files have a known length
        return self.total_frames

# --- Object Detection Function ---
def _detect_person(detect_obj: Detect, frame: np.ndarray, confidence_thresh: float, iou_thresh: float) -> np.ndarray:
    confidence_thresh = 0.01 # This was set to 0.01 in your original code
    
    boxes, scores, class_idx = detect_obj.detect(frame)
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 5))

    nms_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence_thresh, iou_thresh)
    
    if isinstance(nms_indices, tuple) and len(nms_indices) == 0: 
        return np.empty((0, 5))
    if hasattr(nms_indices, 'flatten'):
        nms_indices = nms_indices.flatten()
    
    if len(nms_indices) == 0:
        return np.empty((0,5))

    boxes = boxes[nms_indices]
    scores = scores[nms_indices]
    class_idx = class_idx[nms_indices]

    person_idx = np.where(class_idx == 0)[0] 
    if len(person_idx) == 0:
        return np.empty((0,5))
        
    boxes = boxes[person_idx]
    scores = scores[person_idx]

    H, W = frame.shape[:2]
    boxes = detect_obj.to_xyxy(boxes) * np.array([W, H, W, H])
    dets = np.concatenate([boxes.astype(int), scores.reshape(-1, 1)], axis=1)
    return dets

# --- Video Writer Thread Function ---
def video_writer_thread_func(frame_queue: queue.Queue, stop_event: threading.Event, output_path: str, fourcc_code: int, fps: float, frame_size: Tuple[int, int]):
    """
    Function to be run in a separate thread for writing video frames to a file.
    It consumes frames from a queue.
    """
    print(f"Video writer thread started. Saving to: {output_path}")
    writer = None
    try:
        writer = cv2.VideoWriter(output_path, fourcc_code, fps, frame_size, True)
        if not writer.isOpened():
            print(f"Error: VideoWriter could not be opened at {output_path}")
            return

        while True:
            try:
                # Get frame from queue with a timeout to allow checking stop_event
                # A small timeout prevents the thread from blocking indefinitely if the stop_event is set
                frame = frame_queue.get(timeout=0.1) 
                writer.write(frame)
            except queue.Empty:
                # If the queue is empty and the stop event is set, it means no more frames are coming
                if stop_event.is_set():
                    print("Video writer thread received stop signal and queue is empty. Exiting.")
                    break
                # If queue is empty but not stopping, continue waiting
                continue
            except Exception as e:
                print(f"Error writing frame in video writer thread: {e}")
                break # Exit on any write error

    except Exception as e:
        print(f"Error initializing VideoWriter in thread: {e}")
    finally:
        if writer:
            writer.release()
            print("Video writer thread released VideoWriter.")


# --- Main Application Logic ---
def main(
    src: str, dest: str, model: str, video_fmt: str,
    confidence: float, iou_threshold: float, directions: Dict[str, Tuple[bool]],
    use_box: bool, box_coords: list, border_coords: list, 
    interactive_freehand: bool, interactive_boundary: bool,
    live: bool, device: int,
    flask_host: str, flask_port: int): # Added Flask host/port args

    global _latest_processed_frame, _frame_lock

    polygon = None
    final_box_for_tracker = box_coords
    final_border_for_tracker = border_coords
    stream = None
    video_writer_thread = None # Initialize thread handle
    frame_queue = queue.Queue() # Queue for passing frames to the writer thread
    stop_event = threading.Event() # Event to signal the writer thread to stop
    
    try:
        # Interactive boundary definition still happens in the main thread
        if interactive_freehand:
            print("Starting interactive freehand ROI definition on a static frame...")
            polygon = _get_freehand_boundary_on_static_frame(src, live, device, "Draw Freehand ROI - ENTER to Finish")
            if polygon:
                final_box_for_tracker = None 
                final_border_for_tracker = None
                print(f"Freehand polygon defined with {len(polygon)} points.")
            else:
                print("No freehand polygon was defined.")
        elif interactive_boundary:
            print("Starting interactive line/box ROI definition on a static frame...")
            final_box_for_tracker, final_border_for_tracker = _get_interactive_boundary(src, use_box, live, device)
            if final_box_for_tracker: print(f"Box defined: {final_box_for_tracker}")
            if final_border_for_tracker: print(f"Border defined: {final_border_for_tracker}")
        
        if not os.path.exists(dest):
            os.makedirs(dest, exist_ok=True)

        directions_cfg = {k: direction_config.get(v) for k, v in directions.items()}

        tracker = Tracker(
            directions=directions_cfg, 
            box=final_box_for_tracker, 
            border=final_border_for_tracker, 
            polygon=polygon, 
            use_box=use_box if final_box_for_tracker else False 
        )
        print(f"Using detection model: {model}")
        detect_obj = Detect(model, confidence)
        
        if live:
            stream = CameraStream(device_index=device, resolution=(640, 480))
        else:
            stream = VideoStream(src)
        
        frame_display_count = 0
        total_frames_from_stream = 0
        if isinstance(stream, VideoStream):
            total_frames_from_stream = stream.total_frames
        
        initial_start_time = time.time()
        
        # Start the Flask server in a separate thread
        print(f"Starting Flask streaming server on http://{flask_host}:{flask_port}/video_feed")
        flask_thread = threading.Thread(target=app.run, kwargs={'host': flask_host, 'port': flask_port, 'debug': False})
        flask_thread.daemon = True
        flask_thread.start()
        
        # --- Capture and process the first frame to get video properties ---
        is_running, frame = stream.next()
        if not is_running:
            print("Failed to get first frame from stream. Exiting.")
            return # Exit if no first frame

        # Get frame properties for video writer
        frame_height, frame_width = frame.shape[:2]
        frame_size = (frame_width, frame_height)
        
        # Determine FPS for the video writer
        output_fps = 30 # Default FPS for output video
        if live and isinstance(stream, CameraStream):
            output_fps = stream.framerate
        elif not live and isinstance(stream, VideoStream):
            output_fps = stream.fps

        # Setup output video path and FourCC code
        model_name_base = os.path.basename(model).split(".")[0]
        video_name_base = 'live_feed' if live else os.path.basename(src).split(".")[0]
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        output_filename_base = f"{video_name_base}_{model_name_base}_{timestamp_str}"
        output_video_path = os.path.join(dest, f"{output_filename_base}.{video_fmt}")
        
        fourcc_map = {"mp4": "MP4V", "avi": "XVID"}
        fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(video_fmt, "XVID"))

        # Start the video writer thread
        video_writer_thread = threading.Thread(
            target=video_writer_thread_func, 
            args=(frame_queue, stop_event, output_video_path, fourcc, output_fps, frame_size)
        )
        video_writer_thread.daemon = True # Allow the thread to exit when the main program exits
        video_writer_thread.start()
        print(f"\nOutput video saving to: {output_video_path}")

        # Process the first frame and add to queue
        frame_display_count += 1
        inference_start_time = time.time()
        dets = _detect_person(detect_obj, frame, confidence, iou_threshold)
        processed_frame = tracker.update(frame, dets) 
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time
        
        if polygon:
            if len(polygon) > 1: 
                cv2.polylines(processed_frame, [np.array(polygon, dtype=np.int32).reshape(-1,1,2)], isClosed=True, color=(255,0,255), thickness=1)
        elif final_box_for_tracker:
            cv2.rectangle(processed_frame, final_box_for_tracker[0], final_box_for_tracker[1], (255,0,0), 2)
        elif final_border_for_tracker:
            cv2.line(processed_frame, final_border_for_tracker[0], final_border_for_tracker[1], (0,0,255), 2)

        # Put the processed frame into the queue for the writer thread
        frame_queue.put(processed_frame.copy())
        
        # Update the global frame for Flask streaming
        with _frame_lock:
            _latest_processed_frame = processed_frame.copy()

        # Main processing loop
        while True: 
            frame_start_time = time.time() # Start time for the entire frame processing

            is_running, frame = stream.next()
            if not is_running:
                break # End of stream or error

            frame_display_count += 1
            if total_frames_from_stream > 0: # Only print progress if total_frames is known
                print(f"Processing frame {frame_display_count}/{total_frames_from_stream}", end='\r')

            # --- Inference Time Measurement ---
            inference_start_time = time.time()
            dets = _detect_person(detect_obj, frame, confidence, iou_threshold)
            processed_frame = tracker.update(frame, dets) 
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            
            # Draw ROI on the processed frame
            if polygon:
                if len(polygon) > 1: 
                    cv2.polylines(processed_frame, [np.array(polygon, dtype=np.int32).reshape(-1,1,2)], isClosed=True, color=(255,0,255), thickness=1)
            elif final_box_for_tracker:
                cv2.rectangle(processed_frame, final_box_for_tracker[0], final_box_for_tracker[1], (255,0,0), 2)
            elif final_border_for_tracker:
                cv2.line(processed_frame, final_border_for_tracker[0], final_border_for_tracker[1], (0,0,255), 2)

            # Put the processed frame into the queue for the writer thread
            frame_queue.put(processed_frame.copy()) # Use .copy() to ensure thread safety
            
            # Update the global frame for Flask streaming
            with _frame_lock:
                _latest_processed_frame = processed_frame.copy()
            
            frame_end_time = time.time() # End time for the entire frame processing
            total_frame_time = frame_end_time - frame_start_time
            total_since_start = frame_end_time - initial_start_time
            
            print(f"frame{frame_display_count}: time: {total_frame_time:.4f}s | infer: {inference_time:.4f}s | overall: {total_since_start:.4f} | instantFPS: {1/total_frame_time:.4f}")
    
        if total_frames_from_stream > 0 : print() 

    except Exception as e:
        print(f"\nAn error occurred in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Signal the writer thread to stop and wait for it to finish
        if stop_event:
            stop_event.set()
        if video_writer_thread and video_writer_thread.is_alive():
            video_writer_thread.join()
            print("Video writer thread joined.")

        if stream is not None: # Explicitly check if stream object exists
            stream.release()
            print("Video stream released.")
        cv2.destroyAllWindows() # Ensures any remaining interactive windows are closed
        print("Application finished.")

# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Detection and Tracking with ROI definition.")
    parser.add_argument("--src", help="Path to video source.", default="./data/TownCentreXVID.mp4")
    parser.add_argument("--dest", help="Path to output directory.", default="./outputs/")
    parser.add_argument("--model", help="Path to TFLite model file.", default="./models/yolov5n6_float16.tflite")
    parser.add_argument("--video-fmt", help="Format of output video (mp4, avi).", choices=["mp4", "avi"], default="mp4")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold for detection.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for NMS.")
    parser.add_argument("--directions", default={"total": None}, type=eval, help="Tracker directions config e.g. '{\"total\":None}'")

    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument("--interactive-freehand", action="store_true", help="Draw a freehand ROI on a static frame.")
    roi_group.add_argument("--interactive-boundary", action="store_true", help="Draw a line/box ROI on a static frame.")
    
    parser.add_argument("--use-box", action="store_true", help="If --interactive-boundary, draw a box (else a line).")
    parser.add_argument("--box", dest="box_coords", type=eval, default=None, help="Predefined box ROI as [(x1,y1),(x2,y2)].")
    parser.add_argument("--border", dest="border_coords", type=eval, default=None, help="Predefined line ROI as [(x1,y1),(x2,y2)].")

    parser.add_argument('--live', action='store_true', help='Use live camera feed (Picamera2).')
    parser.add_argument('--device', type=int, default=0, help='Camera device index for live mode.')

    # New arguments for Flask server
    parser.add_argument('--flask-host', type=str, default='0.0.0.0', help='Host IP for the Flask streaming server.')
    parser.add_argument('--flask-port', type=int, default=5000, help='Port for the Flask streaming server.')
    
    args = parser.parse_args()
    
    if not args.interactive_freehand and not args.interactive_boundary and not args.box_coords and not args.border_coords:
        print("Warning: No ROI defined (interactive or pre-defined). Tracker might not count specific crossings/entries.")

    if not args.live and not os.path.exists(args.src):
        print(f"Error: Source video file '{args.src}' not found. Specify a valid --src or use --live.")
        exit(1)
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found. Specify a valid --model path.")
        exit(1)

    main(**vars(args))
