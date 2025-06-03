#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino
# Substantial modifications for static frame boundary selection

import argparse
import os
import time
from typing import Dict, Tuple, List

import cv2
import numpy as np
from picamera2 import Picamera2 # Import Picamera2 for camera handling

# Assuming these are in relative paths or Python path
from detect import Detect
from tracker import Tracker
from utils import direction_config

# --- Globals for Mouse Interaction ---
_boundary_points: List[Tuple[int, int]] = [] # For line/box boundary

_freehand_points: List[Tuple[int, int]] = []    # Holds points for the segment currently being drawn
_drawing: bool = False      # True if mouse button is down and drawing a segment
_all_freehand_segments: List[List[Tuple[int, int]]] = [] # Holds all completed freehand segments

_picam2_stream_instance: Picamera2 = None # Global Picamera2 instance for main streaming

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
    def __init__(self, device_index=0, resolution=(640, 480), framerate=30):
        global _picam2_stream_instance
        self.resolution = resolution
        self.framerate = framerate
        print(f"Initializing CameraStream with Picamera2 (device {device_index})...")
        
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

    # No __len__ method, as live streams don't have a fixed length known in advance.
    # An instance of CameraStream will evaluate to True in boolean contexts by default.

# --- VideoStream Class for Main Video Processing from File ---
class VideoStream:
    def __init__(self, src_path):
        self.cap = cv2.VideoCapture(src_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {src_path}")
        self.src_path = src_path
        self._frame_count = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"VideoStream initialized for {src_path}, {self.total_frames} frames.")

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

# --- Main Application Logic ---
def main(
    src: str, dest: str, model: str, video_fmt: str,
    confidence: float, iou_threshold: float, directions: Dict[str, Tuple[bool]],
    use_box: bool, box_coords: list, border_coords: list, 
    interactive_freehand: bool, interactive_boundary: bool,
    live: bool, device: int):

    polygon = None
    final_box_for_tracker = box_coords
    final_border_for_tracker = border_coords
    stream = None
    writer = None

    try:
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
        if isinstance(stream, VideoStream):
            total_frames_from_stream = stream.total_frames
        else: # For CameraStream or other types where total frames isn't applicable/known beforehand
            total_frames_from_stream = 0

        while True:
            frame_start_time = time.time() # Start time for the entire frame processing

            is_running, frame = stream.next()
            if not is_running:
                break
            
            frame_display_count += 1
            if total_frames_from_stream > 0: # Only print progress if total_frames is known
                print(f"Processing frame {frame_display_count}/{total_frames_from_stream}", end='\r')

            # --- Inference Time Measurement ---
            inference_start_time = time.time()
            dets = _detect_person(detect_obj, frame, confidence, iou_threshold)
            frame = tracker.update(frame, dets) 
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            
            if polygon:
                if len(polygon) > 1: 
                    cv2.polylines(frame, [np.array(polygon, dtype=np.int32).reshape(-1,1,2)], isClosed=True, color=(255,0,255), thickness=1)
            elif final_box_for_tracker:
                cv2.rectangle(frame, final_box_for_tracker[0], final_box_for_tracker[1], (255,0,0), 2)
            elif final_border_for_tracker:
                cv2.line(frame, final_border_for_tracker[0], final_border_for_tracker[1], (0,0,255), 2)

            # --- Display Time Measurement ---
            display_start_time = time.time()
            #cv2.imshow("Object Detection and Tracking", frame)
            #key = cv2.waitKey(1) & 0xFF 
            display_end_time = time.time()
            display_time = display_end_time - display_start_time

            #if key == ord('q'):
            #    break
            
            if writer is None: 
                model_name_base = os.path.basename(model).split(".")[0]
                video_name_base = 'live_feed' if live else os.path.basename(src).split(".")[0]
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                output_filename_base = f"{video_name_base}_{model_name_base}_{timestamp_str}"
                output_video_path = os.path.join(dest, f"{output_filename_base}.{video_fmt}")
                
                fourcc_map = {"mp4": "MP4V", "avi": "XVID"}
                fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(video_fmt, "XVID"))
                writer = cv2.VideoWriter(output_video_path, fourcc, 25, (frame.shape[1], frame.shape[0]), True) 
                print(f"\nOutput video saving to: {output_video_path}")
                
            writer.write(frame) # Write the frame to the output video

            frame_end_time = time.time() # End time for the entire frame processing
            total_frame_time = frame_end_time - frame_start_time

            print(f"Frame {frame_display_count}: Total Frame Time: {total_frame_time:.4f}s | Inference Time: {inference_time:.4f}s | Display Time: {display_time:.4f}s")
    
        if total_frames_from_stream > 0 : print() 

    except Exception as e:
        print(f"\nAn error occurred in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if writer:
            writer.release()
            print("Video writer released.")
        if stream is not None: # Explicitly check if stream object exists
            stream.release()
            print("Video stream released.")
        cv2.destroyAllWindows()
        print("All OpenCV windows destroyed.")
        print("Application finished.")

# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Detection and Tracking with ROI definition.")
    parser.add_argument("--src", help="Path to video source.", default="./data/TownCentreXVID.mp4")
    parser.add_argument("--dest", help="Path to output directory.", default="./outputs/")
    parser.add_argument("--model", help="Path to TFLite model file.", default="./models/yolov5n6_float16.tflite")
    parser.add_argument("--video-fmt", help="Format of output video (mp4, avi).", choices=["mp4", "avi"], default="mp4")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold for detection.")
    parser.add_argument("--iou-threshold", type=float, default=0.4, help="IoU threshold for NMS.")
    parser.add_argument("--directions", default={"total": None}, type=eval, help="Tracker directions config e.g. '{\"total\":None}'")

    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument("--interactive-freehand", action="store_true", help="Draw a freehand ROI on a static frame.")
    roi_group.add_argument("--interactive-boundary", action="store_true", help="Draw a line/box ROI on a static frame.")
    
    parser.add_argument("--use-box", action="store_true", help="If --interactive-boundary, draw a box (else a line).")
    parser.add_argument("--box", dest="box_coords", type=eval, default=None, help="Predefined box ROI as [(x1,y1),(x2,y2)].")
    parser.add_argument("--border", dest="border_coords", type=eval, default=None, help="Predefined line ROI as [(x1,y1),(x2,y2)].")

    parser.add_argument('--live', action='store_true', help='Use live camera feed (Picamera2).')
    parser.add_argument('--device', type=int, default=0, help='Camera device index for live mode.')
    
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