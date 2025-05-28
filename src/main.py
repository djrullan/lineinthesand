#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

import argparse
import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np

from detect import Detect
from streams import VideoStream, CameraStream
from tracker import Tracker
from utils import direction_config

# D Code: Global to hold user clicks
_boundary_points = []
##Freehand drawing
_freehand_points = []
_drawing = False

## Examples
## Line Crossing: python main.py
## Box entry: python main.py --use-box

def _mouse_callback(event, x, y, flags, param):
    """Mouse callback to record two points."""
    global _boundary_points
    if event == cv2.EVENT_LBUTTONDOWN and len(_boundary_points) < 2:
        _boundary_points.append((x, y))
        # draw a small circle at the clicked point
        cv2.circle(param['frame'], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(param['window_name'], param['frame'])

def _freehand_mouse(event, x, y, flags, param):
    """Mouse callback to capture freehand strokes."""
    global _freehand_points, _drawing
    frame = param['frame']
    win   = param['window_name']
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _freehand_points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        _freehand_points.append((x, y))
        pts = np.array(_freehand_points, dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(frame, [pts], False, (0, 255, 0), 2)
        cv2.imshow(win, frame)
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False


def _get_interactive_boundary(src: str, use_box: bool, live: bool, device: int) -> Tuple[list, list]:
    """
    Open a single frame from the source and let user click two points
    to define a border (line) or box (top-left, bottom-right).
    Returns:
        box: List of two (x,y) points or None
        border: List of two (x,y) points or None
    """
    # Read a single frame from the source
    if live:
        cap = cv2.VideoCapture(device)
    else:
        cap = cv2.VideoCapture(src)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame from {'camera' if live else src}")

    window_name = "Define Boundary"
    clone = frame.copy()
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, clone)

    # set mouse callback with frame in param
    cv2.setMouseCallback(window_name, _mouse_callback, {'frame': clone, 'window_name': window_name})
    print(f"Click two points to define the {'box' if use_box else 'line'}. Press any key after selecting.")

    # Wait until two points have been clicked
    while len(_boundary_points) < 2:
        cv2.waitKey(1)
    cv2.waitKey(500)
    cv2.destroyWindow(window_name)

    p1, p2 = _boundary_points[0], _boundary_points[1]
    if use_box:
        return [p1, p2], None
    else:
        return None, [p1, p2]
###


def _get_freehand_boundary(src: str, live: bool, device: int) -> list:
    """
    Show a frame and let the user draw an arbitrary freehand polygon.
    Finish by pressing ENTER.
    Returns:
        polygon: List of (x,y) points in drawing order.
    """
    # grab one frame
    cap = cv2.VideoCapture(device) if live else cv2.VideoCapture(src)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to grab setup frame for freehand ROI")

    win = "Draw ROI – press ENTER to finish"
    clone = frame.copy()
    cv2.namedWindow(win)
    cv2.imshow(win, clone)

    # reset and bind
    global _freehand_points, _drawing
    _freehand_points = []
    _drawing = False
    cv2.setMouseCallback(win, _freehand_mouse, {'frame': clone, 'window_name': win})

    # wait for ENTER (keycode 13)
    while True:
        if cv2.waitKey(1) & 0xFF == 13:
            break
    cv2.destroyWindow(win)
    return list(_freehand_points)
##

def _detect_person(
    detect: Detect,
    frame: np.ndarray,
    confidence: float,
    iou_threshold: float,
) -> np.ndarray:
    """Detect person objects in a frame.

    Returns:
        np.ndarray: Array like [xyxy, score].
    """

    # Detect objects in the frame.
    boxes, scores, class_idx = detect.detect(frame)

    # NMS
    idx = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou_threshold)
    boxes = boxes[idx]
    scores = scores[idx]
    class_idx = class_idx[idx]

    # Filter only person object (class index = 0).
    person_idx = np.where(class_idx == 0)[0]
    boxes = boxes[person_idx]
    scores = scores[person_idx]

    # Scale boxes by frame size.
    H, W = frame.shape[:2]
    boxes = detect.to_xyxy(boxes) * np.array([W, H, W, H])

    # dets:  [xmin, ymin, xmax, ymax, score]
    dets = np.concatenate([boxes.astype(int), scores.reshape(-1, 1)], axis=1)
    return dets


def main(
    src: str,
    dest: str,
    model: str,
    video_fmt: str,
    confidence: float,
    iou_threshold: float,
    directions: Dict[str, Tuple[bool]],
    ## D Code: Code for flagg
    use_box: bool,
    box: list,
    border: list,
    interactive_boundary: bool,
    interactive_freehand: bool,
    live: bool,
    device: int,
    ##
):
    # D code Select stream source
    if live:
        stream = CameraStream(device)
        print(f"Using live camera stream (device {device})")
    else:
        stream = VideoStream(src)
        print(f"Using video file: {src}")
    

    """Track human objects and count the number of human.

    Args:
        src (str): Source video.
        dest (str): Directory to save results.
        model (str): Path to tflite weight.
        confidence (float): Confidence threshold.
        iou_threshold (float): IoU threshold for NMS.
    """

    polygon = None
    if interactive_freehand:
        # freehand ROI
        polygon = _get_freehand_boundary(src, live, device)
        box = None
        border = None
    elif interactive_boundary:
        # Clear any prior clicks
        global _boundary_points
        _boundary_points = []
        # Get box or border interactively
        box, border = _get_interactive_boundary(src, use_box, live, device)

    
    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)

    # The line to count.
    
    #border = [(0, 500), (1920, 500)]
    #box = [(800, 400), (1100, 600)]

    #directions = {key: direction_config.get(d_str) for key, d_str in directions.items()}
    directions = {k: direction_config.get(v) for k, v in directions.items()}

    ## D code: Initialize tracker based on flag
    ##if use_box:
    ##    tracker = Tracker(box=box, border=None, use_box=True, directions=directions)
    ##else:
    ##   tracker = Tracker(box=None, border=border, use_box=False, directions=directions)
    
    ## Commeted out above for interactive UI
    tracker = Tracker(directions=directions, box=box, border=border, polygon=polygon, use_box=use_box)
    ## 

    detect = Detect(model, confidence)
    ##stream = VideoStream(src)
    writer = None

    # D Code:
    try:
        total_frames = len(stream)
    except TypeError:
        total_frames = None
    if total_frames is not None:
        print(f"Total frames: {total_frames}")
   ##


    while True:
        # Read the next frame from stream.
        is_finish, frame = stream.next()
        # height, width, _ = frame.shape
        # print(f"Frame dimensions: Width = {width}, Height = {height}")
        if not is_finish:
            break

        start = time.time()
        dets = _detect_person(detect, frame, confidence, iou_threshold)
        end = time.time()
        ##frame = tracker.update(frame, dets)

        ## D code:
        frame = tracker.update(frame, dets)

        ## Live Preview##
        cv2.imshow("Live Detection", frame)
        # press 'q' in that window to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

        # Update tracker and draw bounding boxes in frame.
        # dets:  [xmin, ymin, xmax, ymax, score]
        ## D code: Test to see if frame is correct size
        print("Frame size:", frame.shape)   # -> (H, W, _)

        # Executed only first time.
        if writer is None:
            # Initialize video writer.
            model_name = os.path.basename(model).split(".")[0] 
            video_name = os.path.basename(src).split(".")[0] if not live else 'camera'
            
            # add timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basename = f"{video_name}_{model_name}_{timestamp}"
            
            output_video = os.path.join(dest, f"{basename}.{video_fmt}")
            fourcc = cv2.VideoWriter_fourcc(*{"mp4": "MP4V", "avi": "MJPG"}[video_fmt])
            writer = cv2.VideoWriter(output_video, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            # Estimate total time.
            second_per_frame = end - start
            print(f"Computation time per a frame: {second_per_frame:.4f} seconds")
            if total_frames is not None:
                print(f"Estimated total time: {second_per_frame * total_frames:.4f}")

        # Save frame as an image and video.
        cv2.imwrite(os.path.join(dest, f"{basename}.jpg"), frame)
        writer.write(frame)

    if writer:
        writer.release()
    stream.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Path to video source.", default="./data/TownCentreXVID.mp4")
    parser.add_argument("--dest", help="Path to output directory", default="./outputs/")
    parser.add_argument("--model", help="Path to YOLOv5 tflite file", default="./models/yolov5n6-fp16.tflite")
    parser.add_argument("--video-fmt", help="Format of output video file.", choices=["mp4", "avi"], default="mp4")
    parser.add_argument("--confidence", type=float, default=0.2, help="Confidence threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.2, help="IoU threshold for NMS.")
    parser.add_argument("--directions", default={"total": None}, type=eval, help="Directions")

    ## D code: New arguments for logic switching
    parser.add_argument("--use-box", action="store_true", help="Use bounding box instead of line")
    parser.add_argument("--box", type=eval, default=[(50,50),(200,200)], help="Bounding box as [(x1,y1),(x2,y2)]")
    parser.add_argument("--border", type=eval, default=[(0,180),(640,180)], help="Line border as [(x1,y1),(x2,y2)]")
    parser.add_argument("--interactive-freehand", action="store_true", help="Draw an arbitrary freehand ROI before processing")
    parser.add_argument("--interactive-boundary", action="store_true", help="Enable interactive drawing of line/box before processing loop")
    parser.add_argument('--live', action='store_true', help='Use live camera feed instead of video file')
    parser.add_argument('--device', type=int, default=0, help='Camera device index for live mode')
    ##

    args = vars(parser.parse_args())
    main(**args)
