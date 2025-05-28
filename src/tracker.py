#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from sort import Sort
from utils import check_direction, is_intersect

## D code: Logic for detecting ppl inside box
def is_inside_box(center: Tuple[int, int], box: List[Tuple[int, int]]) -> bool:
    (x1, y1), (x2, y2) = box
    return x1 <= center[0] <= x2 and y1 <= center[1] <= y2
##

class Tracker(object):
    def __init__(
        self,
        directions,
        #D Code: 
        box: Optional[List[Tuple[int]]] = None,
        border: Optional[List[Tuple[int]]] = None,
        
        ##
        count_callback: Optional[Callable] = None,
        use_box: bool = False,
    ):
        """Constructor of Tracker.

        Args:
            box (List[Tuple[int]]): Detection zone (top-left, bottom-right).
            border (List[Tuple[int]]): Line coordinates.
            use_box (bool): Whether to use box-based logic.
            directions (Dict[str, Tuple[bool]]): Direction filters.
            count_callback (Callable, optional): Callback when count increases.
        """
        self.tracker = Sort()
        ## D Code: initalize
        self.box = box
        self.border = border
        self.use_box = use_box
        ##

        self.count_callback = count_callback
        self.memory = {}
        self.counter = {key: 0 for key in directions.keys()}
        self.directions = directions

        np.random.seed(2021)
        self.COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    ##def _is_count(
      ##  self,
        ##center: Tuple[int],
        ##center_prev: Tuple[int],
        ##border: List[Tuple[int]],
        ##key: str,
    ##) -> bool:
        """Check whether count or not.

        1. check_direction: Check the direction of human movement.
                            If direction is not specified, return True.
        2. is_intersect: Check whether the border and the human movement intersect.

        Args:
            center(Tuple[int]): Current center position.
            center_prev(Tuple[int]): Previous center position.
            border(List[Tuple[int]]): Border.
            key(str): "inside", "outside" or "total".
        """

        # D code:
        """if self.use_box:
            was_outside = not is_inside_box(center_prev, self.box)
            is_now_inside = is_inside_box(center, self.box)
            return was_outside and is_now_inside
        else:
            return check_direction(center, center_prev, self.directions[key]) and is_intersect(
                center, center_prev, self.border[0], self.border[1]
            )"""
        

    def _update_box_counts(self, center, center_prev):
        was_inside = is_inside_box(center_prev, self.box)
        is_inside = is_inside_box(center, self.box)
        # Entry (outside -> inside)
        if not was_inside and is_inside:
            if 'inside' in self.counter:
                self.counter['inside'] += 1
            if 'total' in self.counter:
                self.counter['total'] += 1
        # Exit (inside -> outside)
        elif was_inside and not is_inside:
            if 'outside' in self.counter:
                self.counter['outside'] += 1
            if 'total' in self.counter:
                self.counter['total'] += 1

    def _update_line_counts(self, center, center_prev):
        # For line mode, count crossings based on direction and intersection
        for key in ['inside', 'outside']:
            dir_filter = self.directions.get(key)
            if check_direction(center, center_prev, dir_filter) and is_intersect(
                center, center_prev, self.border[0], self.border[1]
            ):
                if key in self.counter:
                    self.counter[key] += 1
                if 'total' in self.counter:
                    self.counter['total'] += 1
        ##
        


    def update(self, frame: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """Update tracker and draw bounding box in a frame.

        Args:
            frame (np.ndarray): Target frame.
            dets (np.ndarray): Array like [xyxy + score].

        Returns:
            np.ndarray: Frame with bounding box and count.
        """
        # Update Sort.
        tracks = self.tracker.update(dets)

        boxes = []
        index_ids = []
        previous = self.memory.copy()

        for track in tracks.astype(int):
            boxes.append([track[0], track[1], track[2], track[3]])
            index_ids.append(track[4])
            # Add index id and box to memory.
            self.memory[index_ids[-1]] = boxes[-1]

        ## D code:
        # Draw detection region
        if self.use_box:
            cv2.rectangle(frame, self.box[0], self.box[1], (10, 255, 0), 3)
        else:
            cv2.line(frame, self.border[0], self.border[1], (10, 255, 0), 3)

        # Process each track
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            color = [int(c) for c in self.COLORS[index_ids[i] % len(self.COLORS)]]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            idx = index_ids[i]

            if idx in previous:
                pbox = previous[idx]
                center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
                prev_center = (int((pbox[0] + pbox[2]) / 2), int((pbox[1] + pbox[3]) / 2))
                cv2.line(frame, center, prev_center, color, 2)
                # Update counts
                if self.use_box:
                    self._update_box_counts(center, prev_center)
                else:
                    self._update_line_counts(center, prev_center)
                if self.count_callback:
                    self.count_callback(self.counter)

            cv2.putText(
                frame,
                str(idx),
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Overlay counters
        for i, (key, cnt) in enumerate(self.counter.items()):
            cv2.putText(
                frame,
                f"{key}: {cnt}",
                (30, 40 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (10, 255, 0),
                2,
            )
        return frame
        ###