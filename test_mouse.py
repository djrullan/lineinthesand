import cv2
import numpy as np
import time

def mouse_callback(event, x, y, flags, param):
    print(f"Mouse event: {event} at ({x},{y})")

window_name = "Test Window"
dummy_frame = np.zeros((400, 600, 3), dtype=np.uint8) + 100 # Grey image

cv2.namedWindow(window_name)
cv2.imshow(window_name, dummy_frame)

# Explicitly wait for window to be ready
print("Waiting for window to be ready...")
start_time = time.time()
while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
    if time.time() - start_time > 10:
        print("Timeout waiting for window.")
        exit()
    cv2.waitKey(1)
print("Window is ready. Attempting to set mouse callback.")

try:
    cv2.setMouseCallback(window_name, mouse_callback, {})
    print("Mouse callback set successfully.")
except cv2.error as e:
    print(f"OpenCV Error setting mouse callback: {e}")

print("Press 'q' to quit.")
while True:
    cv2.imshow(window_name, dummy_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Exiting.")