import cv2
import time
import numpy as np # Import numpy for frame.shape

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")
    print("Attempting to read frames WITHOUT displaying (headless test)...")
    frame_count = 0
    successful_reads = 0
    for i in range(20): # Try reading more frames to be sure
        ret, frame = cap.read()
        if ret:
            successful_reads += 1
            print(f"  SUCCESS: Frame {i+1} read. Resolution: {frame.shape[1]}x{frame.shape[0]}, Pixels: {np.sum(frame)}")
            # To be absolutely sure it's not a black frame, you can check pixel sum
            # A black frame will have a pixel sum of 0 (assuming 0-255 range).
            if np.sum(frame) > 0:
                print("    (Frame is not entirely black)")
                # If you just want to grab one good frame, you can break here
                # break
            else:
                print("    (Warning: Frame is entirely black. May still be initialization issue.)")
        else:
            print(f"  FAIL: Failed to read frame {i+1}.")
        time.sleep(0.1) # Small delay to give camera time

    cap.release()
    print(f"\nFinished headless test. Total successful reads: {successful_reads} out of {i+1} attempts.")
    if successful_reads > 0:
        print("Conclusion: Camera IS reading frames. The issue is with displaying via X-forwarding.")
    else:
        print("Conclusion: Camera is still not reading frames. There's a deeper camera access issue.")