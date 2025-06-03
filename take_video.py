import cv2
import time
from picamera2 import Picamera2
import os # Import os for output directory handling

# --- Configuration ---
VIDEO_FILENAME = 'quantize_video.mp4'
OUTPUT_DIRECTORY = './outputs/' # Define an output directory
RECORDING_DURATION_SECONDS = 60 # 1 minute
CAMERA_RESOLUTION = (640, 480) # Common resolution for testing
CAMERA_FRAMERATE = 30 # Frames per second

def record_video_picam2():
    """
    Records video for a specified duration using Picamera2 and saves it
    using OpenCV's VideoWriter.
    """
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_DIRECTORY, VIDEO_FILENAME)

    picam2 = None
    writer = None
    
    try:
        print(f"Initializing Picamera2 at {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]} @ {CAMERA_FRAMERATE}fps...")
        picam2 = Picamera2()
        
        # Configure for video recording (BGR888 format is compatible with OpenCV)
        video_config = picam2.create_video_configuration(
            main={"size": CAMERA_RESOLUTION, "format": "BGR888"},
            controls={"FrameRate": CAMERA_FRAMERATE}
        )
        picam2.configure(video_config)
        picam2.start()
        
        # Give the camera a moment to warm up
        time.sleep(1) 

        # Setup OpenCV VideoWriter
        # Use 'MP4V' for .mp4 or 'XVID' for .avi
        fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
        writer = cv2.VideoWriter(output_filepath, fourcc, CAMERA_FRAMERATE, CAMERA_RESOLUTION, True)
        
        if not writer.isOpened():
            raise IOError(f"Could not open video writer for file: {output_filepath}")

        print(f"Starting video recording to {output_filepath} for {RECORDING_DURATION_SECONDS} seconds...")
        
        start_time = time.time()
        frames_recorded = 0
        
        while True:
            # Capture array directly compatible with OpenCV
            frame = picam2.capture_array() 
            
            if frame is None:
                print("Failed to capture frame. Exiting.")
                break
            
            # Write the frame to the output video file
            writer.write(frame)
            frames_recorded += 1
            
            # Print progress without a newline
            elapsed_time = time.time() - start_time
            print(f"Recording: {int(elapsed_time)}s / {RECORDING_DURATION_SECONDS}s ({frames_recorded} frames)", end='\r')

            # Check if recording duration has been reached
            if elapsed_time >= RECORDING_DURATION_SECONDS:
                print(f"\nReached {RECORDING_DURATION_SECONDS} second recording limit. Stopping.")
                break

            # You can add a small delay if the loop runs too fast,
            # but capture_array usually handles this implicitly based on framerate.
            # time.sleep(1 / CAMERA_FRAMERATE) # Optional: if needed to cap FPS precisely

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Provide specific advice for common Picamera2 errors
        if "No camera found" in str(e) or "Failed to open camera" in str(e):
            print("Please ensure your camera module is connected and enabled in raspi-config.")
            print("Also, check if Picamera2 is installed correctly and has permissions.")
    finally:
        if writer:
            writer.release()
            print(f"Video writer released. Video saved to: {output_filepath}")
        if picam2:
            if picam2.started:
                picam2.stop()
            picam2.close()
            print("Picamera2 instance released.")
        cv2.destroyAllWindows()
        print("Script finished.")

if __name__ == "__main__":
    record_video_picam2()