import cv2
import time
from picamera2 import Picamera2
import os

# --- Configuration ---
VIDEO_FILENAME = 'video.mp4'
OUTPUT_DIRECTORY = './outputs/'
# Use --frames from libcamera-vid, not duration
TARGET_FRAMES_TO_RECORD = 2000 
CAMERA_RESOLUTION = (1920, 1080) # Full HD
CAMERA_FRAMERATE = 50 # Higher framerate as requested

def record_and_display_video_picam2():
    """
    Records video for a specified number of frames using Picamera2, 
    displays it in real-time, and saves it using OpenCV's VideoWriter.
    """
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_DIRECTORY, VIDEO_FILENAME)

    picam2 = None
    writer = None
    
    try:
        print(f"Initializing Picamera2 at {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]} @ {CAMERA_FRAMERATE}fps...")
        picam2 = Picamera2()
        
        # Configure for video recording (BGR888 format is compatible with OpenCV)
        # Add denoise control if desired
        video_config = picam2.create_video_configuration(
            main={"size": CAMERA_RESOLUTION, "format": "BGR888"},
            controls={"FrameRate": CAMERA_FRAMERATE} # DenoiseMode 0 = Off
        )
        picam2.configure(video_config)
        picam2.start()
        
        time.sleep(1) # Give the camera a moment to warm up

        # Using MP4V for .mp4 containers, common and broadly compatible
        fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
        writer = cv2.VideoWriter(output_filepath, fourcc, CAMERA_FRAMERATE, CAMERA_RESOLUTION, True)
        
        if not writer.isOpened():
            raise IOError(f"Could not open video writer for file: {output_filepath}")

        print(f"Starting video recording to {output_filepath} for {TARGET_FRAMES_TO_RECORD} frames...")
        print("Press 'q' on the video window to stop recording early.")
        
        start_time = time.time()
        frames_recorded = 0
        
        # Variables for actual FPS calculation during recording
        last_fps_check_time = time.time()
        frames_since_last_check = 0
        
        while frames_recorded < TARGET_FRAMES_TO_RECORD:
            frame = picam2.capture_array() 
            
            if frame is None:
                print("Failed to capture frame. Exiting.")
                break
            
            # Write the frame to the output video file
            writer.write(frame)
            frames_recorded += 1
            
            # Display the frame
            #cv2.imshow("Live Recording Feed", frame)
            
            # Calculate and print actual FPS during recording
            frames_since_last_check += 1
            current_time = time.time()
            if current_time - last_fps_check_time >= 1.0: # Update FPS every second
                actual_fps = frames_since_last_check / (current_time - last_fps_check_time)
                # Clear previous line and print new info
                print(f"\rRecording: {frames_recorded}/{TARGET_FRAMES_TO_RECORD} frames | Live FPS: {actual_fps:.2f}         ", end='')
                frames_since_last_check = 0
                last_fps_check_time = current_time

            # Handle user input for early exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n'q' pressed. Stopping recording early.")
                break
            
        print(f"\nFinished recording {frames_recorded} frames.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
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

    return output_filepath

def analyze_video_properties(video_path: str):
    """
    Analyzes and prints the properties (FPS, dimensions, total frames, color) of a video file.
    """
    print(f"\nAnalyzing video properties for: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path} for analysis.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # To determine color (channels), read one frame
    ret, frame = cap.read()
    if ret:
        channels = frame.shape[2] if len(frame.shape) == 3 else 1 # 3 for color, 1 for grayscale
        color_info = "Color (BGR)" if channels == 3 else "Grayscale"
    else:
        channels = "N/A"
        color_info = "Could not determine (failed to read frame)"

    print(f"  - Frames Per Second (FPS) (from file metadata): {fps:.2f}")
    print(f"  - Dimensions: {width}x{height}")
    print(f"  - Color Format: {color_info} ({channels} channels)")
    print(f"  - Total Frames in file: {total_frames}")

    if fps > 0 and total_frames > 0:
        actual_duration_from_file = total_frames / fps
        print(f"  - Calculated Video Duration: {actual_duration_from_file:.2f} seconds")
    else:
        print("  - Cannot calculate video duration: FPS or total frames invalid.")

    cap.release()

if __name__ == "__main__":
    recorded_video_path = record_and_display_video_picam2()
    if os.path.exists(recorded_video_path):
        analyze_video_properties(recorded_video_path)
    else:
        print(f"Error: Recorded video file not found at {recorded_video_path}. Cannot analyze properties.")