import cv2
import os
from natsort import natsorted

def create_ultrasound_video(image_folder, output_path, fps=5, duplicate_factor=3):
    """
    Creates a slow-motion ultrasound video from a folder of images.

    Parameters:
    - image_folder: Path to folder containing sequential ultrasound images.
    - output_path: Path to save the output video file.
    - fps: Frames per second for the output video (lower = slower).
    - duplicate_factor: Number of times each frame is repeated to slow down playback.
    """
    
    # Collect and sort image filenames
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = natsorted(images)

    if not images:
        raise ValueError("No valid image files found in the folder.")

    # Read first image to get video dimensions
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    if first_frame is None:
        raise ValueError("First image could not be read.")
    height, width, _ = first_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi format
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame multiple times
    for image_name in images:
        frame_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"⚠️ Skipping unreadable image: {image_name}")
            continue
        for _ in range(duplicate_factor):
            video_writer.write(frame)

    video_writer.release()
    print(f"✅ Slow-motion video saved to: {output_path}")

# Example usage
create_ultrasound_video(
    image_folder=r"C:\Users\vasan\OneDrive\Desktop\fetal_pro\main\data\train\1", 
    output_path="ultrasound_video_slow-main.mp4", 
    fps=5, 
    duplicate_factor=3
)
