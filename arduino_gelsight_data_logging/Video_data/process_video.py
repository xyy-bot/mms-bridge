import cv2
import numpy as np
import os


def is_dynamic_frame(current_frame, previous_frame, pixel_threshold=25, dynamic_ratio_threshold=0.05):
    """
    Determine if the current frame is dynamic by comparing it to the previous frame.
    This function converts both frames to grayscale, computes the absolute difference,
    thresholds the difference to identify significantly changed pixels, and calculates
    the ratio of changed pixels to the total number of pixels.

    Args:
        current_frame (ndarray): Current frame (BGR image).
        previous_frame (ndarray): Previous frame (BGR image).
        pixel_threshold (int): The per-pixel intensity difference required to mark a pixel as changed.
        dynamic_ratio_threshold (float): The minimum ratio of changed pixels (0 to 1) required to consider the frame dynamic.

    Returns:
        (bool, float): A tuple where the first element is True if the frame is dynamic, and the
                       second element is the computed ratio of changed pixels.
    """
    # Convert frames to grayscale
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(gray_current, gray_previous)

    # Threshold the difference image to mark significantly changed pixels
    _, diff_thresh = cv2.threshold(diff, pixel_threshold, 255, cv2.THRESH_BINARY)

    # Calculate the ratio of changed pixels to the total number of pixels
    non_zero_count = np.count_nonzero(diff_thresh)
    total_pixels = diff_thresh.size
    ratio = non_zero_count / total_pixels

    return ratio > dynamic_ratio_threshold, ratio


def extract_dynamic_frames(video_path, output_folder, pixel_threshold=25, dynamic_ratio_threshold=0.05):
    """
    Extract dynamic frames from a video file and save them as JPEG images.

    The detection compares each frame (starting with frame 5) to its predecessor.
    Additionally, the very last dynamic frame detected is ignored (i.e. not kept in the output).

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder where the dynamic frames will be saved.
        pixel_threshold (int): Threshold for per-pixel change.
        dynamic_ratio_threshold (float): Minimum ratio of changed pixels to consider the frame dynamic.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0
    previous_frame = None
    saved_frames = []  # To store the filenames of saved dynamic frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1

        # For the first 4 frames, assume they are static and only initialize the previous frame.
        if frame_count < 5:
            previous_frame = frame
            continue

        # Check if the current frame is dynamic compared to the previous frame.
        dynamic, ratio = is_dynamic_frame(frame, previous_frame, pixel_threshold, dynamic_ratio_threshold)
        if dynamic:
            # Save the dynamic frame as a JPEG file with a sequential filename.
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved dynamic frame {frame_count:04d} (changed ratio: {ratio:.4f}) as {frame_filename}")
            saved_frames.append(frame_filename)
            saved_frame_count += 1

        previous_frame = frame  # Update the previous frame for the next iteration

    cap.release()

    # If at least one dynamic frame was saved, ignore (delete) the last one.
    if saved_frames:
        last_frame_filename = saved_frames[-1]
        try:
            os.remove(last_frame_filename)
            saved_frame_count -= 1
            print(f"Ignored the last dynamic frame by removing {last_frame_filename}")
        except Exception as e:
            print(f"Error removing the last dynamic frame {last_frame_filename}: {e}")

    print(f"Extraction complete: {saved_frame_count} dynamic frames saved from {video_path}")


# -------------------------
# Set input and output paths
# -------------------------
video_directory = './'  # Folder containing your video files (e.g. .mov, .mp4, .avi)
processed_directory = './processed/dynamic_frames/'  # Base folder for saving extracted dynamic frames

# Iterate over all video files in the video directory
for video_filename in os.listdir(video_directory):
    if video_filename.lower().endswith(('.mov', '.mp4', '.avi')):
        video_path = os.path.join(video_directory, video_filename)
        # Create a subfolder named after the video (without extension)
        video_basename = os.path.splitext(video_filename)[0]
        output_folder = os.path.join(processed_directory, video_basename)
        extract_dynamic_frames(video_path, output_folder,
                               pixel_threshold=3,
                               dynamic_ratio_threshold=0.01)

print("All videos processed.")
