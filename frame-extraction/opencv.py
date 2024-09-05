import cv2

def split_video_into_frames(video_path, output_dir):
    """Splits a video into individual frames and saves them to the specified directory.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the extracted frames.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_path = f"{output_dir}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

# Example usage:
video_path = 
output_dir = 
split_video_into_frames(video_path, output_dir)
