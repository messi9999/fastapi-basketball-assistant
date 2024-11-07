import os
import cv2
from ultralytics import YOLO

def process_video_with_yolo(video_path):
    # Load the YOLO model
    model = YOLO("models/best1.pt")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties for the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' if you prefer
    file_name_with_extension = os.path.basename(video_path)
    file_name, file_extension = os.path.splitext(file_name_with_extension)
    output_video_path = f"videos/result/{file_name}_output{file_extension}"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Write the annotated frame to the output video file
            out.write(annotated_frame)
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and the VideoWriter object
    cap.release()
    out.release()
    os.remove(video_path)

    return output_video_path