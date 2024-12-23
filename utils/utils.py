import os
import cv2
from ultralytics import YOLO
import numpy as np
import subprocess
import mediapipe as mp

def process_video_with_yolo_and_pose(video_path, target_class=[3]):
    # Load the YOLO model
    model = YOLO("models/best1.pt")

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    pose_estimator = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties for the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up FFMPEG process for writing the video
    file_name_with_extension = os.path.basename(video_path)
    file_name, file_extension = os.path.splitext(file_name_with_extension)
    output_video_path = f"videos/result/{file_name}_output.mp4"

    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Overwrite if file exists
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{frame_width}x{frame_height}', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model(frame)

            names = results[0].names
            boxes = results[0].boxes
            xyxys = boxes.xyxy
            cls = boxes.cls

            for cla, xyxy in zip(cls, xyxys):
                cla_key = int(cla.item())
                tensor_list = xyxy.tolist()
                int_list = [int(element) for element in tensor_list]

                # Check if the detection class matches target_class
                if cla_key in target_class:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = int_list
                    label = names[cla_key]

                    # Crop the detected player region
                    player_crop = frame[y1:y2, x1:x2]

                    # Run pose estimation on the cropped region
                    pose_results = pose_estimator.process(cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB))

                    # Draw the bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # If pose landmarks are detected, overlay them on the original frame
                    if pose_results.pose_landmarks:
                        # Draw pose landmarks and connections
                        mp_drawing.draw_landmarks(
                            frame[y1:y2, x1:x2],
                            pose_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )

            # Write the processed frame to FFMPEG
            ffmpeg_process.stdin.write(frame.tobytes())
        else:
            # Break the loop if the end of the video is reached
            break

    # Close the FFMPEG process
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    # Release the video capture object
    cap.release()

    return output_video_path


def process_video_with_yolo(video_path, target_class=[3]):
    # Load the YOLO model
    model = YOLO("models/best1.pt")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties for the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up FFMPEG process for writing the video
    file_name_with_extension = os.path.basename(video_path)
    file_name, file_extension = os.path.splitext(file_name_with_extension)
    output_video_path = f"videos/result/{file_name}_output.mp4"

    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Overwrite if file exists
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{frame_width}x{frame_height}', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model(frame)
            
            names = results[0].names
            boxes = results[0].boxes
            xyxys = boxes.xyxy
            cls = boxes.cls

            for cla, xyxy in zip(cls, xyxys):
                cla_key = int(cla.item())
                tensor_list = xyxy.tolist()
                int_list = [int(element) for element in tensor_list]

                # Check if the detection class matches target_class
                if cla_key in target_class:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = int_list
                    label = names[cla_key]

                    # Draw the bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Write the processed frame to FFMPEG
            ffmpeg_process.stdin.write(frame.tobytes())
        else:
            # Break the loop if the end of the video is reached
            break

    # Close the FFMPEG process
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    # Release the video capture object
    cap.release()
    os.remove(video_path)

    return output_video_path