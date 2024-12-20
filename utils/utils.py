import os
import cv2
from ultralytics import YOLO
import numpy as np
import subprocess

# def process_video_with_yolo(video_path, target_class=[3]):
#     # Load the YOLO model
#     model = YOLO("models/best1.pt")

#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Get video properties for the output video
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or 'XVID' if you prefer
#     file_name_with_extension = os.path.basename(video_path)
#     file_name, file_extension = os.path.splitext(file_name_with_extension)
#     output_video_path = f"videos/result/{file_name}_output.mp4"
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     # Loop through the video frames
#     while cap.isOpened():
#         # Read a frame from the video
#         success, frame = cap.read()

#         if success:
#             # Run YOLO inference on the frame
#             results = model(frame)
            
#             names = results[0].names
#             boxes = results[0].boxes
#             xyxys = boxes.xyxy
#             cls = boxes.cls
            
#             # print(names[0])
           
#             for cla, xyxy in zip(cls, xyxys):
#                 cla_key = int(cla.item())
#                 tensor_list = xyxy.tolist()
#                 int_list = [int(element) for element in tensor_list]

#                 # Assuming results is a list of detections
#                 # Check if the detection class name matches target_class
#                 if cla_key in target_class:
#                     # Extract xyxy bounding box coordinates
#                     x1, y1, x2, y2 = int_list
#                     # Define the bounding box and label
#                     # bbox = (x1, y1, x2, y2)
#                     label = names[cla_key]
#                     # Draw the bounding box and label on the frame
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#             # Write the annotated frame to the output video file
#             out.write(frame)


#         else:
#             # Break the loop if the end of the video is reached
#             break

#     # Release the video capture object and the VideoWriter object
#     cap.release()
#     out.release()
#     os.remove(video_path)

#     return output_video_path


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