from fastapi import APIRouter, HTTPException, UploadFile, File, FastAPI, Header, Response, Form
from typing import List
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
import os
import shutil
# import multiprocessing
from pathlib import Path
# from utils.utils import process_video_with_yolo, process_video_with_yolo_and_pose
import re

from dotenv import load_dotenv

from utils import utils

import uuid

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

router = APIRouter()

# Global dictionary to store queues by process ID
process_queues = {}

# # Function to wrap the call to process_video_with_yolo
# def task_wrapper(video_path, queue, target_class=[0]):
#     try:
#         output_video_path = process_video_with_yolo(video_path, target_class)
#         filename_with_extension = os.path.basename(output_video_path)
        
        
#         queue.put({"status": "completed", "result": filename_with_extension})
#     except Exception as e:
#         queue.put({"status": "error", "result": str(e)})
# # Function to wrap the call to process_video_with_yolo
# def task_wrapper2(video_path, queue, target_class=[3]):
#     try:
#         output_video_path = process_video_with_yolo_and_pose(video_path, target_class)
#         filename_with_extension = os.path.basename(output_video_path)
        
        
#         queue.put({"status": "completed", "result": filename_with_extension})
#     except Exception as e:
#         queue.put({"status": "error", "result": str(e)})


# @router.post("/create-process-player")
# async def create_process_player(file: UploadFile = File(...)):
#     print("File Downloading...")
#     random_string = str(uuid.uuid4())
#     base_name, extension = os.path.splitext(file.filename)
#     new_file_name = f"{base_name}_{random_string}{extension}"
#     file_location = f"videos/source/{new_file_name}"
    
#     with open(file_location, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     print("Processing...")
#     # Create a Queue to receive the output video path from the task
#     queue = multiprocessing.Queue()

#     target_class = [3]
#     # Create a Process that will execute the task_wrapper function
#     process = multiprocessing.Process(target=task_wrapper, args=(file_location, queue, target_class))

#     # Start the process
#     process.start()

#     # Store the queue in the global dictionary using the process ID
#     process_queues[process.pid] = queue

#     # Return the process ID (task ID)
#     return {"task_id": process.pid}


# @router.post("/create-process-ball")
# async def create_process_ball(file: UploadFile = File(...)):
#     print("File Downloading...")
#     random_string = str(uuid.uuid4())
#     base_name, extension = os.path.splitext(file.filename)
#     new_file_name = f"{base_name}_{random_string}{extension}"
#     file_location = f"videos/source/{new_file_name}"
    
#     with open(file_location, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     print("Processing...")
#     # Create a Queue to receive the output video path from the task
#     queue = multiprocessing.Queue()
    
#     target_class = [0]

#     # Create a Process that will execute the task_wrapper function
#     process = multiprocessing.Process(target=task_wrapper, args=(file_location, queue, target_class))

#     # Start the process
#     process.start()

#     # Store the queue in the global dictionary using the process ID
#     process_queues[process.pid] = queue

#     # Return the process ID (task ID)
#     return {"task_id": process.pid}

# @router.post("/create-process-pose")
# async def create_process_pose(file: UploadFile = File(...)):
#     print("File Downloading...")
#     random_string = str(uuid.uuid4())
#     base_name, extension = os.path.splitext(file.filename)
#     new_file_name = f"{base_name}_{random_string}{extension}"
#     file_location = f"videos/source/{new_file_name}"
    
#     with open(file_location, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     print("Processing...")
#     # Create a Queue to receive the output video path from the task
#     queue = multiprocessing.Queue()
    
#     target_class = [3]

#     # Create a Process that will execute the task_wrapper function
#     process = multiprocessing.Process(target=task_wrapper2, args=(file_location, queue, target_class))

#     # Start the process
#     process.start()

#     # Store the queue in the global dictionary using the process ID
#     process_queues[process.pid] = queue

#     # Return the process ID (task ID)
#     return {"task_id": process.pid}


# @router.get("/get-result/{task_id}")
# async def get_result(task_id: int):
#     # Convert task_id to int, as it will be received as a string from the path
#     task_id = int(task_id)

#     # Check if the task_id is in the global dictionary
#     if task_id in process_queues:
#         queue = process_queues[task_id]
#         try:
#             # Non-blocking get from the queue with a timeout
#             result = queue.get_nowait()
#             return result
#         except multiprocessing.queues.Empty:
#             # If the queue is empty, the task is still running
#             return {"status": "running", "result": task_id}
#     else:
#         raise HTTPException(status_code=404, detail="Task ID not found")



# @router.get("/download-video/{filename}")
# async def download_video(filename: str):
#     video_path = Path(f"videos/result/{filename}")

#     if not video_path.is_file():
#         raise HTTPException(status_code=404, detail="Video not found")

#     # return FileResponse(path=video_path, filename=filename, media_type='application/octet-stream',  headers={"Content-Disposition": "attachment; filename=video.mp4"})
#     return FileResponse(path=video_path,  headers={"Content-Disposition": f"attachment; filename={filename}"})



# @router.get("/stream/{filename}")
# def stream_video(filename: str, range: str = None):
#     video_path = Path(f"videos/result/{filename}")
#     if not video_path.exists():
#         return {"error": "Video not found"}

#     file_size = video_path.stat().st_size
#     start = 0
#     end = file_size - 1

#     if range:
#         match = re.match(r"bytes=(\d+)-(\d*)", range)
#         if match:
#             start = int(match.group(1))
#             if match.group(2):
#                 end = int(match.group(2))

#     chunk_size = (end - start) + 1
#     with open(video_path, "rb") as f:
#         f.seek(start)
#         data = f.read(chunk_size)

#     headers = {
#         "Content-Range": f"bytes {start}-{end}/{file_size}",
#         "Accept-Ranges": "bytes",
#         "Content-Length": str(chunk_size),
#     }

#     return Response(data, status_code=206, headers=headers, media_type="video/mp4")


@router.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    FastAPI endpoint to upload multiple images and send them to GPT-4.
    """
    image_paths = []
    for file in files:
        file_path = f"./videos/source/temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_paths.append(file_path)
    
    api_key = os.getenv("API_KEY")
    prompt = "I attached the screenshots of my weightlifting forms. At the first please check the images very carefully and give me advice what is wrong in forms. Give me maximumn 6 sentences for each image. Give me raw text without any symbols. If the images are not related with weightlifting, please ask to upload weightlifting related images."
    
    response = utils.send_image_to_gpt4(api_key, image_paths, prompt)
    return {"response": response}



@router.post("/upload2")
async def upload_images2(files: List[UploadFile] = File(...)):
    """
    FastAPI endpoint to upload multiple images and send them to GPT-4.
    """
    image_paths = []
    for file in files:
        file_path = f"./videos/source/temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_paths.append(file_path)
    
    api_key = os.getenv("API_KEY")
    prompt = "I attached the screenshots of my basketball forms. At the first please check the images very carefully and give me advice what is wrong in forms. Give me maximumn 6 sentences for each image. Give me raw text without any symbols. If the images are not related with basketball, please ask to upload basketball related images."
    
    response = utils.send_image_to_gpt4(api_key, image_paths, prompt)
    return {"response": response}

