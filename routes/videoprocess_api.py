from fastapi import APIRouter, HTTPException, UploadFile, File, FastAPI, Header, Response
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import os
import shutil
import multiprocessing
from pathlib import Path
from utils.utils import process_video_with_yolo
import re

import uuid

# Initialize FastAPI app
app = FastAPI()

router = APIRouter()

# Global dictionary to store queues by process ID
process_queues = {}

# Function to wrap the call to process_video_with_yolo
def task_wrapper(video_path, queue, target_class=[0]):
    try:
        output_video_path = process_video_with_yolo(video_path, target_class)
        filename_with_extension = os.path.basename(output_video_path)
        
        
        queue.put({"status": "completed", "result": filename_with_extension})
    except Exception as e:
        queue.put({"status": "error", "result": str(e)})


@router.post("/create-process")
async def create_process(file: UploadFile = File(...)):
    print("File Downloading...")
    random_string = str(uuid.uuid4())
    base_name, extension = os.path.splitext(file.filename)
    new_file_name = f"{base_name}_{random_string}{extension}"
    file_location = f"videos/source/{new_file_name}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print("Processing...")
    # Create a Queue to receive the output video path from the task
    queue = multiprocessing.Queue()

    target_class = [3]
    # Create a Process that will execute the task_wrapper function
    process = multiprocessing.Process(target=task_wrapper, args=(file_location, queue, target_class))

    # Start the process
    process.start()

    # Store the queue in the global dictionary using the process ID
    process_queues[process.pid] = queue

    # Return the process ID (task ID)
    return {"task_id": process.pid}


@router.post("/create-process1")
async def create_process(file: UploadFile = File(...)):
    print("File Downloading...")
    random_string = str(uuid.uuid4())
    base_name, extension = os.path.splitext(file.filename)
    new_file_name = f"{base_name}_{random_string}{extension}"
    file_location = f"videos/source/{new_file_name}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print("Processing...")
    # Create a Queue to receive the output video path from the task
    queue = multiprocessing.Queue()
    
    target_class = [0]

    # Create a Process that will execute the task_wrapper function
    process = multiprocessing.Process(target=task_wrapper, args=(file_location, queue, target_class))

    # Start the process
    process.start()

    # Store the queue in the global dictionary using the process ID
    process_queues[process.pid] = queue

    # Return the process ID (task ID)
    return {"task_id": process.pid}


@router.get("/get-result/{task_id}")
async def get_result(task_id: int):
    # Convert task_id to int, as it will be received as a string from the path
    task_id = int(task_id)

    # Check if the task_id is in the global dictionary
    if task_id in process_queues:
        queue = process_queues[task_id]
        try:
            # Non-blocking get from the queue with a timeout
            result = queue.get_nowait()
            return result
        except multiprocessing.queues.Empty:
            # If the queue is empty, the task is still running
            return {"status": "running", "result": task_id}
    else:
        raise HTTPException(status_code=404, detail="Task ID not found")



@router.get("/download-video/{filename}")
async def download_video(filename: str):
    video_path = Path(f"videos/result/{filename}")

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    # return FileResponse(path=video_path, filename=filename, media_type='application/octet-stream',  headers={"Content-Disposition": "attachment; filename=video.mp4"})
    return FileResponse(path=video_path,  headers={"Content-Disposition": f"attachment; filename={filename}"})


# @app.get("/download_video")
# async def download_video():
#     # Path to the generated video file
#     output_dir = './output'
#     output_path = os.path.join(output_dir, 'video.mp4')
#     # Ensure the output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     # Ensure the file exists before returning it
#     if not os.path.exists(output_path):
#         raise HTTPException(status_code=404, detail="Video file not found.")
#     # torch.cuda.empty_cache()
#     # Return the file with Content-Disposition header set to 'attachment' to force download
#     return FileResponse(output_path, headers={"Content-Disposition": "attachment; filename=video.mp4"})


from fastapi.responses import Response

@router.get("/stream/{filename}")
def stream_video(filename: str, range: str = None):
    video_path = Path(f"videos/result/{filename}")
    if not video_path.exists():
        return {"error": "Video not found"}

    file_size = video_path.stat().st_size
    start = 0
    end = file_size - 1

    if range:
        match = re.match(r"bytes=(\d+)-(\d*)", range)
        if match:
            start = int(match.group(1))
            if match.group(2):
                end = int(match.group(2))

    chunk_size = (end - start) + 1
    with open(video_path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size)

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
    }

    return Response(data, status_code=206, headers=headers, media_type="video/mp4")