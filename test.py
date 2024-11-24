from PIL import Image

from ultralytics import YOLO

from utils import utils

utils.process_video_with_yolo("videos/source/1.mp4", target_class=[0, 3])