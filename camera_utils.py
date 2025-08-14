# camera_utils.py

import cv2
import time
from PIL import Image
import numpy as np

def capture_image():
    """
    Captures an image from the webcam with multiple attempts and returns
    a PIL Image object.
    """
    cap = None
    try:
        # Loop through potential camera indices
        for camera_index in range(4):
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    break  # Found a working camera
                else:
                    cap.release()
                    cap = None
            else:
                if cap:
                    cap.release()
                cap = None
        
        if not cap or not cap.isOpened():
            return None, "Could not access any camera. Please check camera permissions."
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        time.sleep(0.5)
        
        best_frame = None
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                best_frame = frame
            time.sleep(0.1)
        
        cap.release()
        
        if best_frame is None:
            return None, "Failed to capture image from camera."
        
        rgb_image = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image, None
        
    except Exception as e:
        if cap:
            cap.release()
        return None, f"Camera error: {str(e)}"