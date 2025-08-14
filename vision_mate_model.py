import numpy as np
import torch
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import pyttsx3
import threading
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

class VisionMateModel:
    """
    A core model for visual analysis, including object detection,
    color recognition, and lighting analysis.
    """
    def __init__(self):
        self.processor = None
        self.model = None
        self.model_loaded = False
    
    def load_model(self):
        """Loads and caches the DETR object detection model."""
        try:
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            self.model_loaded = True
            return True, "Model loaded successfully."
        except Exception as e:
            self.model_loaded = False
            return False, f"Failed to load model: {e}"

    def speak_text(self, text, rate=150, volume=1.0):
        """Converts text to speech in a separate thread to prevent UI blocking."""
        def _speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", rate)
                engine.setProperty("volume", volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"Text-to-speech not available: {e}")
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

    def get_dominant_colors(self, image_region, n_colors=3):
        """Extracts dominant colors from an image region using k-means clustering."""
        try:
            img_array = np.array(image_region)
            data = img_array.reshape((-1, 3))
            
            brightness = np.mean(data, axis=1)
            data = data[brightness > 30]
            
            if len(data) == 0:
                return [(128, 128, 128)]
            
            k = min(n_colors, len(data))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(-counts)
            
            return [tuple(colors[i]) for i in sorted_indices]
        except Exception:
            return [(128, 128, 128)]
    
    def rgb_to_color_name(self, rgb):
        """Converts an RGB tuple to a human-readable color name using HSV space."""
        r, g, b = rgb

        # Convert RGB to HSV (OpenCV expects BGR, so reverse the tuple)
        hsv = cv2.cvtColor(np.uint8([[ [b, g, r] ]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv

        # HSV color ranges (Hue: 0-179 in OpenCV)
        hsv_colors = {
            'red': [(0, 70, 50), (10, 255, 255), (170, 70, 50), (180, 255, 255)],
            'orange': [(11, 70, 50), (25, 255, 255)],
            'yellow': [(26, 70, 50), (35, 255, 255)],
            'green': [(36, 70, 50), (85, 255, 255)],
            'cyan': [(86, 70, 50), (95, 255, 255)],
            'blue': [(96, 70, 50), (125, 255, 255)],
            'purple': [(126, 70, 50), (155, 255, 255)],
            'pink': [(156, 70, 50), (169, 255, 255)],
            'white': [(0, 0, 200), (180, 40, 255)],
            'gray': [(0, 0, 50), (180, 40, 199)],
            'black': [(0, 0, 0), (180, 255, 49)],
            'brown': [(10, 100, 20), (25, 255, 200)]
        }

        # Check exact HSV range match
        for color_name, ranges in hsv_colors.items():
            if len(ranges) == 4:  # for colors like red that wrap hue space
                lower1, upper1, lower2, upper2 = ranges
                if (lower1[0] <= h <= upper1[0] and lower1[1] <= s <= upper1[1] and lower1[2] <= v <= upper1[2]) or \
                   (lower2[0] <= h <= upper2[0] and lower2[1] <= s <= upper2[1] and lower2[2] <= v <= upper2[2]):
                    return color_name
            else:
                lower, upper = ranges
                if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                    return color_name

        # If no match, return closest by hue distance
        min_distance = float('inf')
        closest_color = 'unknown'
        for color_name, ranges in hsv_colors.items():
            if len(ranges) == 4:
                lower1, upper1, lower2, upper2 = ranges
                center_h = ((lower1[0] + upper1[0]) // 2 + (lower2[0] + upper2[0]) // 2) / 2
            else:
                lower, upper = ranges
                center_h = (lower[0] + upper[0]) // 2
            distance = abs(h - center_h)
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name

        return closest_color
    
    def analyze_lighting_conditions(self, image):
        """Analyzes lighting conditions of an image based on brightness and contrast."""
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            if mean_brightness < 40: primary_condition = "very dark"
            elif mean_brightness < 80: primary_condition = "dark"
            elif mean_brightness < 120: primary_condition = "dimly lit"
            elif mean_brightness < 160: primary_condition = "moderately lit"
            elif mean_brightness < 200: primary_condition = "well lit"
            else: primary_condition = "very bright"
            
            descriptors = []
            if std_brightness < 25: descriptors.append("low contrast")
            elif std_brightness > 70: descriptors.append("high contrast")
            if np.sum(gray > 240) / gray.size > 0.1: descriptors.append("some overexposure")
            if np.sum(gray < 20) / gray.size > 0.1: descriptors.append("some dark areas")
            
            full_description = primary_condition
            if descriptors:
                full_description += " with " + ", ".join(descriptors)
            
            return full_description
            
        except Exception:
            return "moderate lighting"
    
    def detect_objects_safe(self, image, confidence_threshold=0.7):
        """Performs object detection on an image using the loaded DETR model."""
        try:
            if not self.model_loaded:
                return None, "Model not loaded. Please load the model first."
            
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )[0]
            
            return results, None
            
        except Exception as e:
            return None, f"Object detection failed: {e}"
