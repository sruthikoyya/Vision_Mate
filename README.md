# Vision Mate
AI-powered accessibility assistant

## Purpose
Vision Mate was built with one main goal — to help visually impaired individuals understand their surroundings better.  
It doesn’t just detect objects; it describes the whole scene in a human-like way — including lighting, colors, positions, and even how far things are.

## How It’s Useful
Imagine standing in a room and wanting to know:
- Is it bright or dim?
- What objects are around you?
- Where exactly are they?
- How far are they from you?

Vision Mate answers all of that in clear speech.  
It’s like having a helpful friend describing what they see in real time.

## What I Used
- **Python** – The brain of the project
- **Streamlit** – Easy-to-use interface
- **DETR (facebook/detr-resnet-50)** – Transformer-based object detection
- **OpenCV** – Image processing and lighting analysis
- **Scikit-learn (KMeans)** – Finding dominant colors in objects
- **Pyttsx3** – Offline text-to-speech
- **PIL / NumPy** – Image handling and number processing

## How It Works
1. Captures a scene through the webcam
2. Detects objects and identifies their colors, sizes, and positions
3. Analyzes lighting conditions
4. Estimates how far each object is
5. Speaks a natural description of the scene aloud
