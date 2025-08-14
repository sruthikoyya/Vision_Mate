import streamlit as st
import time
import os
from vision_mate_model import VisionMateModel
from camera_utils import capture_image
# Set page config for a formal look
st.set_page_config(
    page_title="Vision Mate",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for persistent variables
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_description' not in st.session_state:
    st.session_state.last_description = ""
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'model_instance' not in st.session_state:
    st.session_state.model_instance = VisionMateModel()

def generate_rich_description(image, results, settings):
    """
    Generates a comprehensive scene description by combining multiple
    analysis results.
    """
    descriptions = []

    # Lighting analysis
    if settings.get('include_lighting', True):
        lighting = st.session_state.model_instance.analyze_lighting_conditions(image)
        descriptions.append(f"The scene appears {lighting}.")
    
    # Object detection results
    if len(results["scores"]) == 0:
        descriptions.append("No clear objects are detected in the current view.")
        return " ".join(descriptions)

    detected_objects = []
    image_width, image_height = image.size
    center_x, center_y = image_width / 2, image_height / 2
    pixels_per_foot = image_width / settings.get('physical_width', 15)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box_coords = [round(i, 2) for i in box.tolist()]
        object_name = st.session_state.model_instance.model.config.id2label[label.item()]
        
        obj_center_x, obj_center_y = (box_coords[0] + box_coords[2]) / 2, (box_coords[1] + box_coords[3]) / 2
        obj_width, obj_height = box_coords[2] - box_coords[0], box_coords[3] - box_coords[1]

        # Position and size logic remains the same
        horizontal_pos = "center"
        if obj_center_x < center_x - image_width * 0.15: horizontal_pos = "left"
        elif obj_center_x > center_x + image_width * 0.15: horizontal_pos = "right"
        
        vertical_pos = "middle"
        if obj_center_y < center_y - image_height * 0.15: vertical_pos = "upper"
        elif obj_center_y > center_y + image_height * 0.15: vertical_pos = "lower"

        distance_pixels = abs(obj_center_x - center_x)
        distance_feet = round(distance_pixels / pixels_per_foot, 1)

        obj_size = "medium-sized"
        area_ratio = (obj_width * obj_height) / (image_width * image_height)
        if area_ratio > 0.3: obj_size = "large"
        elif area_ratio < 0.05: obj_size = "small"

        color_desc = ""
        if settings.get('include_colors', True):
            try:
                x1, y1, x2, y2 = map(int, box_coords)
                object_region = image.crop((x1, y1, x2, y2))
                dominant_colors = st.session_state.model_instance.get_dominant_colors(object_region, 2)
                if len(dominant_colors) > 0:
                    primary_color = st.session_state.model_instance.rgb_to_color_name(dominant_colors[0])
                    color_desc = f"{primary_color} "
            except Exception as e:
                print(f"Color analysis failed: {e}")
                pass
        
        detected_objects.append({
            'name': object_name, 'color': color_desc, 'size': obj_size,
            'horizontal_pos': horizontal_pos, 'vertical_pos': vertical_pos,
            'distance': distance_feet
        })

    object_groups = {}
    for obj in detected_objects:
        key = f"{obj['color']}{obj['name']}"
        if key not in object_groups: object_groups[key] = []
        object_groups[key].append(obj)
    
    for group_name, objects in object_groups.items():
        if len(objects) == 1:
            obj = objects[0]
            position = f"on the {obj['horizontal_pos']}"
            if obj['vertical_pos'] != "middle": position = f"in the {obj['vertical_pos']} {obj['horizontal_pos']}"
            distance_info = f" about {obj['distance']} feet away" if obj['distance'] > 0.5 else ""
            descriptions.append(f"I can see a {obj['size']} {obj['color']}{obj['name']} {position}{distance_info}.")
        else:
            positions = [f"{obj['horizontal_pos']}" for obj in objects]
            position_counts = {}
            for pos in positions: position_counts[pos] = position_counts.get(pos, 0) + 1
            
            position_desc = ", ".join([f"{count} on the {pos}" for pos, count in position_counts.items()])
            descriptions.append(f"There are {len(objects)} {objects[0]['color']}{objects[0]['name']}s with {position_desc}.")
    
    return " ".join(descriptions)

def run_analysis():
    """The main function to orchestrate the entire analysis pipeline."""
    st.session_state.processing = True
    
    try:
        settings = {
            'confidence': st.session_state.get('confidence', 0.7),
            'include_colors': st.session_state.get('include_colors', True),
            'include_lighting': st.session_state.get('include_lighting', True),
            'physical_width': st.session_state.get('physical_width', 15),
            'speech_rate': st.session_state.get('speech_rate', 150),
            'speech_volume': st.session_state.get('speech_volume', 1.0),
        }
        
        if not st.session_state.model_instance.model_loaded:
            with st.spinner("Loading AI model..."):
                success, message = st.session_state.model_instance.load_model()
                if not success:
                    st.warning(f"{message}")
                    st.session_state.processing = False
                    return
        
        with st.spinner("📷 Capturing image..."):
            image, error = capture_image()
            if image is None:
                st.warning(f"{error}")
                st.session_state.processing = False
                return
        
        st.session_state.captured_image = image
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Captured Scene", use_column_width=True)

        with st.spinner("🔍 Analyzing objects..."):
            results, error = st.session_state.model_instance.detect_objects_safe(image, settings['confidence'])
            if results is None:
                st.warning(f"{error}")
                st.session_state.processing = False
                return
        
        with st.spinner("Generating detailed description..."):
            description = generate_rich_description(image, results, settings)
            st.session_state.last_description = description
        
        with col2:
            st.success("Analysis Complete!")
            st.write(f"**Objects Detected:** {len(results['scores'])}")
            if settings['include_lighting']:
                st.info(f"**Lighting:** {st.session_state.model_instance.analyze_lighting_conditions(image)}")

        st.subheader("Analysis Report")
        st.write(description)
        
        with st.spinner("Converting to speech..."):
            st.session_state.model_instance.speak_text(description, settings['speech_rate'], settings['speech_volume'])
        
        st.success("Audio description is playing.")
        
    except Exception as e:
        st.warning(f"An unexpected error occurred: {str(e)}")
    
    finally:
        st.session_state.processing = False

def main():
    """Main application layout for the desktop interface."""
    st.markdown(
    """
    <h1 style='text-align: center; color: black;'>Vision Mate</h1>
    <h3 style='text-align: center; color: black;'>Accessibility Assistant for Visually Impaired</h3>
    """,
    unsafe_allow_html=True
)


    st.markdown("---")
    
    # Sidebar for a formal settings panel
    with st.sidebar:
        st.header("System Settings")
        
        with st.expander("Detection Parameters", expanded=True):
            st.session_state.confidence = st.slider("Confidence Threshold", 0.3, 0.95, 0.7, 0.05, help="Controls the sensitivity of object detection.")
            st.session_state.physical_width = st.number_input("Scene Width (feet)", 5, 30, 15, 1, help="Approximate width of the scene in feet for distance estimation.")
        
        with st.expander("Feature Toggles", expanded=True):
            st.session_state.include_colors = st.checkbox("Enable Color Detection", value=True)
            st.session_state.include_lighting = st.checkbox("Enable Lighting Analysis", value=True)
        
        with st.expander("Speech Settings", expanded=False):
            st.session_state.speech_rate = st.slider("Speech Rate (WPM)", 80, 250, 150, 10)
            st.session_state.speech_volume = st.slider("Speech Volume", 0.3, 1.0, 1.0, 0.1)

        st.markdown("---")
        
        st.subheader("System Actions")
        if st.button("Test Camera"):
            with st.spinner("Testing camera..."):
                test_image, error = capture_image()
                if test_image:
                    st.success("Camera connection is working.")
                    st.image(test_image, caption="Camera Test Image", use_column_width=True)
                else:
                    st.warning(f"Camera error: {error}")
        
        if st.session_state.last_description and st.button("Save Last Description"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"vision_mate_report_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write(f"Vision Mate Pro Analysis Report - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n")
                f.write(st.session_state.last_description)
            st.success(f"Report saved as **{filename}**")

    # Main content area
    st.markdown("### Application Overview")
    st.markdown(
        """
        This application delivers an in-depth visual analysis by harnessing state-of-the-art computer vision techniques.It is built to provide users with meaningful insights into their surroundings through precise object detection,accurate color recognition, and environmental condition assessment. By combining advanced deep learning models with an intuitive interface, the system aims to enhance situational awareness,support informed decision-making, and make visual information more accessible and actionable for a wide range of real-world scenarios.
        """
        )
    st.markdown("---")
    
    st.subheader("Live Scene Analysis")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.processing:
            if st.button("▶Start Analysis", type="primary", use_container_width=True):
                run_analysis()
        else:
            st.button("Processing...", disabled=True, use_container_width=True)
    
    if st.session_state.last_description:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Last Analysis Report")
            st.info(st.session_state.last_description)
        
        with col2:
            st.subheader("Report Actions")
            if st.button("Repeat Audio Report", use_container_width=True):
                st.session_state.model_instance.speak_text(
                    st.session_state.last_description, 
                    st.session_state.speech_rate, 
                    st.session_state.speech_volume
                )
            
            if st.session_state.captured_image and st.button("Show Captured Image", use_container_width=True):
                st.image(st.session_state.captured_image, caption="Captured Scene", use_column_width=True)
    
    # Detailed usage and requirements
    with st.expander("Detailed Usage and Installation Guide", expanded=False):
        st.markdown("""
        ### Instructions for Use
        1. **Configure Settings:** Adjust the parameters in the sidebar to fine-tune the analysis, such as confidence threshold and speech rate.
        2. **Initiate Analysis:** Click the "Start Analysis" button to capture an image from your webcam and begin the processing pipeline.
        3. **Review Results:** The application will display a textual report and play an audio summary of the scene.
        
        ### Installation Requirements
        - **Python 3.8+**
        - **Webcam**
        - **CPU:** A modern multi-core processor is recommended for smooth performance.
        
        ### Python Package Dependencies
        Install all necessary packages using pip:
        ```bash
        pip install streamlit opencv-python torch transformers pillow pyttsx3 scikit-learn
        ```
        """)

if __name__ == "__main__":
    main()