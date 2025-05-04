# to run this script: streamlit run app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import time
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import supervision as sv
import hashlib
import warnings
import math

# Suppress torch warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

# Constants
COIN_CLASS_ID = 11  # 10sen coin
COIN_DIAMETER_MM = 18.80  # 10sen coin diameter in mm
CLASS_NAMES = {
    0: 'long lag screw',
    1: 'wood screw',
    2: 'lag wood screw',
    3: 'short wood screw',
    4: 'shiny screw',
    5: 'black oxide screw',
    6: 'nut',
    7: 'bolt',
    8: 'large nut',
    9: 'machine screw',
    10: 'short machine screw',
    11: '10sen Coin'
}
CATEGORY_COLORS = {
    'long lag screw': (255, 0, 0),
    'wood screw': (0, 255, 0),
    'lag wood screw': (0, 0, 255),
    'short wood screw': (255, 255, 0),
    'shiny screw': (255, 0, 255),
    'black oxide screw': (0, 255, 255),
    'nut': (128, 0, 128),
    'bolt': (255, 165, 0),
    'large nut': (128, 128, 0),
    'machine screw': (0, 128, 128),
    'short machine screw': (128, 0, 0),
    '10sen Coin': (192, 192, 192)
}
LABEL_FONT_SIZE = 20
BORDER_WIDTH = 3

# Load YOLO model
model = YOLO("yolo11-obb12classes.pt")

# Initialize session state for tracking
if 'tracked_objects' not in st.session_state:
    st.session_state.tracked_objects = {}  # Changed from set to dict to store class info
if 'binary_threshold' not in st.session_state:
    st.session_state.binary_threshold = 150
if 'blur_kernel' not in st.session_state:
    st.session_state.blur_kernel = 5
if 'erode_iterations' not in st.session_state:
    st.session_state.erode_iterations = 1
if 'dilate_iterations' not in st.session_state:
    st.session_state.dilate_iterations = 1

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    # Image Processing Controls
    st.subheader("Image Processing")
    binary_threshold = st.slider("Binary Threshold", 0, 255, 150)
    st.session_state.binary_threshold = binary_threshold
    
    blur_kernel = st.slider("Blur Kernel Size", 1, 15, 5, step=2)
    st.session_state.blur_kernel = blur_kernel
    
    erode_iterations = st.slider("Erode Iterations", 0, 5, 1)
    st.session_state.erode_iterations = erode_iterations
    
    dilate_iterations = st.slider("Dilate Iterations", 0, 5, 1)
    st.session_state.dilate_iterations = dilate_iterations
    
    # Detection Controls
    st.subheader("Detection Settings")
    input_method = st.radio(
        "Input Source",
        ("Webcam (Live Camera)", "Upload Image", "Upload Video"),
        index=0
    )
    IOU_THRESHOLD = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.7, step=0.05)
    CONFIDENCE_THRESHOLD = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)
    
    if input_method == "Webcam (Live Camera)":
        WEBCAM_WIDTH = st.slider("Webcam Width", 320, 1920, 640, step=160)
        WEBCAM_HEIGHT = st.slider("Webcam Height", 240, 1080, 480, step=120)
        SHOW_FPS = st.checkbox("Show FPS", value=True)
    
    SHOW_DETECTIONS = st.checkbox("Show Detections", value=True)
    SHOW_SUMMARY = st.checkbox("Show Summary", value=True)
    SHOW_ORIENTATION = st.checkbox("Show Orientation", value=True)
    RESET_COUNTER = st.button("Reset Detection Counter")

# Reset counter if button pressed
if RESET_COUNTER:
    st.session_state.tracked_objects = {}

def get_text_size(draw, text, font):
    if hasattr(draw, 'textbbox'):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        return draw.textsize(text, font=font)

def xywhr_to_corners(xywhr):
    """Convert xywhr format to four corner points of the rotated rectangle"""
    x, y, w, h, r = xywhr
    cos_r = math.cos(r)
    sin_r = math.sin(r)
    
    # Calculate half width and height
    half_w = w / 2
    half_h = h / 2
    
    # Calculate the four corners relative to center
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])
    
    # Rotate the corners
    rotation_matrix = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # Translate corners to absolute position
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    
    return rotated_corners.astype(int)

def non_max_suppression(detections, iou_threshold):
    """Improved NMS for OBB that keeps only the highest confidence detection for each physical object"""
    if len(detections) == 0:
        return []

    # Convert detections to numpy arrays for easier processing
    boxes = []
    scores = []
    classes = []
    xywhr = []

    for det in detections:
        if len(det.xyxy) > 0:
            boxes.append(det.xyxy[0].cpu().numpy())
            scores.append(det.conf[0].cpu().numpy())
            classes.append(det.cls[0].cpu().numpy())
            xywhr.append(det.xywhr[0].cpu().numpy())

    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    xywhr = np.array(xywhr)

    # Sort by confidence score
    indices = np.argsort(scores)[::-1]
    keep_indices = []

    while len(indices) > 0:
        current = indices[0]
        keep_indices.append(current)
        rest = indices[1:]

        if len(rest) == 0:
            break

        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        rest_boxes = boxes[rest]
        
        # Calculate intersection coordinates
        xA = np.maximum(current_box[0], rest_boxes[:, 0])
        yA = np.maximum(current_box[1], rest_boxes[:, 1])
        xB = np.minimum(current_box[2], rest_boxes[:, 2])
        yB = np.minimum(current_box[3], rest_boxes[:, 3])
        
        # Calculate areas
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        currentArea = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        restAreas = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        
        # Calculate IoU
        ious = interArea / (currentArea + restAreas - interArea + 1e-6)
        
        # Keep boxes that have IoU less than threshold (regardless of class)
        to_keep = (ious <= iou_threshold)
        indices = rest[to_keep]

    return [detections[i] for i in keep_indices]

def generate_object_id(bbox, class_name):
    """Generate a unique ID based on bbox and class"""
    x1, y1, x2, y2 = map(int, bbox)
    return hashlib.md5(f"{x1}{y1}{x2}{y2}{class_name}".encode()).hexdigest()

def xywhr_to_contour(xywhr, image_shape):
    """Convert xywhr format to contour points for cv2.findContours"""
    x, y, w, h, r = xywhr
    
    # Calculate half width and height
    half_w = w / 2
    half_h = h / 2
    
    # Create the four corners relative to center
    corners = np.array([
        [-half_w, -half_h],  # top-left
        [half_w, -half_h],   # top-right
        [half_w, half_h],    # bottom-right
        [-half_w, half_h]    # bottom-left
    ])
    
    # Create rotation matrix
    cos_r = math.cos(r)
    sin_r = math.sin(r)
    rotation_matrix = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    
    # Rotate corners
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # Translate to image coordinates
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    
    # Ensure corners are within image bounds
    rotated_corners = np.clip(rotated_corners, 0, np.array([image_shape[1], image_shape[0]]))
    
    # Convert to integer coordinates
    corners_int = rotated_corners.astype(np.int32)
    
    # Create a binary mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Draw the rotated rectangle on the mask
    cv2.fillPoly(mask, [corners_int], 255)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        return contour
    return None

def apply_edge_detection(image, method="Canny", **kwargs):
    """Apply different edge detection methods"""
    if method == "Canny":
        return cv2.Canny(image, kwargs.get('low', 50), kwargs.get('high', 150))
    elif method == "Sobel":
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs.get('ksize', 3))
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kwargs.get('ksize', 3))
        return cv2.magnitude(dx, dy)
    elif method == "Laplacian":
        return cv2.Laplacian(image, cv2.CV_64F, ksize=kwargs.get('ksize', 3))
    elif method == "Scharr":
        dx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        dy = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        return cv2.magnitude(dx, dy)
    elif method == "Prewitt":
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        dx = cv2.filter2D(image, cv2.CV_64F, kernelx)
        dy = cv2.filter2D(image, cv2.CV_64F, kernely)
        return cv2.magnitude(dx, dy)
    return image

def process_frame(frame, px_to_mm_ratio=None):
    """Process a single frame and return annotated image and detection data"""
    results = model(frame, conf=CONFIDENCE_THRESHOLD)
    if not results:
        return frame, [], px_to_mm_ratio
    
    result = results[0]
    filtered_detections = non_max_suppression(result.obb, IOU_THRESHOLD)
    
    # Convert frame to RGB for drawing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    detected_objects = []
    current_px_to_mm_ratio = px_to_mm_ratio
    
    # First pass: Find and draw contours for all detections
    for detection in filtered_detections:
        if len(detection.cls) > 0 and len(detection.xywhr) > 0 and len(detection.xyxy) > 0:
            class_id = int(detection.cls[0])
            confidence = detection.conf[0]
            xywhr = detection.xywhr[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            class_name = CLASS_NAMES.get(class_id, f"Class {int(class_id)}")
            color = CATEGORY_COLORS.get(class_name, (0, 255, 0))

            # Get OBB corners
            corners = xywhr_to_corners(xywhr)
            
            # Create a slightly larger mask for contour detection
            # while keeping the original OBB for visualization
            expand_factor = 1.1  # Expand the mask by 10%
            center = np.mean(corners, axis=0)
            expanded_corners = corners.copy()
            expanded_corners = center + (expanded_corners - center) * expand_factor
            
            # Create mask with expanded corners
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [expanded_corners.astype(np.int32)], 255)
            
            # Extract the OBB region with expanded mask
            roi = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)
            roi = roi[y1:y2, x1:x2]  # Crop to the axis-aligned bbox for processing
            
            if roi.size > 0:  # Check if ROI is valid
                # Convert ROI to grayscale
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                
                # Apply Gaussian blur to smooth edges
                blurred = cv2.GaussianBlur(gray, (st.session_state.blur_kernel, st.session_state.blur_kernel), 0)
                
                # Apply binary thresholding
                _, thresh = cv2.threshold(blurred, st.session_state.binary_threshold, 255, cv2.THRESH_BINARY)
                
                # Create kernel for morphological operations
                kernel = np.ones((3,3), np.uint8)
                
                # Apply erosion to remove small noise
                if st.session_state.erode_iterations > 0:
                    thresh = cv2.erode(thresh, kernel, iterations=st.session_state.erode_iterations)
                
                # Apply dilation to connect nearby components
                if st.session_state.dilate_iterations > 0:
                    thresh = cv2.dilate(thresh, kernel, iterations=st.session_state.dilate_iterations)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if contours:
                    # Get all contours that are significant (not too small)
                    significant_contours = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 100:  # Minimum area threshold
                            significant_contours.append(contour)
                    
                    if significant_contours:
                        # Sort contours by area and get the largest one
                        largest_contour = max(significant_contours, key=cv2.contourArea)
                        
                        # Adjust contour coordinates to global image coordinates
                        final_contour = largest_contour + np.array([x1, y1])
                        
                        # Draw contour
                        cv2.drawContours(frame_rgb, [final_contour], -1, color, 3, lineType=cv2.LINE_AA)

    # Second pass: Draw OBBs, orientation indicators, and labels
    for detection in filtered_detections:
        if len(detection.cls) > 0 and len(detection.xywhr) > 0 and len(detection.xyxy) > 0:
            class_id = int(detection.cls[0])
            confidence = detection.conf[0]
            xywhr = detection.xywhr[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            class_name = CLASS_NAMES.get(class_id, f"Class {int(class_id)}")
            color = CATEGORY_COLORS.get(class_name, (0, 255, 0))

            # Generate unique ID for the object
            obj_id = generate_object_id((x1, y1, x2, y2), class_name)
            
            # Only count if not tracked before
            if obj_id not in st.session_state.tracked_objects:
                detected_objects.append({
                    "class_name": class_name,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(confidence),
                    "orientation": float(xywhr[4])
                })
                st.session_state.tracked_objects[obj_id] = class_name

            # Get OBB corners
            corners = xywhr_to_corners(xywhr)
            
            # Draw OBB
            cv2.polylines(frame_rgb, [corners.astype(np.int32)], True, (255, 0, 0), 2)
            
            # Draw orientation indicator if enabled
            if SHOW_ORIENTATION:
                center = (int(xywhr[0]), int(xywhr[1]))
                endpoint = (int(center[0] + 20 * math.cos(xywhr[4])), 
                          int(center[1] + 20 * math.sin(xywhr[4])))
                cv2.line(frame_rgb, center, endpoint, (255, 255, 255), 2)
            
            # Prepare label text
            label_text = f"{class_name}"
            if class_id == COIN_CLASS_ID and current_px_to_mm_ratio:
                diameter_px = (xywhr[2] + xywhr[3]) / 2
                diameter_mm = diameter_px * current_px_to_mm_ratio
                label_text += f", Dia: {diameter_mm:.2f}mm"
            elif class_id != COIN_CLASS_ID and current_px_to_mm_ratio:
                length_px = max(xywhr[2], xywhr[3])
                length_mm = length_px * current_px_to_mm_ratio
                label_text += f", Length: {length_mm:.2f}mm"
            elif class_id != COIN_CLASS_ID:
                label_text += ", Length: N/A (No Coin)"
            elif class_id == COIN_CLASS_ID:
                label_text += ", Dia: N/A (No Ratio)"

            # Draw label background and text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            cv2.rectangle(frame_rgb, 
                        (x1, y1 - text_height - 10),
                        (x1 + text_width + 10, y1),
                        color, -1)
            
            cv2.putText(frame_rgb, label_text,
                      (x1 + 5, y1 - 5),
                      font, font_scale,
                      (255, 255, 255), thickness)

    return frame_rgb, detected_objects, current_px_to_mm_ratio

class VideoCallback:
    def __init__(self):
        self.px_to_mm_ratio = None
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_frame(self, frame):
        # Convert from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        processed_frame, detected_objects, self.px_to_mm_ratio = process_frame(frame, self.px_to_mm_ratio)
        
        # Calculate FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Convert back to BGR for display
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        # Add FPS text if enabled
        if 'SHOW_FPS' in globals() and SHOW_FPS:
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return processed_frame

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Get or create the video callback instance
    if 'video_callback' not in st.session_state:
        st.session_state.video_callback = VideoCallback()
    
    # Process the frame
    processed_img = st.session_state.video_callback.process_frame(img)
    
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# Function to reset detection summary
def reset_detection_summary():
    st.session_state.tracked_objects = {}
    with summary_placeholder:
        st.empty()

# Main app
st.title("🔍 Screw Detection and Measurement (YOLOv11 OBB)")

# Create placeholders for content
frame_placeholder = st.empty()
main_content = st.container()  # Main content goes here

# Create placeholder at the bottom for summary
st.markdown("---")  # Add divider
summary_placeholder = st.container()  # Summary will be displayed here

def show_summary():
    """Display the detection summary with counts for each category"""
    if SHOW_SUMMARY:
        # Get all unique detected objects from session state
        if hasattr(st.session_state, 'tracked_objects') and st.session_state.tracked_objects:
            # Filter out the coin class
            filtered_objects = {
                obj_id: class_name
                for obj_id, class_name in st.session_state.tracked_objects.items()
                if class_name != "10sen Coin"
            }
            
            # Count objects by class name
            class_counts = Counter(filtered_objects.values())
            
            summary_text = "### ✨ Unique Detections ✨\n"
            
            # Display total (excluding the coin)
            total_objects = len(filtered_objects)
            summary_text += f"- **Total unique objects detected (excluding coin): {total_objects}**\n\n"
            
            # Display count for each category
            summary_text += "#### Breakdown by Category:\n"
            for class_name, count in sorted(class_counts.items()):
                # Get color for this class
                color = CATEGORY_COLORS.get(class_name, (0, 255, 0))
                color_hex = "#{:02x}{:02x}{:02x}".format(*color)
                
                # Add colored category count
                summary_text += f"- <span style='color:{color_hex}'><b>{class_name}</b>: {count}</span>\n"
            
            with summary_placeholder:
                st.markdown(summary_text, unsafe_allow_html=True)
        else:
            with summary_placeholder:
                st.info("No screws or nuts detected yet.")

if input_method == "Upload Image":
    reset_detection_summary()  # Clear summary
    with main_content:
        st.subheader("Image Input")
        image_input_method = st.radio("Choose Input Method:", ("Upload", "Capture"))

        if image_input_method == "Upload":
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                frame = np.array(image)
            else:
                frame = None
        elif image_input_method == "Capture":
            captured_image = st.camera_input("Take a Picture")
            if captured_image is not None:
                frame = cv2.imdecode(np.frombuffer(captured_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            else:
                frame = None

        if frame is not None:
            processed_frame, detected_objects, _ = process_frame(frame)
            st.image(processed_frame, channels="RGB")
            show_summary()

elif input_method == "Upload Video":
    reset_detection_summary()  # Clear summary
    with main_content:
        st.subheader("Video Input")
        video_input_method = st.radio("Choose Input Method:", ("Upload", "Capture"))

        if video_input_method == "Upload":
            uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                video_path = tfile.name
            else:
                video_path = None
        elif video_input_method == "Capture":
            captured_video = st.camera_input("Record a Video")
            if captured_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(captured_video.getvalue())
                video_path = tfile.name
            else:
                video_path = None

        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
            px_to_mm_ratio = None

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Placeholders
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🎥 Input Video")
                original_frame_placeholder = st.empty()
            with col2:
                st.markdown("### 🛠️ Processed Video")
                processed_frame_placeholder = st.empty()

            progress_bar = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame, _, px_to_mm_ratio = process_frame(frame, px_to_mm_ratio)

                # Display frames
                original_frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                processed_frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)

                # Update progress
                current_frame += 1
                progress_bar.progress(min(current_frame / frame_count, 1.0))
                time.sleep(1 / fps)

            cap.release()
            show_summary()

elif input_method == "Webcam (Live Camera)":
    reset_detection_summary()  # Clear summary
    with main_content:
        st.subheader("Live Camera Detection")
        webrtc_ctx = webrtc_streamer(
            key="screw-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": WEBCAM_WIDTH},
                    "height": {"ideal": WEBCAM_HEIGHT},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True
        )
        show_summary()
