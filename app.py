import cv2
import numpy as np
import os
import json
import requests
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, send_from_directory
import threading
import time

# --- CONFIGURATION ---
# We are locked to the webcam as requested
# VIDEO_SOURCE = 0                                    # Use local webcam
# VIDEO_SOURCE = "http://192.168.1.4:8080/video"      # Use ESP32 Stream 1
VIDEO_SOURCE = "http://10.225.59.111/stream"        # Use ESP32 Stream 2

# --- AI MODEL LOADING ---
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = 'haarcascade_eye.xml' 

if not os.path.exists(EYE_CASCADE_PATH) or not os.path.exists(FACE_CASCADE_PATH):
    print(f"❌ ERROR: Cannot find cascade files.")
    print(f"Ensure both {FACE_CASCADE_PATH} and {EYE_CASCADE_PATH} exist.")
    print("Please download them from OpenCV's GitHub.")
    exit()

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
print("✅ Face and Eye detection models loaded.")


# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Disable caching

# --- THREAD-SAFE FRAME STORAGE ---
latest_frame_bytes = None
frame_lock = threading.Lock()

# --- BACKGROUND THREADS ---

def read_from_url(stream_url):
    """
    Continuously reads the MJPEG stream from a URL using 'requests'
    and manually parses JPEG frames.
    """
    global latest_frame_bytes
    
    print(f"🌎 Starting URL stream reader thread for {stream_url}...")
    
    while True:
        try:
            r = requests.get(stream_url, stream=True, timeout=10.0)
            r.raise_for_status()
            print("✅ (Thread) Stream connection established.")
            
            stream_bytes = b''
            
            for chunk in r.iter_content(chunk_size=1024):
                stream_bytes += chunk
                a = stream_bytes.find(b'\xff\xd8') # Start of JPEG
                b = stream_bytes.find(b'\xff\xd9') # End of JPEG
                
                if a != -1 and b != -1:
                    jpg = stream_bytes[a:b+2]
                    with frame_lock:
                        latest_frame_bytes = jpg
                    stream_bytes = stream_bytes[b+2:]
            
            print("❌ (Thread) Stream ended unexpectedly. Reconnecting in 5s...")

        except requests.exceptions.ConnectionError as e:
            print(f"❌ (Thread) Connection Error: {e}. Retrying in 5s...")
        except requests.exceptions.Timeout as e:
            print(f"❌ (Thread) Connection Timeout: {e}. Retrying in 5s...")
        except requests.exceptions.RequestException as e:
            print(f"❌ (Thread) Generic Request Error: {e}. Retrying in 5s...")
        except Exception as e:
            print(f"❌ (Thread) Unknown Error in stream reader: {e}")
            
        with frame_lock:
            latest_frame_bytes = None
        time.sleep(5)

def read_from_webcam(device_index):
    """
    Continuously reads from a local webcam using cv2.VideoCapture
    and encodes frames to JPEG bytes.
    """
    global latest_frame_bytes
    print(f"📸 Starting Webcam reader thread for device {device_index}...")
    
    while True:
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            print(f"❌ (Thread) Cannot open webcam {device_index}. Retrying in 5s...")
            with frame_lock:
                latest_frame_bytes = None
            time.sleep(5)
            continue
        
        print(f"✅ (Thread) Webcam {device_index} opened.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ (Thread) Webcam frame read failed. Re-initializing...")
                break # Break inner loop to re-initialize cap
            
            # Flip the frame horizontally (webcams are often mirrored)
            frame = cv2.flip(frame, 1) 
            
            # Encode frame to JPEG bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("❌ (Thread) JPEG encoding failed.")
                continue
                
            # Update the global frame
            with frame_lock:
                latest_frame_bytes = buffer.tobytes()
                
            # Control frame rate
            time.sleep(0.05) # ~20 FPS
        
        cap.release()
        with frame_lock:
            latest_frame_bytes = None
        time.sleep(1) # Wait a sec before retrying

def video_stream_manager():
    """
    Checks the VIDEO_SOURCE config and starts the correct
    reader function. This is the target for the background thread.
    """
    if isinstance(VIDEO_SOURCE, int):
        # It's a webcam index
        read_from_webcam(VIDEO_SOURCE)
    elif isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.startswith('http'):
        # It's a URL
        read_from_url(VIDEO_SOURCE)
    else:
        print(f"❌ ERROR: Invalid VIDEO_SOURCE: {VIDEO_SOURCE}")
        print("Set to 0 for webcam or a 'http://...' URL string.")


# --- CORE IMAGE ANALYSIS ---

def create_error_image(frame: np.ndarray, error_message: str) -> (np.ndarray, dict):
    """Creates a 2x2 grid with an error message."""
    grid_size = (320, 240)
    
    error_img = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    error_img = cv2.resize(error_img, (grid_size[0]*2, grid_size[1]*2))
    
    overlay = np.zeros_like(error_img, dtype=np.uint8)
    overlay = cv2.rectangle(overlay, (0,0), (grid_size[0]*2, 100), (0,0,0), -1)
    
    final_img = cv2.addWeighted(error_img, 0.7, overlay, 0.3, 0)
    cv2.putText(final_img, "ANALYSIS FAILED", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(final_img, error_message, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    report = {"error": error_message, "timestamp": datetime.now().isoformat()}
    return final_img, report

def analyze_single_eye(eye_roi_gray, eye_roi_color):
    """
    Analyzes a single cropped eye image using a robust contour-finding method.
    Returns: (report_dict, overlay_img, red_channel_img, reflection_map_img, sclera_mask_img) 
    Returns (None, ...) on failure.
    """
    
    overlay_img = eye_roi_color.copy()

    # --- Find Pupil (IMPROVED CONTOUR-BASED METHOD) ---
    
    # 1. Preprocessing
    blur = cv2.bilateralFilter(eye_roi_gray, 9, 75, 75)
    
    # 2. Thresholding to isolate dark regions (pupil)
    thresh = cv2.adaptiveThreshold(blur, 255, 
                                     cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY_INV, 
                                     25, 
                                     4)

    # 3. Morphological Cleaning
    kernel = np.ones((3, 3), np.uint8)
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. Find Contours
    contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None, None # No contours found

    # 5. Filter Contours
    best_contour = None
    best_score = 0
    min_pupil_area = 50 # 50 pixels
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_pupil_area:
            continue
            
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
            
        circularity = (4 * np.pi * area) / (perimeter**2)
        score = (area * 0.6) + (circularity * 100 * 0.4)
        
        if score > best_score:
            best_score = score
            best_contour = c

    if best_contour is None:
        try:
            best_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best_contour) < min_pupil_area:
                return None, None, None, None
        except ValueError:
             return None, None, None, None

    # 6. Get Circle from best contour
    (x, y), radius = cv2.minEnclosingCircle(best_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    if radius < 5: # Final check
        return None, None, None, None

    # --- Calculate Metrics ---
    
    pupil_diameter = int(radius * 2)
    iris_diameter = pupil_diameter * 2.5 
    iris_radius = int(iris_diameter / 2)
    pi_ratio = pupil_diameter / iris_diameter if iris_diameter > 0 else 0
    
    cv2.circle(overlay_img, center, radius, (0, 255, 0), 2) # Pupil
    cv2.circle(overlay_img, center, iris_radius, (0, 255, 255), 1) # Iris

    # --- Redness Index ---
    sclera_mask = np.zeros_like(eye_roi_gray)
    sclera_outer_radius = int(iris_radius * 1.8)
    cv2.circle(sclera_mask, center, sclera_outer_radius, 255, -1) 
    cv2.circle(sclera_mask, center, iris_radius, 0, -1)     
    
    red_channel = eye_roi_color[:, :, 2]
    green_channel = eye_roi_color[:, :, 1]
    
    sclera_red_pixels = red_channel[sclera_mask == 255]
    sclera_green_pixels = green_channel[sclera_mask == 255]

    if len(sclera_red_pixels) == 0 or len(sclera_green_pixels) == 0:
        redness_index = 0
    else:
        sclera_red = np.mean(sclera_red_pixels)
        sclera_green = np.mean(sclera_green_pixels)
        redness_index = sclera_red / sclera_green if sclera_green > 0 else 0
        
    if np.isnan(redness_index): redness_index = 0

    # --- Dryness Index ---
    iris_mask = np.zeros_like(eye_roi_gray)
    cv2.circle(iris_mask, center, iris_radius, 255, -1)
    
    ret, reflection_map_full = cv2.threshold(eye_roi_gray, 190, 255, cv2.THRESH_BINARY)
    reflection_map = cv2.bitwise_and(reflection_map_full, reflection_map_full, mask=iris_mask)
    
    reflection_pixels = np.sum(reflection_map == 255)
    iris_pixels = np.sum(iris_mask == 255)
    
    dryness_index = (reflection_pixels / (iris_pixels + 1)) * 10000 

    # --- Prepare return data ---
    report = {
        "pupil_center_x_rel": center[0], # Relative to the crop
        "pupil_center_y_rel": center[1],
        "pupil_diameter_px": int(pupil_diameter),
        "iris_diameter_px": int(iris_diameter),
        "pi_ratio": float(f"{pi_ratio:.3f}"),
        "redness_index": float(f"{redness_index:.3f}"),
        "dryness_index": float(f"{dryness_index:.3f}"),
        "reflection_pixels": int(reflection_pixels)
    }
    
    red_channel_img = cv2.applyColorMap(eye_roi_color[:, :, 2], cv2.COLORMAP_HOT)
    reflection_map_img = cv2.cvtColor(reflection_map, cv2.COLOR_GRAY2BGR)
    sclera_mask_img = cv2.cvtColor(sclera_mask, cv2.COLOR_GRAY2BGR)
    
    return report, overlay_img, red_channel_img, reflection_map_img, sclera_mask_img

def process_frame(frame: np.ndarray) -> (np.ndarray, dict):
    """
    Analyzes a single video frame using the Face -> Eyes pipeline.
    """
    if frame is None:
        return create_error_image(None, "Frame was empty.")

    # --- Step 1: Detect Face ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    if len(faces) == 0:
        return create_error_image(frame, "No face detected. Please face the camera.")
    if len(faces) > 1:
        return create_error_image(frame, "Multiple faces detected. Only one person.")

    (fx, fy, fw, fh) = faces[0]
    
    # --- Step 2: Detect Eyes *within* the Face ROI ---
    face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
    
    eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(fw//6, fh//6)) 
    
    if len(eyes) != 2:
        return create_error_image(frame, f"Detected {len(eyes)} eyes. Please look at the camera.")

    absolute_eyes = []
    for (ex, ey, ew, eh) in eyes:
        absolute_rect = (fx + ex, fy + ey, ew, eh)
        absolute_eyes.append(absolute_rect)

    eyes = sorted(absolute_eyes, key=lambda e: e[0])
    
    left_eye_rect = eyes[0]
    right_eye_rect = eyes[1]

    # --- Step 3: Analyze Left Eye ---
    (ex, ey, ew, eh) = left_eye_rect
    left_eye_roi_gray = gray[ey:ey+eh, ex:ex+ew]
    left_eye_roi_color = frame[ey:ey+eh, ex:ex+ew]
    
    (left_report, left_overlay, left_red, 
     left_reflection, left_sclera) = analyze_single_eye(left_eye_roi_gray, left_eye_roi_color)
    
    if left_report is None:
        return create_error_image(frame, "Pupil detection failed on LEFT eye.")

    # --- Step 4: Analyze Right Eye ---
    (ex, ey, ew, eh) = right_eye_rect
    right_eye_roi_gray = gray[ey:ey+eh, ex:ex+ew]
    right_eye_roi_color = frame[ey:ey+eh, ex:ex+ew]
    
    (right_report, right_overlay, right_red, 
     right_reflection, right_sclera) = analyze_single_eye(right_eye_roi_gray, right_eye_roi_color)

    if right_report is None:
        return create_error_image(frame, "Pupil detection failed on RIGHT eye.")

    # --- Step 5: Calculate Interpupillary Distance (IPD) ---
    left_pupil_abs = (
        left_eye_rect[0] + left_report['pupil_center_x_rel'],
        left_eye_rect[1] + left_report['pupil_center_y_rel']
    )
    right_pupil_abs = (
        right_eye_rect[0] + right_report['pupil_center_x_rel'],
        right_eye_rect[1] + right_report['pupil_center_y_rel']
    )
    
    ipd_px = np.sqrt( (left_pupil_abs[0] - right_pupil_abs[0])**2 + (left_pupil_abs[1] - right_pupil_abs[1])**2 )
    
    # --- Step 6: Create TOP HALF (Dual 2x2 Grid) ---
    grid_size = (320, 240) # Each quadrant
    
    l_img1 = cv2.resize(left_overlay, grid_size)
    cv2.putText(l_img1, "Left Eye Overlay", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    l_img2 = cv2.resize(left_red, grid_size)
    cv2.putText(l_img2, "Left Red Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    l_img3 = cv2.resize(left_reflection, grid_size)
    cv2.putText(l_img3, "Left Reflections", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    l_img4 = cv2.resize(left_sclera, grid_size)
    cv2.putText(l_img4, "Left Sclera Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    left_grid = np.vstack((np.hstack((l_img1, l_img2)), np.hstack((l_img3, l_img4))))

    r_img1 = cv2.resize(right_overlay, grid_size)
    cv2.putText(r_img1, "Right Eye Overlay", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    r_img2 = cv2.resize(right_red, grid_size)
    cv2.putText(r_img2, "Right Red Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    r_img3 = cv2.resize(right_reflection, grid_size)
    cv2.putText(r_img3, "Right Reflections", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    r_img4 = cv2.resize(right_sclera, grid_size)
    cv2.putText(r_img4, "Right Sclera Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    right_grid = np.vstack((np.hstack((r_img1, r_img2)), np.hstack((r_img3, r_img4))))
    
    top_half = np.hstack((left_grid, right_grid)) # Final size: (480, 1280)
    
    # --- Step 7: Create BOTTOM HALF (Summary Diagram) ---
    diagram_img = frame.copy()
    
    (lx, ly, lw, lh) = left_eye_rect
    cv2.rectangle(diagram_img, (lx, ly), (lx+lw, ly+lh), (0, 255, 255), 2)
    (rx, ry, rw, rh) = right_eye_rect
    cv2.rectangle(diagram_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
    
    cv2.circle(diagram_img, (int(left_pupil_abs[0]), int(left_pupil_abs[1])), 5, (0, 255, 0), -1)
    cv2.circle(diagram_img, (int(right_pupil_abs[0]), int(right_pupil_abs[1])), 5, (0, 255, 0), -1)
    
    cv2.line(diagram_img, (int(left_pupil_abs[0]), int(left_pupil_abs[1])), (int(right_pupil_abs[0]), int(right_pupil_abs[1])), (0, 255, 0), 2)
    
    mid_x = int((left_pupil_abs[0] + right_pupil_abs[0]) / 2)
    mid_y = int((left_pupil_abs[1] + right_pupil_abs[1]) / 2) - 10
    ipd_text = f"IPD: {ipd_px:.0f} px"
    cv2.putText(diagram_img, ipd_text, (mid_x - 50, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    target_width = 1280
    orig_h, orig_w = diagram_img.shape[:2]
    target_height = int(orig_h * (target_width / orig_w))
    bottom_half = cv2.resize(diagram_img, (target_width, target_height))

    # --- Step 8: Add Text Report Strip ---
    report_strip = np.ones((180, 1280, 3), dtype=np.uint8) * 255
    
    cv2.putText(report_strip, "LEFT EYE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(report_strip, "RIGHT EYE", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(report_strip, f"Interpupillary Distance (IPD): {ipd_px:.0f} px", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
    
    y_offset = 60
    metrics = [
        ("Pupil", "pupil_diameter_px", "px"),
        ("P/I Ratio", "pi_ratio", ""),
        ("Redness", "redness_index", ""),
        ("Dryness", "dryness_index", "")
    ]
    
    for label, key, unit in metrics:
        left_val = left_report[key]
        left_txt = f"{label}: {left_val} {unit}"
        cv2.putText(report_strip, left_txt, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        
        right_val = right_report[key]
        right_txt = f"{label}: {right_val} {unit}"
        cv2.putText(report_strip, right_txt, (650, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        
        y_offset += 25

    # --- Step 9: Combine all parts ---
    output_image = np.vstack((top_half, bottom_half, report_strip))
    
    # --- Step 10: Create Final JSON Report ---
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "interpupillary_distance_px": float(f"{ipd_px:.2f}"),
        "left_eye_box": { "x": int(left_eye_rect[0]), "y": int(left_eye_rect[1]), "w": int(left_eye_rect[2]), "h": int(left_eye_rect[3]) },
        "right_eye_box": { "x": int(right_eye_rect[0]), "y": int(right_eye_rect[1]), "w": int(right_eye_rect[2]), "h": int(right_eye_rect[3]) },
        "left_eye_report": left_report,
        "right_eye_report": right_report
    }
    
    return output_image, final_report

# --- FLASK WEB ROUTES ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

def create_offline_jpeg():
    """Helper to create a JPEG byte array for offline/error state."""
    img = np.ones((240, 320, 3), dtype=np.uint8) * 100
    cv2.putText(img, "STREAM OFFLINE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    (flag, encoded_image) = cv2.imencode(".jpg", img)
    if flag:
        return encoded_image.tobytes()
    return None

def generate_frames():
    """
    Generator function to yield frames for the /video_feed route.
    Reads from the global 'latest_frame_bytes' variable.
    """
    offline_jpg = create_offline_jpeg()
    
    while True:
        frame_bytes = None
        
        with frame_lock:
            if latest_frame_bytes is not None:
                frame_bytes = latest_frame_bytes
        
        if frame_bytes is None:
            if offline_jpg:
                frame_bytes = offline_jpg
            else:
                time.sleep(0.5)
                continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.05) # ~20 FPS

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    """Capture and analysis route."""
    
    frame_to_process_bytes = None
    
    with frame_lock:
        if latest_frame_bytes is not None:
            frame_to_process_bytes = latest_frame_bytes
    
    if frame_to_process_bytes is None:
        print("❌ Capture failed: Stream is offline (latest_frame_bytes is None).")
        return jsonify({"success": False, "error": "Cannot connect to camera stream."})
    
    try:
        frame_np_arr = np.frombuffer(frame_to_process_bytes, np.uint8)
        frame_to_process = cv2.imdecode(frame_np_arr, cv2.IMREAD_COLOR)
        if frame_to_process is None:
            raise Exception("cv2.imdecode returned None")
    except Exception as e:
        print(f"❌ Capture failed: Could not decode JPEG frame. Error: {e}")
        return jsonify({"success": False, "error": "Failed to decode frame from stream."})

    print("🧠 Processing captured frame...")
    processed_img, report_data = process_frame(frame_to_process)
    
    if "error" in report_data:
        print(f"❌ Analysis failed: {report_data['error']}")
        return jsonify({"success": False, "error": report_data['error']})
    
    if processed_img is None:
        return jsonify({"success": False, "error": "Failed to process frame."})

    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    img_filename = f"capture_{now_str}.jpg"
    data_filename = f"capture_{now_str}.json"
    
    # Create the 'static/captures' directory if it doesn't exist
    capture_dir = os.path.join('static', 'captures')
    os.makedirs(capture_dir, exist_ok=True)
    
    img_save_path = os.path.join(capture_dir, img_filename)
    data_save_path = os.path.join(capture_dir, data_filename)
    
    cv2.imwrite(img_save_path, processed_img)
    
    with open(data_save_path, 'w') as f:
        json.dump(report_data, f, indent=4)
        
    print(f"✅ Capture saved: {img_filename} and {data_filename}")

    return jsonify({
        "success": True,
        # We use relative URLs for the frontend
        "image_url": f"static/captures/{img_filename}",
        "data_url": f"static/captures/{data_filename}",
        "data": report_data
    })

# --- NEW ROUTE FOR HISTORY DASHBOARD ---
@app.route('/get_captures')
def get_captures():
    """
    Scans the 'static/captures' directory and returns a list
    of all capture.json and capture.jpg pairs, sorted by date.
    """
    capture_dir = os.path.join('static', 'captures')
    captures = []
    
    if not os.path.exists(capture_dir):
        return jsonify([]) # Return empty list if dir doesn't exist

    for filename in os.listdir(capture_dir):
        if filename.endswith('.json'):
            basename = filename.replace('.json', '')
            img_filename = basename + '.jpg'
            img_path = os.path.join(capture_dir, img_filename)
            
            if os.path.exists(img_path):
                try:
                    # Parse timestamp from filename for sorting
                    # Filename format: capture_YYYY-MM-DD_HHMMSS.json
                    timestamp_str = basename.replace('capture_', '')
                    timestamp_obj = datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M%S")
                    
                    captures.append({
                        "json_url": f"static/captures/{filename}",
                        "image_url": f"static/captures/{img_filename}",
                        "timestamp": timestamp_obj.isoformat(),
                        "display_time": timestamp_obj.strftime("%Y-%m-%d %I:%M:%S %p")
                    })
                except ValueError as e:
                    print(f"Skipping file with malformed name: {filename}. Error: {e}")

    # Sort by timestamp, newest first
    captures.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify(captures)


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/captures', exist_ok=True)
    
    # Start the background thread
    stream_thread = threading.Thread(target=video_stream_manager, daemon=True)
    stream_thread.start()
    
    print(f"🚀 Starting server... Open http://127.0.0.1:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)