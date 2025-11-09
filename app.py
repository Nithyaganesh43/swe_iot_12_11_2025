import cv2
import numpy as np
import os
import json
import requests
from datetime import datetime
from flask import Flask, Response, render_template, jsonify
import threading
import time

# --- CONFIGURATION ---
ESP32_STREAM_URL = "http://100.84.46.14:8080/video" 

# --- AI MODEL LOADING ---
EYE_CASCADE_PATH = 'haarcascade_eye.xml' 

if not os.path.exists(EYE_CASCADE_PATH):
    print(f"❌ ERROR: Cannot find {EYE_CASCADE_PATH}")
    print("Please download it from OpenCV's GitHub.")
    exit()

eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Disable caching

# --- THREAD-SAFE FRAME STORAGE ---
# We will have one background thread reading the stream
# and storing the latest *JPEG BYTES* here.
latest_frame_bytes = None
frame_lock = threading.Lock()

# --- BACKGROUND THREAD FOR VIDEO READING (NEW ROBUST METHOD) ---
def video_stream_reader():
    """
    Continuously reads the MJPEG stream from the ESP32 using 'requests'
    and manually parses JPEG frames. This is more robust than cv2.VideoCapture.
    """
    global latest_frame_bytes
    
    print(f"🌎 Starting robust stream reader thread for {ESP32_STREAM_URL}...")
    
    while True:
        try:
            # Set a longer timeout for the initial connection
            r = requests.get(ESP32_STREAM_URL, stream=True, timeout=10.0)
            r.raise_for_status()
            print("✅ (Thread) Stream connection established.")
            
            stream_bytes = b''
            
            # Read the stream chunk by chunk
            for chunk in r.iter_content(chunk_size=1024):
                stream_bytes += chunk
                
                # Look for JPEG start and end markers
                a = stream_bytes.find(b'\xff\xd8') # Start of JPEG
                b = stream_bytes.find(b'\xff\xd9') # End of JPEG
                
                if a != -1 and b != -1:
                    # We have a full frame
                    jpg = stream_bytes[a:b+2]
                    
                    # Update the global frame
                    with frame_lock:
                        latest_frame_bytes = jpg
                        
                    # Remove this frame from the buffer
                    stream_bytes = stream_bytes[b+2:]
            
            # If loop exits, stream ended
            print("❌ (Thread) Stream ended unexpectedly. Reconnecting in 5s...")

        except requests.exceptions.ConnectionError as e:
            print(f"❌ (Thread) Connection Error: {e}. Retrying in 5s...")
        except requests.exceptions.Timeout as e:
            print(f"❌ (Thread) Connection Timeout: {e}. Retrying in 5s...")
        except requests.exceptions.RequestException as e:
            print(f"❌ (Thread) Generic Request Error: {e}. Retrying in 5s...")
        except Exception as e:
            print(f"❌ (Thread) Unknown Error in stream reader: {e}")
            
        # If any error occurs, clear the frame and wait before retrying
        with frame_lock:
            latest_frame_bytes = None
        time.sleep(5)


# --- CORE IMAGE ANALYSIS (Unchanged) ---

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
    # Use a bilateral filter to reduce noise while keeping edges sharp.
    # This is better than medianBlur for preserving the pupil edge.
    blur = cv2.bilateralFilter(eye_roi_gray, 9, 75, 75)
    
    # 2. Thresholding to isolate dark regions (pupil)
    # --- OLD METHOD ---
    # A hardcoded threshold is not robust to lighting changes.
    # ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    
    # --- NEW ROBUST METHOD ---
    # Use adaptive thresholding. It automatically finds the best
    # threshold for local regions, making it robust to lighting.
    # Block size (25) must be odd.
    # C (4) is a constant subtracted from the mean, helping to fine-tune.
    thresh = cv2.adaptiveThreshold(blur, 255, 
                                   cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 
                                   25, 
                                   4)

    # 3. Morphological Cleaning (NEW STEP)
    # Use MORPH_OPEN to remove small white noise specks (e.g., from
    # reflections or eyelashes) which could be mistaken for contours.
    kernel = np.ones((3, 3), np.uint8)
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. Find Contours
    # --- UPDATED ---
    # Find contours on the *cleaned* thresholded image, not the raw one.
    contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None, None # No contours found

    # 5. Filter Contours
    # We're looking for the contour that is most likely the pupil.
    # (This logic was already good and remains unchanged)
    
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
            
        # Circularity: 1.0 is a perfect circle
        circularity = (4 * np.pi * area) / (perimeter**2)
        
        # We score based on a combination of area and circularity
        score = (area * 0.6) + (circularity * 100 * 0.4)
        
        if score > best_score:
            best_score = score
            best_contour = c

    if best_contour is None:
        # If no contour passed our filter, try a fallback of just finding the largest one.
        try:
            best_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best_contour) < min_pupil_area:
                return None, None, None, None # Largest is still too small
        except ValueError:
             return None, None, None, None # contours list might be empty

    # 6. Get Circle from best contour
    (x, y), radius = cv2.minEnclosingCircle(best_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    if radius < 5: # Final check
        return None, None, None, None # Pupil too small

    # --- Calculate Metrics ---
    # (This logic remains unchanged)
    
    pupil_diameter = int(radius * 2)
    iris_diameter = pupil_diameter * 2.5 
    iris_radius = int(iris_diameter / 2)
    pi_ratio = pupil_diameter / iris_diameter if iris_diameter > 0 else 0
    
    # Draw overlays
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
        redness_index = 0 # Avoid division by zero if mask is empty
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
    
    # Create all 4 analysis images
    red_channel_img = cv2.applyColorMap(eye_roi_color[:, :, 2], cv2.COLORMAP_HOT)
    reflection_map_img = cv2.cvtColor(reflection_map, cv2.COLOR_GRAY2BGR)
    sclera_mask_img = cv2.cvtColor(sclera_mask, cv2.COLOR_GRAY2BGR)
    
    return report, overlay_img, red_channel_img, reflection_map_img, sclera_mask_img

def process_frame(frame: np.ndarray) -> (np.ndarray, dict):
    """
    Analyzes a single video frame, finds *two* eyes, and generates a
    comparative report with a new visual layout.
    """
    if frame is None:
        return create_error_image(None, "Frame was empty.")

    # --- Step 1: Detect *all* eyes ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, minNeighbors=8) 
    
    if len(eyes) != 2:
        return create_error_image(frame, f"Detected {len(eyes)} eyes. Please capture exactly two.")

    eyes = sorted(eyes, key=lambda e: e[0])
    
    left_eye_rect = eyes[0]
    right_eye_rect = eyes[1]

    # --- Step 2: Analyze Left Eye ---
    (ex, ey, ew, eh) = left_eye_rect
    left_eye_roi_gray = gray[ey:ey+eh, ex:ex+ew]
    left_eye_roi_color = frame[ey:ey+eh, ex:ex+ew]
    
    (left_report, left_overlay, left_red, 
     left_reflection, left_sclera) = analyze_single_eye(left_eye_roi_gray, left_eye_roi_color)
    
    if left_report is None:
        return create_error_image(frame, "Pupil detection failed on LEFT eye.")

    # --- Step 3: Analyze Right Eye ---
    (ex, ey, ew, eh) = right_eye_rect
    right_eye_roi_gray = gray[ey:ey+eh, ex:ex+ew]
    right_eye_roi_color = frame[ey:ey+eh, ex:ex+ew]
    
    (right_report, right_overlay, right_red, 
     right_reflection, right_sclera) = analyze_single_eye(right_eye_roi_gray, right_eye_roi_color)

    if right_report is None:
        return create_error_image(frame, "Pupil detection failed on RIGHT eye.")

    # --- Step 4: Calculate Interpupillary Distance (IPD) ---
    left_pupil_abs = (
        left_eye_rect[0] + left_report['pupil_center_x_rel'],
        left_eye_rect[1] + left_report['pupil_center_y_rel']
    )
    right_pupil_abs = (
        right_eye_rect[0] + right_report['pupil_center_x_rel'],
        right_eye_rect[1] + right_report['pupil_center_y_rel']
    )
    
    ipd_px = np.sqrt( (left_pupil_abs[0] - right_pupil_abs[0])**2 + (left_pupil_abs[1] - right_pupil_abs[1])**2 )
    
    # --- Step 5: Create TOP HALF (Dual 2x2 Grid) ---
    grid_size = (320, 240) # Each quadrant
    
    # Create Left 2x2 Grid
    l_img1 = cv2.resize(left_overlay, grid_size)
    cv2.putText(l_img1, "Left Eye Overlay", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    l_img2 = cv2.resize(left_red, grid_size)
    cv2.putText(l_img2, "Left Red Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    l_img3 = cv2.resize(left_reflection, grid_size)
    cv2.putText(l_img3, "Left Reflections", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    l_img4 = cv2.resize(left_sclera, grid_size)
    cv2.putText(l_img4, "Left Sclera Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    left_grid = np.vstack((np.hstack((l_img1, l_img2)), np.hstack((l_img3, l_img4))))

    # Create Right 2x2 Grid
    r_img1 = cv2.resize(right_overlay, grid_size)
    cv2.putText(r_img1, "Right Eye Overlay", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    r_img2 = cv2.resize(right_red, grid_size)
    cv2.putText(r_img2, "Right Red Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    r_img3 = cv2.resize(right_reflection, grid_size)
    cv2.putText(r_img3, "Right Reflections", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    r_img4 = cv2.resize(right_sclera, grid_size)
    cv2.putText(r_img4, "Right Sclera Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    right_grid = np.vstack((np.hstack((r_img1, r_img2)), np.hstack((r_img3, r_img4))))
    
    # Combine grids
    top_half = np.hstack((left_grid, right_grid)) # Final size: (480, 1280)
    
    # --- Step 6: Create BOTTOM HALF (Summary Diagram) ---
    diagram_img = frame.copy()
    
    # Draw Eye Boxes
    (lx, ly, lw, lh) = left_eye_rect
    cv2.rectangle(diagram_img, (lx, ly), (lx+lw, ly+lh), (0, 255, 255), 2)
    (rx, ry, rw, rh) = right_eye_rect
    cv2.rectangle(diagram_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
    
    # Draw Pupil Centers
    cv2.circle(diagram_img, (int(left_pupil_abs[0]), int(left_pupil_abs[1])), 5, (0, 255, 0), -1)
    cv2.circle(diagram_img, (int(right_pupil_abs[0]), int(right_pupil_abs[1])), 5, (0, 255, 0), -1)
    
    # Draw IPD Line and Text
    cv2.line(diagram_img, (int(left_pupil_abs[0]), int(left_pupil_abs[1])), (int(right_pupil_abs[0]), int(right_pupil_abs[1])), (0, 255, 0), 2)
    
    # Calculate midpoint for text
    mid_x = int((left_pupil_abs[0] + right_pupil_abs[0]) / 2)
    mid_y = int((left_pupil_abs[1] + right_pupil_abs[1]) / 2) - 10
    ipd_text = f"IPD: {ipd_px:.0f} px"
    cv2.putText(diagram_img, ipd_text, (mid_x - 50, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Resize bottom half to match top half width (1280)
    target_width = 1280
    orig_h, orig_w = diagram_img.shape[:2]
    target_height = int(orig_h * (target_width / orig_w))
    bottom_half = cv2.resize(diagram_img, (target_width, target_height))

    # --- Step 7: Add Text Report Strip ---
    report_strip = np.ones((180, 1280, 3), dtype=np.uint8) * 255
    
    # Column titles
    cv2.putText(report_strip, "LEFT EYE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(report_strip, "RIGHT EYE", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    # IPD
    cv2.putText(report_strip, f"Interpupillary Distance (IPD): {ipd_px:.0f} px", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
    
    # Data
    y_offset = 60
    metrics = [
        ("Pupil", "pupil_diameter_px", "px"),
        ("P/I Ratio", "pi_ratio", ""),
        ("Redness", "redness_index", ""),
        ("Dryness", "dryness_index", "")
    ]
    
    for label, key, unit in metrics:
        # Left column
        left_val = left_report[key]
        left_txt = f"{label}: {left_val} {unit}"
        cv2.putText(report_strip, left_txt, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        
        # Right column
        right_val = right_report[key]
        right_txt = f"{label}: {right_val} {unit}"
        cv2.putText(report_strip, right_txt, (650, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        
        y_offset += 25

    # --- Step 8: Combine all parts ---
    output_image = np.vstack((top_half, bottom_half, report_strip))
    
    # --- Step 9: Create Final JSON Report ---
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "interpupillary_distance_px": float(f"{ipd_px:.2f}"),
        "left_eye_box": { "x": int(left_eye_rect[0]), "y": int(left_eye_rect[1]), "w": int(left_eye_rect[2]), "h": int(left_eye_rect[3]) },
        "right_eye_box": { "x": int(right_eye_rect[0]), "y": int(right_eye_rect[1]), "w": int(right_eye_rect[2]), "h": int(right_eye_rect[3]) },
        "left_eye_report": left_report,
        "right_eye_report": right_report
    }
    
    return output_image, final_report

# --- FLASK WEB ROUTES (Refactored) ---

@app.route('/')
def index():
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
        
        # Get a thread-safe copy of the frame bytes
        with frame_lock:
            if latest_frame_bytes is not None:
                frame_bytes = latest_frame_bytes
        
        if frame_bytes is None:
            # Use the pre-encoded offline image
            if offline_jpg:
                frame_bytes = offline_jpg
            else:
                # Failsafe
                time.sleep(0.5)
                continue
        
        # Yield the frame in the multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Control frame rate
        time.sleep(0.05) # ~20 FPS

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route.
    This now uses the 'generate_frames' generator which reads
    from the background thread's 'latest_frame'.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    """
    Capture route.
    This now reads from the 'latest_frame_bytes' variable and decodes it.
    """
    
    frame_to_process_bytes = None
    
    # Get a thread-safe copy of the latest frame bytes
    with frame_lock:
        if latest_frame_bytes is not None:
            frame_to_process_bytes = latest_frame_bytes
    
    if frame_to_process_bytes is None:
        print("❌ Capture failed: Stream is offline (latest_frame_bytes is None).")
        return jsonify({"success": False, "error": "Cannot connect to camera stream."})
    
    # --- DECODE THE JPEG BYTES ---
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
    
    if processed_img is None:
        return jsonify({"success": False, "error": "Failed to process frame."})

    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    img_filename = f"capture_{now_str}.jpg"
    data_filename = f"capture_{now_str}.json"
    img_save_path = os.path.join('static', 'captures', img_filename)
    data_save_path = os.path.join('static', 'captures', data_filename)
    
    cv2.imwrite(img_save_path, processed_img)
    
    with open(data_save_path, 'w') as f:
        json.dump(report_data, f, indent=4)
        
    print(f"✅ Capture saved: {img_filename} and {data_filename}")

    return jsonify({
        "success": True,
        "image_url": f"/static/captures/{img_filename}",
        "data_url": f"/static/captures/{data_filename}",
        "data": report_data
    })

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/captures', exist_ok=True)
    
    # Start the background thread
    # daemon=True means the thread will exit when the main program exits
    stream_thread = threading.Thread(target=video_stream_reader, daemon=True)
    stream_thread.start()
    
    print(f"🚀 Starting server... Open http://127.0.0.1:5000 in your browser.")
    # Run the Flask app
    # threaded=True is important to handle multiple requests (e.g., / and /video_feed)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)