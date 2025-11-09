import cv2
import numpy as np
from datetime import datetime
import os

def analyze_eye_image(input_path: str, output_path: str) -> dict:
    # Step 1: Load image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError("❌ Image not found. Check your input path.")
    
    # Step 2: Preprocess (grayscale + blur)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Step 3: Detect Pupil/Iris using HoughCircles
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20,
        param1=50, param2=30, minRadius=20, maxRadius=120
    )
    
    pupil_diameter_h = pupil_diameter_v = iris_diameter = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:  # take first circle (main pupil)
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img, center, radius, (0, 255, 0), 2)
            cv2.circle(img, center, 2, (0, 0, 255), 3)
            pupil_diameter_h = pupil_diameter_v = radius * 2
            iris_diameter = radius * 2.5  # approximate ratio

    # Step 4: Redness Index
    red_channel = img[:, :, 2]
    redness_index = np.mean(red_channel) / 255.0

    # Step 5: Dryness Index (based on reflection/brightness)
    brightness = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reflections = np.sum(brightness > 200)
    total_pixels = brightness.size
    dryness_index = 1 - (reflections / total_pixels)

    # Step 6: Create 2x2 grid (original, edges, circle overlay, dryness map)
    edge = cv2.Canny(gray, 100, 200)
    edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    # Dryness heatmap visualization
    dryness_map = cv2.applyColorMap(brightness, cv2.COLORMAP_JET)

    top = np.hstack((img, edge_color))
    bottom = np.hstack((dryness_map, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
    combined = np.vstack((top, bottom))

    # Step 7: Add text report on bottom white strip
    report_strip = np.ones((150, combined.shape[1], 3), dtype=np.uint8) * 255
    report_text = f"Pupil: {pupil_diameter_h}px | Redness: {redness_index:.2f} | Dryness: {dryness_index:.2f}"
    cv2.putText(report_strip, report_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    output_image = np.vstack((combined, report_strip))
    cv2.imwrite(output_path, output_image)

    # Step 8: Create full report
    report = {
        "pupil_diameter_horizontal": pupil_diameter_h,
        "pupil_diameter_vertical": pupil_diameter_v,
        "iris_diameter": iris_diameter,
        "redness_index": float(redness_index),
        "dryness_index": float(dryness_index),
        "reflection_count": int(reflections),
        "timestamp": datetime.now().isoformat()
    }

    # Step 9: Print report
    print("\n🧠 EYE ANALYSIS REPORT")
    print("---------------------------")
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"\n✅ Output saved at: {output_path}")

    return report


# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    input_image_path = r"eye_pic.jpeg"
    output_image_path = r"eye_report.jpeg"
    
    report = analyze_eye_image(input_image_path, output_image_path)
