````markdown
# 👁️ Eye Analysis Web Application

A Python + Flask-based real-time computer vision web app that detects faces and eyes using a webcam or ESP-CAM stream, performs detailed eye analysis, and displays results directly in your browser.

---

## 🚀 Project Run Steps (Simplest Form)

### 1. Turn On Mobile Hotspot
- **Hotspot Name (SSID):** `Sorry`  
- **Password:** `aaaaaaaa`  
- Use a **4G phone**, not 5G.  
- Allow **at least 2 connections** (laptop + ESP device).  

---

### 2. Connect Laptop to Hotspot
- On the laptop, connect Wi-Fi to the hotspot named **Sorry**.

---

### 3. Power On the ESP Device
- Simply **turn on the device**.

---

### 4. Find ESP IP
- Open terminal in the **project folder**.
- Run:
  ```bash
  python ip.py
````

* Wait until you see a line like:

  ```
  10.225.59.111 -> 200 iamespcam
  ```
* **Copy that IP** (example: `10.225.59.111`).

---

### 5. Edit the IP in `app.py`

* Open the file `app.py`.
* Find this line near the top:

  ```python
  VIDEO_SOURCE = "http://xxx.xxx.xxx.xxx/stream"
  ```
* Replace **only the IP part** with your copied one.
  *(Keep `/stream` exactly the same.)*

---

### 6. Install Dependencies

* Open the file **`dependencies.txt`**.
* Copy all its contents.
* In the terminal (same folder), paste and run it to install the required packages.

---

### 7. Run the Server

Run:

```bash
python app.py
```

Wait until you see:

```
🚀 Starting server... Open http://127.0.0.1:5000
```

---

### 8. Open the Webpage

* Copy the shown link:
  `http://127.0.0.1:5000`
* Open it in your browser.

---

### 9. Use the App

* You’ll see the **live camera stream**.
* Click **Capture** to take frames.
* Open **Analyze** to view detailed eye analysis and reports.

---

### 🧩 Technologies Used

* **Python**
* **Flask** (Web Framework)
* **OpenCV** (Computer Vision)
* **Haar Cascades** (Face & Eye Detection)
* **HTML/CSS** (Frontend Templates)

---

### 📂 Folder Structure

```
project_root/
│
├── app.py                  # Main Flask web server
├── pythonip.py             # ESP IP finder script
├── dependencies.htxt       # Dependencies installation list
├── templates/              # Frontend HTML files
├── static/                 # Captured images & analysis data
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
└── haarcascade_eye_tree_eyeglasses.xml
```

---

### ✅ Notes

* Make sure the **ESP-CAM and laptop** are connected to the **same hotspot**.
* If the stream doesn’t load, rerun `pythonip.py` and update the IP again in `app.py`.
* Do **not** close the terminal while the server is running.

---

**Developed by:** Nithyaganesh
**Repository:** [GitHub - swe_iot_12_11_2025](https://github.com/Nithyaganesh43/swe_iot_12_11_2025)

```
 
