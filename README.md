<h1 align="center">
  <img
    src="https://readme-typing-svg.herokuapp.com?size=26&duration=2500&pause=700&color=9400D3&center=true&vCenter=true&width=750&lines=📡+R.A.D.A.R;Real-time+Automatic+Detection+And+Recognition;End-to-End+Vehicle+Tracking+Pipeline;Automated+Number+Plate+Recognition;Powered+by+YOLOv8+and+EasyOCR;✨+Explore+the+code;✨+Watch+the+demo"
    alt="Animated R.A.D.A.R welcome"
  />
</h1>

<p align="center">
    <img src="assets/Logo.jpg" alt="R.A.D.A.R logo" width="420">
</p>

<p align="center">
    <img src="assets/output_gif.gif" alt="R.A.D.A.R demo animation" width="860">
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/PyTorch-Native-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff?logo=ultralytics&logoColor=black" alt="YOLOv8">
    <img src="https://img.shields.io/badge/License-CC%20BY%204.0-success" alt="License">
</p>

<p align="center">
    <em>Real-time Automatic Detection And Recognition for vehicle tracking and automatic number plate recognition.</em>
</p>

---

## 📖 Overview

**R.A.D.A.R** is an end-to-end Automatic Number Plate Recognition (ANPR) pipeline built for noisy, real-world traffic footage. It combines YOLOv8-based vehicle tracking, a custom-trained license plate detector, OCR-based text extraction, and mathematical interpolation logic to keep detections stable across dropped frames.

> **Academic Context:** This project was developed as a term project for the B.Sc. (Honours) Data Science and Artificial Intelligence program at the **Indian Institute of Technology (IIT) Guwahati**.

---

## ✨ Highlights

*   🎯 **Dual-Model Detection:** Utilizes `yolov8n.pt` for robust vehicle tracking and a custom-trained `yolo26n.pt` for precise license plate localization.
*   📈 **Mathematical Interpolation:** Employs SciPy-based 1-D interpolation to smooth missing bounding-box coordinates and completely eliminate visual flicker.
*   🔍 **Optimized OCR Workflow:** Designed for practical traffic scenes with dynamic preprocessing (grayscale & inverse binary thresholding) and confidence filtering.
*   📐 **Spatial Validation:** Hardcoded logic ensures that plates are only accepted and processed if they are detected strictly *inside* a tracked vehicle's bounding box.
*   📊 **Interactive UI:** Includes Streamlit dashboard support for a seamless and interactive user workflow.

---

## ⚙️ How It Works

1.  **Spot:** A vehicle is detected and continuously tracked using YOLOv8.
2.  **Localize:** The custom plate detector scans for license plates exclusively inside the cropped vehicle frame.
3.  **Process:** The isolated plate crop is preprocessed to maximize contrast and sent to the EasyOCR engine.
4.  **Filter:** Valid alphanumeric text is filtered by confidence scores and written to structured output.
5.  **Stabilize:** Linear interpolation mathematically fills any detection gaps to keep the final visual track buttery smooth.

---

## 📂 Project Structure

```text
R.A.D.A.R/
├── assets/
│   ├── Logo.jpg
│   └── output_gif.gif
├── License Plate Recognition.v11i.yolov8/
├── License_Plate_10k/
├── src/
│   ├── anpr_utils.py
│   ├── app.py
│   ├── main.py
│   ├── visualization.py
│   └── preprocessing/
├── yolo26n.pt
├── yolov8n.pt
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Option 1: Using `uv` (Recommended)

Clone the repository and set up your environment using `uv` for blazing fast package installation:

```bash
git clone [https://github.com/YourUsername/RADAR.git](https://github.com/YourUsername/RADAR.git)
cd RADAR
uv venv
```

Activate the environment:
```bash
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Option 2: Using `pip`

```bash
git clone [https://github.com/YourUsername/RADAR.git](https://github.com/YourUsername/RADAR.git)
cd RADAR
python -m venv .venv

# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 💻 Usage

**Run the Core ANPR Pipeline:**  
Process the sample video and generate an annotated output:
```bash
python src/main.py sample_30fps.mp4
```
*(To process your own video, simply replace `sample_30fps.mp4` with the path to your file).*

**Launch the Interactive Dashboard:**  
```bash
streamlit run src/app.py
```

---

## 📦 Dependencies

The core Python packages required to run this project are listed in `requirements.txt` and include:

*   **OpenCV** (Image and video rendering)
*   **EasyOCR** (PyTorch-native text extraction)
*   **SciPy** (Bounding box interpolation)
*   **PyTorch** (Deep learning backend)
*   **Ultralytics YOLO** (Object detection engine)
*   **Streamlit** (Dashboard UI)
*   **Plotly** (Data visualization)
*   **LangChain** (Integrations used by the dashboard agent)

---

## 📊 Dataset and Credits

The custom license plate detector (`yolo26n.pt`) was trained on the **[License Plate Recognition Computer Vision Model](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11)** dataset, hosted on [Roboflow Universe](https://universe.roboflow.com/).

**Special thanks to:**
*   **Roboflow** and the open-source computer vision community for dataset curation.
*   **Ultralytics** for the incredible YOLOv8 architecture.
*   **JaidedAI** for the robust EasyOCR engine.

---

## 👨‍💻 Author

**Onkar Upadhyay**  
*B.Sc. (Honours), Data Science and Artificial Intelligence*  
Indian Institute of Technology (IIT) Guwahati

---

## 📄 License

This repository includes model weights and dataset-derived assets that may have separate terms. See the `LICENSE-WEIGHTS` file for weight-specific licensing details. The primary source code is distributed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.