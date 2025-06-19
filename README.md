# 🚨 Real-Time Violence Detection in Surveillance Streams

This project presents a real-time, deep learning-based violence detection system designed for deployment in urban surveillance infrastructure. Leveraging a fine-tuned DenseNet121 CNN and multithreaded processing, the system processes multiple RTSP camera feeds concurrently at **30 FPS**, provides live dashboard visualization, and stores metadata securely in the cloud.

---

## 🧠 Overview

Modern public safety systems require intelligent, automated video analysis to detect violent activity instantly. This project bridges academic machine learning research with real-world application by integrating:

- 🧠 Deep Learning with **DenseNet121**
- 🧵 Multithreaded RTSP stream processing
- 📊 Real-time Flask dashboard
- ☁️ Cloud backup using **Dropbox API**
- 🗂️ Metadata logging with **MongoDB**
- 🖥️ **Packaged executable** for easy deployment (via PyInstaller)

---

## ⚙️ System Features

| Feature                        | Description                                                  |
|-------------------------------|--------------------------------------------------------------|
| 🎯 Model Accuracy              | 92% (F1-score: 0.91 on UCF-Crime subset)                    |
| ⚡ Real-time Inference         | Achieves **30 FPS** on a T4 GPU                              |
| 📡 RTSP Stream Support         | Handles multiple live surveillance streams                   |
| 🔐 Privacy-Preserving         | Anonymized frames (blurred faces, no personal identifiers)   |
| 💾 Cloud Storage              | Auto-upload flagged frames to Dropbox                        |
| 📬 Alert System                | Email alerts via Flask-Mail                                  |
| 📈 Visualization Dashboard     | Live camera feeds + frame gallery using Flask                |
| 💼 Deployment Ready            | One-click `.exe` built via PyInstaller                       |

---

## 📦 Project Structure

Real-Time-Violence-Detection/
├── model/ # Trained DenseNet121 (.h5)
├── detection/ # Stream handlers, threading
├── app.py # Flask web server
├── utils/ # Helper scripts
├── templates/ # HTML templates (Flask)
├── static/ # Optional frontend assets
├── config.py # Dropbox, email, stream configs
├── requirements.txt # Dependencies
└── README.md

---

## 🧪 Quickstart (Python 3.7+)

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
2️⃣ Configure the System
Edit config.py with:

Your RTSP stream URLs

Dropbox access token

Email credentials for alerts

3️⃣ Run the App
bash
Copy
Edit
python app.py
Then open your browser and go to:
http://localhost:5000
🧠 Model Summary
Base: DenseNet121 (ImageNet pretrained)

Input Size: 224 × 224 × 3

Augmentations: Rotation, zoom, shear, shift, horizontal flip

Fine-tuning: Last 30 layers unfrozen

Optimization: Adam, learning rate 1e-5, early stopping enabled

📊 Performance
Metric	Value
Accuracy	92%
F1 Score	0.91
Recall (Violence)	0.80
FPS (T4 GPU)	30 FPS
RTSP Streams	4+ concurrently supported

📁 Dataset
Uses a curated, anonymized subset of the UCF-Crime Dataset with 1000 frames (60% non-violent, 40% violent). Data augmentation mitigates class imbalance and overfitting.

🌐 Live Demo & Code Access
🔗 GitHub Repository:
https://github.com/vavi39/Real-Time-Violence-Detection-in-Surveillance-Streams

🛡️ Ethical Usage
This system is intended for research and public safety applications only. It anonymizes faces before detection and should be deployed in accordance with local privacy and surveillance regulations.

📚 Citation
If you use this work, please cite:


Verma, Avi. "Real-Time Violence Detection in Surveillance Streams." Delhi Technological University, 2025.
👨‍💻 Author
Avi Verma
Department of Applied Mathematics
Delhi Technological University
📧 aviverma_mc22a13_59@dtu.ac.in(UNIVERSITY)
📧vavi3984@gmail.com(PERSONAL)




---

### ✅ To Use This:
1. Copy this into a new `README.md` file in your repo.
2. Add a screenshot of the dashboard (optional) under `assets/dashboard.png` and update the image path.
3. Push to GitHub:
```bash
git add README.md
git commit -m "Added detailed README with 30 FPS update"
git push
