# Webcam All-Faces Auth with Live Anti-Spoofing 🔐👁️

Real-time **multi-face recognition** from your webcam with **advanced spoof detection** (photos, videos, masks) — all running locally, securely, and fast.

https://github.com/Harshbeast/college-security-system/blob/main/live%20demo.mp4  
*(Live demo: multiple people detected, spoof attempts rejected in real-time)*

## Features ✨

- **Recognizes multiple faces simultaneously** in real time
- **Encrypted known faces database** using Fernet (AES)
- **Strong anti-spoofing** using Silent-Face-Anti-Spoofing model ensemble
- Rejects photos, replay videos, 3D masks, and silicone faces
- Threaded processing → smooth performance even with multiple faces
- Configurable tolerance, speed/accuracy trade-off
- Clean visual feedback: color-coded boxes and status

| Status             | Color       | Meaning                     |
|--------------------|-------------|-----------------------------|
| Authorized         | Green       | Known + Real person         |
| Denied (Spoof)     | Red         | Known but fake (photo/video)|
| Denied (Unknown)   | Orange      | Real but not in database    |
| Denied (Both)      | Orange/Red  | Unknown + fake              |

## Quick Start 🚀

### 1. Prerequisites

```bash
Python 3.8+
OpenCV
dlib (with CUDA recommended)
face_recognition
cryptography
