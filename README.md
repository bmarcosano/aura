# AURA (Augmented User Real-time Assistant)
> **An Explainable "Magic Mirror" for 360° Postural Tracking and Multimodal Biofeedback**

---

## 1. Overview and Academic Objective
**AURA** is an interactive kiosk prototype ("Magic Mirror") designed for real-time postural correction and fitness coaching. Developed using modern software engineering and MLOps practices, the project bridges the gap between complex computer vision algorithms and real-time edge reactivity, ensuring a rigorous, transparent, and user-friendly framework.

## 2. Core Architecture & Technical Requirements
Unlike heavy, proprietary black-box systems, AURA is engineered following an **edge-friendly** philosophy:
* **Hardware Target:** Designed to run smoothly on standard Consumer Off-The-Shelf (COTS) hardware (**CPU-based** execution without requiring dedicated GPU accelerators).
* **Performance Targets:** Maintains $\ge 30$ FPS with an end-to-end processing latency of $<35$ ms per frame.
* **Core Stack:** Python $\ge 3.10$, OpenCV, MediaPipe Pose, FastDTW (Dynamic Time Warping), and NumPy.

## 3. Flexible Optical Setup (Single-Camera vs. 45° Mirror Mode)
To solve the multi-view occlusion problem without invasive multi-camera setups or heavy 3D mesh recovery networks, AURA supports a dual optical configuration:
* **Standard Mode:** Direct front-facing webcam view for baseline exercises.
* **Advanced "Magic Mirror" Mode (45° Side Mirror Setup):** Pairs a single front-facing webcam with a side mirror. A single video frame simultaneously captures both the direct (frontal) and reflected (lateral) views, leveraging the physics of light to achieve complete **360° biomechanical analysis** with minimal CPU overhead via dynamic Region-of-Interest (ROI) cropping.

## 4. User Experience (HCI & UX)
* **The Clean Mirror:** The display functions primarily as a clean mirror, allowing users to focus entirely on their own physical execution without distracting moving overlays.
* **The Vigilant Virtual Coach (NPC):** A calm, stationary interface manages session states (Standby $\rightarrow$ Exercise Selection $\rightarrow$ Execution $\rightarrow$ Rest Timer) and intervenes through multimodal feedback only when biomechanical deviations occur.

## 5. Algorithmic Pipeline
1. **Perception (`detector.py`):** Captures frames, handles optical ROI slicing, and extracts lightweight 3D skeletal joint coordinates via MediaPipe.
2. **Geometric Engine (`geometry.py`):** Computes instantaneous angular metrics (e.g., knee flexion, spine inclination) using robust NumPy vector operations.
3. **Temporal Alignment (`dtw_engine.py`):** Utilizes Dynamic Time Warping to compare user execution time-series against a reference sequence (*Gold Standard* from datasets like UI-PRMD) handling speed and rhythm variations.
4. **Explainable AI (`xai_feedback.py`):** Evaluates sub-millisecond geometric threshold rules to trigger localized visual overlays (soft red glowing aura on faulty joints) and targeted feedback.

## 6. Modular Software Structure
The codebase follows a clean separation of concerns:
```text
aura/
├── README.md
├── pyproject.toml
├── src/
│   └── aura/
│       ├── __init__.py
│       ├── detector.py      # Camera stream & ROI optical management
│       ├── geometry.py      # Vector math & angular computations
│       ├── dtw_engine.py    # Temporal alignment against Gold Standard
│       ├── xai_feedback.py  # Threshold rules & visual/audio feedback triggers
│       └── ui_kiosk.py      # State machine & mirror interface
└── tests/
    └── assets/              # Mock video files & reference datasets