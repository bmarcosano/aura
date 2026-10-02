"""create_milestones.py: Automates the creation of GitHub milestones for AURA."""

import subprocess

milestones = [
    {
        "title": "Milestone 1: Core Perception & Geometry",
        "description": "Project initialization, geometry module, and MediaPipe pose detector integration.",
        "state": "closed",
    },
    {
        "title": "Milestone 2: Posture Analysis Engine",
        "description": "Implement posture analysis metrics, unit tests, and live webcam feed script.",
        "state": "open",
    },
    {
        "title": "Milestone 3: Biofeedback & UI Visualizer",
        "description": "Real-time visual feedback overlay on OpenCV frames and alert triggers.",
        "state": "open",
    },
    {
        "title": "Milestone 4: Temporal Tracking & DTW Assessment",
        "description": "Integrate fastdtw for sequence comparison and session reporting.",
        "state": "open",
    },
]

repo = "bmarcosano/aura"

print("🚀 Inizializzazione creazione milestone su GitHub...")

for m in milestones:
    cmd = [
        "gh",
        "api",
        f"repos/{repo}/milestones",
        "-f",
        f"title={m['title']}",
        "-f",
        f"description={m['description']}",
        "-f",
        f"state={m['state']}",
    ]

    print(f"Creazione in corso: '{m['title']}' (Stato: {m['state']})...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  [OK] Creato con successo.")
    else:
        print(f"  [ATTENZIONE/ERRORE]: {result.stderr.strip()}")

print("\n✨ Tutte le milestone sono state elaborate!")
