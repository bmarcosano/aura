"""create_issues.py: Automates the creation of GitHub issues mapped to milestones for AURA."""

import json
import subprocess

repo = "bmarcosano/aura"

print("🔍 Recupero delle milestone esistenti da GitHub...")
result = subprocess.run(
    ["gh", "api", f"repos/{repo}/milestones?state=all"], capture_output=True, text=True
)
if result.returncode != 0:
    print(f"❌ Errore nel recupero delle milestone: {result.stderr}")
    exit(1)

milestones_data = json.loads(result.stdout)
milestone_map = {m["title"]: m["number"] for m in milestones_data}
print(f"Trovate {len(milestone_map)} milestone attive.")

# Definizione delle issue per ciascuna milestone
issues = [
    # Milestone 1: Core Perception & Geometry (Chiusa)
    {
        "title": "Project initialization, pyproject.toml, and basic structure",
        "body": "Initialize repository structure, set up pyproject.toml, and configure dependency management.",
        "milestone": "Milestone 1: Core Perception & Geometry",
        "state": "closed",
    },
    {
        "title": "Geometric calculation module (geometry.py) and unit tests",
        "body": "Implement core geometric helper functions for vectors, angles, and distances with pytest unit tests.",
        "milestone": "Milestone 1: Core Perception & Geometry",
        "state": "closed",
    },
    {
        "title": "MediaPipe Pose integration (detector.py) and robust test suite",
        "body": "Create PoseDetector class to handle MediaPipe landmark extraction and test with synthetic frames.",
        "milestone": "Milestone 1: Core Perception & Geometry",
        "state": "closed",
    },
    # Milestone 2: Posture Analysis Engine (Aperta)
    {
        "title": "Implement PostureAnalyzer for posture metrics",
        "body": "Implement PostureAnalyzer to calculate shoulder tilt, hip alignment, and head forward posture from landmarks.",
        "milestone": "Milestone 2: Posture Analysis Engine",
        "state": "open",
    },
    {
        "title": "Write unit tests for postural metrics",
        "body": "Add comprehensive unit tests covering edge cases and expected angles in PostureAnalyzer.",
        "milestone": "Milestone 2: Posture Analysis Engine",
        "state": "open",
    },
    {
        "title": "Create live webcam execution script (scripts/run_webcam.py)",
        "body": "Build a real-time script integrating PoseDetector and OpenCV to visualize landmarks on live webcam stream.",
        "milestone": "Milestone 2: Posture Analysis Engine",
        "state": "open",
    },
    # Milestone 3: Biofeedback & UI Visualizer (Aperta)
    {
        "title": "Real-time visual feedback overlay on OpenCV frames",
        "body": "Implement color-coded visual cues and feedback overlays on the skeleton based on posture thresholds.",
        "milestone": "Milestone 3: Biofeedback & UI Visualizer",
        "state": "open",
    },
    {
        "title": "Audio/visual alert triggers for poor posture",
        "body": "Trigger notifications or sound cues when posture deviation exceeds configured thresholds over time.",
        "milestone": "Milestone 3: Biofeedback & UI Visualizer",
        "state": "open",
    },
    # Milestone 4: Temporal Tracking & DTW Assessment (Aperta)
    {
        "title": "Integrate fastdtw for sequence comparison",
        "body": "Use fastdtw to compare movement sequences against reference templates for postural dynamic feedback.",
        "milestone": "Milestone 4: Temporal Tracking & DTW Assessment",
        "state": "open",
    },
    {
        "title": "Session reporting and summary metrics storage",
        "body": "Save session logs and generate summary reports of posture performance over time.",
        "milestone": "Milestone 4: Temporal Tracking & DTW Assessment",
        "state": "open",
    },
]

print("\n🚀 Creazione delle issue su GitHub in corso...")

for issue in issues:
    m_title = issue["milestone"]
    if m_title not in milestone_map:
        print(
            f"  [ATTENZIONE] Milestone '{m_title}' non trovata per l'issue '{issue['title']}'. Saltata."
        )
        continue

    # Creazione dell'issue tramite GitHub CLI
    cmd = [
        "gh",
        "issue",
        "create",
        "--title",
        issue["title"],
        "--body",
        issue["body"],
        "--milestone",
        m_title,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        issue_url = result.stdout.strip()
        print(f"  [OK] Creata: '{issue['title']}' -> {m_title}")

        # Se l'issue appartiene alla Milestone 1 (già completata), la chiudiamo subito
        if issue["state"] == "closed":
            # Estrae il numero dell'issue dall'URL restituito (es. .../issues/1)
            issue_number = issue_url.split("/")[-1]
            close_cmd = [
                "gh",
                "issue",
                "close",
                issue_number,
                "--comment",
                "Completed during Milestone 1 phase.",
            ]
            close_res = subprocess.run(close_cmd, capture_output=True, text=True)
            if close_res.returncode == 0:
                print(f"       ↳ Chiusa con successo (Issue #{issue_number})")
    else:
        print(
            f"  [ERRORE] Impossibile creare '{issue['title']}': {result.stderr.strip()}"
        )

print("\n✨ Tutte le issue sono state generate e assegnate alle rispettive milestone!")
