"""fix_milestone1.py: Temporarily reopens Milestone 1 to create and close its historical issues."""

import subprocess

repo = "bmarcosano/aura"

print("🔄 Riapertura temporanea della Milestone 1...")
# Trova l'ID della Milestone 1
import json
res = subprocess.run(["gh", "api", f"repos/{repo}/milestones?state=all"], capture_output=True, text=True)
milestones = json.loads(res.stdout)
m1 = next((m for m in milestones if "Milestone 1" in m["title"]), None)

if not m1:
    print("❌ Milestone 1 non trovata!")
    exit(1)

m1_number = m1["number"]

# 1. Riapri la Milestone 1
subprocess.run(["gh", "api", f"repos/{repo}/milestones/{m1_number}", "-X", "PATCH", "-f", "state=open"], check=True)

# 2. Crea le 3 issue della Milestone 1
m1_issues = [
    {
        "title": "Project initialization, pyproject.toml, and basic structure",
        "body": "Initialize repository structure, set up pyproject.toml, and configure dependency management."
    },
    {
        "title": "Geometric calculation module (geometry.py) and unit tests",
        "body": "Implement core geometric helper functions for vectors, angles, and distances with pytest unit tests."
    },
    {
        "title": "MediaPipe Pose integration (detector.py) and robust test suite",
        "body": "Create PoseDetector class to handle MediaPipe landmark extraction and test with synthetic frames."
    }
]

for issue in m1_issues:
    print(f"Creazione issue M1: '{issue['title']}'...")
    cmd = [
        "gh", "issue", "create",
        "--title", issue["title"],
        "--body", issue["body"],
        "--milestone", m1["title"]
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        issue_url = result.stdout.strip()
        issue_number = issue_url.split("/")[-1]
        print(f"  [OK] Creata e chiusura in corso (Issue #{issue_number})...")
        subprocess.run(["gh", "issue", "close", issue_number, "--comment", "Completed during Milestone 1 phase."], check=True)
    else:
        print(f"  [ERRORE]: {result.stderr.strip()}")

# 3. Richiudi la Milestone 1
print("🔒 Chiusura definitiva della Milestone 1...")
subprocess.run(["gh", "api", f"repos/{repo}/milestones/{m1_number}", "-X", "PATCH", "-f", "state=closed"], check=True)

print("✨ Fatto! Tutte le issue di tutte le milestone sono ora perfettamente allineate su GitHub.")