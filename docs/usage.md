# AURA User Guide: Biomechanical Coaching & Evaluation Workflow

This guide details the step-by-step workflow for operating **AURA**, running multi-view pose tracking, recording reference archetypes (*Golden Standards*), and evaluating user executions in real-time.

---

## 1. Project Directory Structure

AURA automatically manages exercise-specific data persistence using a structured directory layout under the hood:
* **`golden_executions/<exercise_name>/golden.json`**: Stores the single-repetition reference archetype recorded from a trainer or ideal video.
* **`scripts/main.py`**: The core execution entry point.

---

## 2. On-Screen Keyboard Controls

While running AURA, you can manage the recording states and file operations directly using your keyboard:

| Key | Action / Description |
| :---: | :---|
| **`G`** | Start recording a single-repetition **Golden Standard** (Trainer archetype). |
| **`U`** | Start evaluating a **User Execution** (streams real-time pacing and form checks). |
| **`S`** | **Stop** the active recording and compute final DTW evaluation metrics. |
| **`W`** | **Write/Save** the recorded Golden Standard to disk (`golden_executions/<exercise>/golden.json`). |
| **`L`** | **Load** an existing Golden Standard JSON file into memory (`Mem: READY`). |
| **`V`** | Print current memory status and file verification report to the terminal. |
| **`D`** | **Delete** the saved Golden Standard JSON file from disk. |
| **`Q`** | Terminate and quit AURA safely. |

---

## 3. Step-by-Step Operating Workflow

To obtain a rigorous biomechanical judgment, follow this strict two-phase sequence:

### Phase A: Creating the Golden Standard (The Archetype)
*Note: The Golden Standard must capture **only a single repetition** (e.g., one clean descent and ascent), not the entire multi-rep workout.*

1. Launch AURA with your reference video specifying the target exercise:
   ```bash
   python scripts/main.py --source input_videos/gold_squat.mp4 --exercise squats
   ```
2. Watch the video play. The moment the trainer begins the target repetition, press **`[G]`**.
3. The moment that single repetition concludes (returning to the starting position), press **`[S]`** (Stop).
4. Press **`[W]`** to save the reference trajectory into `golden_executions/squats/golden.json`.
5. Press **`[Q]`** to exit.

### Phase B: Evaluating the User Execution
1. Launch AURA using the user's video (or local webcam `--source 0`):
   ```bash
   python scripts/main.py --source input_videos/user_squat.mp4 --exercise squats
   ```
2. Press **`[L]`** to load the pre-saved Golden Standard. Verify that the on-screen panel displays **`Mem: READY`** in green.
3. When the user begins performing the exercise in the video stream, press **`[U]`** (User) to start real-time tracking. 
   * *AURA will display live pacing hints (`PACE: OPTIMAL`, `TOO FAST`, `TOO SLOW`) and postural HUD alerts.*
4. When the user finishes the sequence, press **`[S]`** (Stop).
5. Review the final **DTW Score (0-100)** and qualitative feedback (*EXCELLENT, GOOD, POOR*) rendered directly on the HUD and terminal.
6. Press **`[Q]`** to quit.