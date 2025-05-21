# 🧘‍♂️ Posture Correction & Workout Plan Generator

This project detects human posture, identifies postural imbalances, and generates customized workout plans to correct them. It leverages **MoveNet**, a fast and accurate pose estimation model trained extensively on the **COCOPose** dataset, to analyze posture from webcam input or images.

---

## 📌 Features

- 📷 **Real-Time Posture Detection** using webcam or image input
- 🧍 **Posture Classification**: Identifies common deviations like:
  - Forward Head Posture
  - Rounded Shoulders
  - Anterior Pelvic Tilt
- 🏋️ **Workout Plan Generation**: Tailored exercise routines based on detected postural issues

---

## 🧠 Model & Dataset

### 🔍 MoveNet (by TensorFlow Hub)
- Lightning-fast, accurate pose estimation
- Supports **SinglePose** and **MultiPose** modes
- Well-suited for real-time applications

### 🗂️ Dataset: [COCOPose](https://cocodataset.org/)
- Derived from the COCO dataset with rich keypoint annotations
- Contains 100K+ human pose images
- MoveNet was pre-trained and fine-tuned on this dataset

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/posture-correction.git
cd posture-correction
pip install -r requirements.txt
````

---

## 🛠️ Usage

### 1. Analyze Posture via Webcam

```bash
python analyze_posture.py --input webcam
```

* Captures live video and performs pose estimation
* Outputs detected keypoints and identifies posture issues

### 2. Generate a Workout Plan

```bash
python generate_workout.py --analysis posture_results.json
```

* Inputs the JSON output from posture analysis
* Returns a customized workout plan to correct imbalances

---

## 📁 Project Structure

```
posture-correction/
│
├── data/                     # Sample inputs (images/videos)
├── models/                   # Posture analysis and classification logic
├── utils/                    # Helper scripts (e.g., keypoint parser)
├── workouts/                 # Exercise metadata and plans
├── analyze_posture.py        # Main posture detection script
├── generate_workout.py       # Workout plan generator
├── requirements.txt
└── README.md
```

---

## 🧪 Example

**Input**: Webcam feed of user standing in side profile
**Detected Issues**:

* Forward Head Posture
* Rounded Shoulders

**Workout Plan**:

* ✅ Chin Tucks – 3 sets of 10 reps
* ✅ Wall Angels – 3 sets of 15 reps
* ✅ Thoracic Extensions – 3 sets of 12 reps

---

## 📈 Future Enhancements

* [ ] Fully automated real-time webcam posture tracking
* [ ] Voice-guided exercise feedback system
* [ ] Progress tracker and reporting dashboard

---

## 🧑‍💻 Author

**M.NagaHarshith Rao**
GitHub: [@MNagaHarshithRao](https://github.com/MNagaHarshithRao)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).


