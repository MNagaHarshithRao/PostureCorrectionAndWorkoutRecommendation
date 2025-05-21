
# 🧘‍♂️ Posture Correction and Workout Plan Generator

This project is focused on **detecting human posture**, identifying **postural imbalances**, and generating a **customized workout plan** to improve posture. Using pose estimation via **MoveNet**, trained heavily on the **COCOPose** dataset, our system can assess posture from images or live video and recommend corrective exercises accordingly.

---

## 📌 Features

* 📷 **Posture Detection**: Uses MoveNet to analyze 2D skeletal keypoints from images or video frames.
* 🧍 **Posture Classification**: Detects common postural deviations like forward head, rounded shoulders, anterior pelvic tilt, etc.
* 🏋️ **Workout Plan Generation**: Suggests personalized exercises based on detected issues.
* 📊 **Progress Tracking** *(Optional)*: Logs posture improvements over time with visual keypoint comparison.

---

## 🧠 Model and Dataset

### Pose Estimation Model: [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet)

* A fast and accurate model for human pose estimation.
* Supports both **SinglePose** (for one person) and **MultiPose** variants.

### Dataset: [COCOPose](https://cocodataset.org/)

* A curated version of the COCO dataset focused on keypoint annotations.
* Contains over 100k labeled images of human poses.
* MoveNet was fine-tuned on COCOPose for superior skeletal landmark accuracy.

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/posture-correction.git
cd posture-correction
pip install -r requirements.txt
```

---

## 🛠️ Usage

### 1. Run Posture Analysis

```bash
python analyze_posture.py --input your posture through the webcam detection 
```

* Outputs keypoints and highlights posture deviations.

### 2. Generate Workout Plan

```bash
python generate_workout.py --analysis posture_results.json
```

* Returns a structured workout plan targeting weak or tight muscle groups.

---

## 📁 Project Structure

```
posture-correction/
│
├── data/                     # Sample input images and videos
├── models/                   # Pose detection and classification logic
├── utils/                    # Helper scripts (e.g., keypoint parser, angle calculator)
├── workouts/                 # Library of exercises with metadata
├── analyze_posture.py        # Main posture detection script
├── generate_workout.py       # Personalized plan generator
├── requirements.txt
└── README.md
```

---

## 🧪 Example

1. **Input**: webcam detection of the persons posture 
2. **Detected Issue**: Forward Head Posture, Rounded Shoulders.
3. **Generated Plan**:

   * Chin tucks – 3x10 reps
   * Wall angels – 3x15 reps
   * Thoracic extensions – 3x12 reps

---

## 📈 Future Improvements

* Integrate real-time webcam support.
* Add AI-based feedback during workouts.
* Expand to full-body biomechanics analysis.

---

