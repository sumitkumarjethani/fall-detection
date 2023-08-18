# Fall Detection System

## Datasets

### Fall Dataset

[**Fall dataset**](https://falldataset.com/)

- **Code:** [Source](./src/fall_detection/datasets/falldataset.py)

- **Download fall dataset:**

```bash
python ./scripts/download_fall_dataset.py
-o ./data/falldataset \
--train \
--valid \
--test
```

- **Process fall dataset:**

```bash
python ./scripts/process_fall_dataset.py \
-i ./data/fall_dataset/train \
-o ./data/fall_dataset-processed/train
```

```bash
python ./scripts/process_fall_dataset.py \
-i ./data/fall_dataset/test \
-o ./data/fall_dataset-processed/test
```

### Convert Roboflow Yolo Dataset
```bash
python scripts/convert_yolo_dataset.py \
--input "../../data/custom_dataset_yolo" \
--output "../../data/custom_dataset"
```

### Generate Landmarks Dataset for Fall Detection

**Movenet:**
```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/personal-dataset" \
-o "../../data/movenet-personal-dataset-out" \
-f "../../data/movenet-personal-dataset-csv" \
--pose-model-name "movenet" \
--movenet-version "movenet_thunder"
```

**Mediapipe:**
```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/personal-dataset" \
-o "../../data/mediapipe-personal-dataset-out" \
-f "../../data/mediapipe-personal-dataset-csv" \
--pose-model-name "mediapipe"
```

**Yolo:**
```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/personal-dataset" \
-o "../../data/yolo-personal-dataset-out" \
-f "../../data/yolo-personal-dataset-csv" \
--pose-model-name "yolo"
--yolo-pose-model-path "../../models/yolov8n-pose.pt"
```

## Pose Models

### Image Pose Inference

**Movenet:**
```bash
python scripts/image_pose_inference.py \
-i "../../data/fall_sample.jpg" \
-o "../../data/fall_sample_out.jpg" \
--pose-model-name "movenet" \
--movenet-version "movenet_thunder"
```

**Mediapipe:**
```bash
python scripts/image_pose_inference.py \
-i "../../data/fall_sample.jpg" \
-o "../../data/fall_sample_out.jpg" \
--pose-model-name "mediapipe"
```

**Yolo:**
```bash
python scripts/image_pose_inference.py \
-i "../../data/fall_sample.jpg" \
-o "../../data/fall_sample_out.jpg" \
--pose-model-name "yolo"
--yolo-pose-model-path "../../models/yolov8n-pose.pt"
```

### Webcam Pose Inference

**Movenet:**
```bash
python scripts/webcam_pose_inference.py \
--pose-model-name "movenet" \
--movenet-version "movenet_thunder"
```

**Mediapipe:**
```bash
python scripts/webcam_pose_inference.py \
--pose-model-name "mediapipe"
```

**Yolo:**
```bash
python scripts/webcam_pose_inference.py \
--pose-model-name "yolo"
--yolo-pose-model-path "../../models/yolov8n-pose.pt"
```

## Train Pose Classifier

### Random Forest

**Movenet:**
```bash
python scripts/train_pose_classifier.py \
-i "../../data/movenet-personal-dataset-csv" \
-o "../../models" \
--model "rf" \
--model-name "movenet_rf_pose_classifier" \
--n-kps 17 \
--n-dim 3
```

**Mediapipe:**
```bash
python scripts/train_pose_classifier.py \
-i "../../data/mediapipe-personal-dataset-csv" \
-o "../../models" \
--model "rf" \
--model-name "mediapipe_rf_pose_classifier" \
--n-kps 33 \
--n-dim 3
```

**Yolo:**
```bash
python scripts/train_pose_classifier.py \
-i "../../data/yolo-personal-dataset-csv" \
-o "../../models" \
--model "rf" \
--model-name "yolo_rf_pose_classifier" \
--n-kps 17 \
--n-dim 3
```

### KNN

**Movenet:**
```bash
python scripts/train_pose_classifier.py \
-i "../../data/movenet-personal-dataset-csv" \
-o "../../models" \
--model "knn" \
--model-name "movenet_knn_pose_classifier" \
--n-kps 17 \
--n-dim 3 \
--n-neighbours 10
```

**Mediapipe:**
```bash
python scripts/train_pose_classifier.py \
-i "../../data/mediapipe-personal-dataset-csv" \
-o "../../models" \
--model "knn" \
--model-name "mediapipe_knn_pose_classifier" \
--n-kps 33 \
--n-dim 3 \
--n-neighbours 10
```

**Yolo:**
```bash
python scripts/train_pose_classifier.py \
-i "../../data/yolo-personal-dataset-csv" \
-o "../../models" \
--model "knn" \
--model-name "yolo_knn_pose_classifier" \
--n-kps 17 \
--n-dim 3 \
--n-neighbours 10
```

## Evaluate Pose Classifier

### Random Forest

**Movenet:**
```bash
python scripts/evaluate_pose_classifier.py \
-i "../../data/movenet-personal-dataset-csv/test" \
-o "../../metrics/" \
-f "movenet_rf_personal_dataset" \
--pose-classifier "../../models/movenet_rf_pose_classifier.pkl"
```

**Mediapipe:**
```bash
python scripts/evaluate_pose_classifier.py \
-i "../../data/mediapipe-personal-dataset-csv/test" \
-o "../../metrics/" \
-f "mediapipe_rf_personal_dataset" \
--pose-classifier "../../models/mediapipe_rf_pose_classifier.pkl"
```

**Yolo:**
```bash
python scripts/evaluate_pose_classifier.py \
-i "../../data/yolo-personal-dataset-csv/test" \
-o "../../metrics/" \
-f "yolo_rf_personal_dataset" \
--pose-classifier "../../models/yolo_rf_pose_classifier.pkl"
```

### KNN

**Movenet:**
```bash
python scripts/evaluate_pose_classifier.py \
-i "../../data/movenet-personal-dataset-csv/test" \
-o "../../metrics/" \
-f "movenet_knn_personal_dataset" \
--pose-classifier "../../models/movenet_knn_pose_classifier.pkl"
```

**Mediapipe:**
```bash
python scripts/evaluate_pose_classifier.py \
-i "../../data/mediapipe-personal-dataset-csv/test" \
-o "../../metrics/" \
-f "mediapipe_knn_personal_dataset" \
--pose-classifier "../../models/mediapipe_knn_pose_classifier.pkl"
```

**Yolo:**
```bash
python scripts/evaluate_pose_classifier.py \
-i "../../data/yolo-personal-dataset-csv/test" \
-o "../../metrics/" \
-f "yolo_knn_personal_dataset" \
--pose-classifier "../../models/yolo_knn_pose_classifier.pkl"
```

## Video Fall Inference 

**Movenet:**
```bash
python scripts/video_fall_inference.py \
-i "../../data/videos/uri.mp4" \
-o "../../data/videos/movenet_uri_out.mp4" \
--pose-model-name "movenet" \
--movenet-version "movenet_thunder" \
--pose-classifier "../../models/movenet_rf_pose_classifier.pkl" 
```

**Mediapipe:**
```bash
python scripts/video_fall_inference.py \
-i "../../data/videos/uri.mp4" \
-o "../../data/videos/mediapipe_uri_out.mp4" \
--pose-model-name "mediapipe" \
--pose-classifier "../../models/mediapipe_rf_pose_classifier.pkl" 
```

**Yolo:**
```bash
python scripts/video_fall_inference.py \
-i "../../data/videos/uri.mp4" \
-o "../../data/videos/yolo_uri_out.mp4" \
--pose-model-name "yolo" \
--yolo-pose-model-path "../../models/yolov8n-pose.pt" \
--pose-classifier "../../models/yolo_rf_pose_classifier.pkl" 
```

## WebCam Fall Inference

**Movenet:**
```bash
python scripts/webcam_fall_inference.py \
--pose-model-name "movenet" \
--movenet-version "movenet_thunder" \
--pose-classifier "../../models/movenet_rf_pose_classifier.pkl" 
```

**Mediapipe:**
```bash
python scripts/webcam_fall_inference.py \
--pose-model-name "mediapipe" \
--pose-classifier "../../models/mediapipe_rf_pose_classifier.pkl" 
```

**Yolo:**
```bash
python scripts/webcam_fall_inference.py \
--pose-model-name "yolo" \
--yolo-pose-model-path "../../models/yolov8n-pose.pt" \
--pose-classifier "../../models/yolo_rf_pose_classifier.pkl" 
```

## Fall Detector Pipeline

**Movenet:**
```bash
python scripts/video_fall_pipeline.py \
-i "../../data/videos/uri.mp4" \
-o "../../data/videos/movenet_uri_pipeline_out.mp4" \
--pose-model-name "movenet" \
--movenet-version "movenet_thunder" \
--yolo-object-model-path "../../models/yolov8n.pt" \
--pose-classifier "../../models/movenet_rf_pose_classifier.pkl" 
```

**Mediapipe:**
```bash
python scripts/video_fall_pipeline.py \
-i "../../data/videos/uri.mp4" \
-o "../../data/videos/mediapipe_uri_pipeline_out.mp4" \
--pose-model-name "mediapipe" \
--yolo-object-model-path "../../models/yolov8n.pt" \
--pose-classifier "../../models/mediapipe_rf_pose_classifier.pkl" 
```

**Yolo:**
```bash
python scripts/video_fall_pipeline.py \
-i "../../data/videos/uri.mp4" \
-o "../../data/videos/yolo_uri_pipeline_out.mp4" \
--pose-model-name "yolo" \
--yolo-pose-model-path "../../models/yolov8n-pose.pt" \
--yolo-object-model-path "../../models/yolov8n.pt" \
--pose-classifier "../../models/yolo_rf_pose_classifier.pkl" 
```

## Steps to Evaluate new Dataset / Model

1. Convert dataset to folder dataset (Optional)
2. Generate landmarks dataset
3. Train pose classifier
4. Evaluate Pose classifier
5. Run pipeline inference on example image/video/webcam