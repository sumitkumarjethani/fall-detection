# Realtime fall detection


## Datasets

## [Falldataset](https://falldataset.com/)

#### Requirements:
- wget
- tar

### Code

- [Source](./src/datasets/falldataset.py)

### Descarga del falldataset

- [Script](./src/scripts/download_falldataset.py)

```bash
python src/scripts/download_falldataset.py -o ./data
```

### Procesado del dataset

- [Script](./src/scripts/process_falldataset.py)

```bash
python src/scripts/process_falldataset.py -i ./data
```

## Convert Yolo Dataset

```bash
python scripts/convert_yolo_dataset.py --input "../../data/custom_dataset_yolo" --output "../../data/custom_dataset"
```

## Pose Models

### Movenet image pose inference
```bash
python scripts/image_pose_inference.py \
-i "../../data/fall-sample.png" \
-o "../../data/fall-sample-output.jpg" \
-m "movenet" \
-mv  "movenet_thunder"
```

### Mediapipe image pose inference
```bash
python scripts/image_pose_inference.py \
-i "../../data/fall-sample.png" \
-o "../../data/fall-sample-output.jpg" \
-m "mediapipe"
```

### Yolo Image pose inference
```bash
python scripts/image_pose_inference.py \
-i "../../data/fall-sample.png" \
-o "../../data/fall-sample-output.jpg" \
-m "yolo" \
-p "../../models/yolov7-w6-pose.pt"
```

### Webcam Movenet Pose Inference
```bash
python scripts/webcam_pose_inference.py  \
-m "movenet"
```

### Webcam Mediapipe Pose Inference
```bash
python scripts/webcam_pose_inference.py  \
-m "mediapipe"
```

### Webcam Yolo Pose Inference
```bash
python scripts/webcam_pose_inference.py  \
-m "yolo" \
-p "../../models/yolov7-w6-pose.pt"
```

## Generate Landmarks Dataset for Fall Detection

#### Small dataset with mediapipe
```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/test_dataset" \
-o "../../data/test_dataset_out" \
-f "../../data/test_dataset_csv" \
-m "mediapipe"
```

#### Small dataset with movenet
```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/test_dataset" \
-o "../../data/movenet_test_dataset_out" \
-f "../../data/movenet_test_dataset_csv" \
-m "movenet"
```

#### Small dataset with yolo
```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/test_dataset" \
-o "../../data/yolo_test_dataset_out" \
-f "../../data/yolo_test_dataset_csv" \
-m "yolo"
```

#### Process full dataset
```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/samples" \
-o "../../data/mediapipe_samples_out" \
-f "../../data/mediapipe_samples_csv_out" \
-m "mediapipe" \
--max-samples 6000
```

#### process full dataset
```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/samples" \
-o "../../data/movenet_samples_out" \
-f "../../data/movenet_samples_csv_out" \
-m "movenet" \
--max-samples 6000
```

#### process full dataset
```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/samples" \
-o "../../data/yolo_samples_out" \
-f "../../data/yolo_samples_csv_out" \
-m "yolo" \
--max-samples 6000
```

## Train Pose Classification

- test train knn on small dataset
```bash
python scripts/train_knn_pose_classifier.py \
-i "../../data/test_dataset_csv" \
-m "../../models/test_pose_classification_model.pkl" \
--n-kps 33 \
--n-dim 3 \
--n-neighbours 10
```

-  train knn 
```bash
python scripts/train_knn_pose_classifier.py \
-i "../../data/mediapipe_samples_csv_out" \
-m "../../models/mediapipe_knn_model.pkl" \
--n-kps 33 \
--n-dim 3 \
--n-neighbours 10
```

```bash
python scripts/train_knn_pose_classifier.py \
-i "../../data/movenet_samples_csv_out" \
-m "../../models/movenet_knn_model.pkl" \
--n-kps 17 \
--n-dim 3 \
--n-neighbours 10
```

```bash
python scripts/train_knn_pose_classifier.py \
-i "../../data/yolo_samples_csv_out" \
-m "../../models/yolo_knn_model.pkl" \
--n-kps 17 \
--n-dim 3 \
--n-neighbours 10
```

- train estimator
```bash
python scripts/train_estimator_pose_classifier.py \
-i "../../data/mediapipe_samples_csv_out" \
-m "../../models/mediapipe_estimator_model.pkl" \
--n-kps 33 \
--n-dim 3 \
```

```bash
python scripts/train_estimator_pose_classifier.py \
-i "../../data/movenet_samples_csv_out" \
-m "../../models/movenet_estimator_model.pkl" \
--n-kps 17 \
--n-dim 3 
```

```bash
python scripts/train_estimator_pose_classifier.py \
-i "../../data/yolo_samples_csv_out" \
-m "../../models/yolo_estimator_model.pkl" \
--n-kps 17 \
--n-dim 3
```

## Video Fall Detection 

```bash
python scripts/video_inference_fall_detector.py \
-i "../../data/videos/uri.mp4" \
-o "../../data/videos/uri_meidapipe_out.mp4" \
-m "mediapipe" \
-c "../../models/mediapipe_knn_model.pkl" 
```

```bash
python scripts/video_inference_fall_detector.py \
-i "../../data/videos/euge.mp4" \
-o "../../data/videos/euge_movenet_out.mp4" \
-m "movenet" \
-c "../../models/movenet_knn_model.pkl" 
```

```bash
python scripts/video_inference_fall_detector.py \
-i "../../data/videos/uri.mp4" \
-o "../../data/videos/uri_yolo_out.mp4" \
-m "yolo" \
-c "../../models/yolo_knn_model.pkl" 
```

```bash
python scripts/video_inference_fall_detector.py \
-i "../../data/videos/uri.mp4" \
-o "../../data/videos/uri_out.mp4" \
-m "mediapipe" \
-c "../../models/mediapipe_estimator_model.pkl" 
```

```bash
python scripts/video_inference_fall_detector.py \
-i "../../data/videos/euge.mp4" \
-o "../../data/videos/euge_estimator_movenet_out.mp4" \
-m "movenet" \
-c "../../models/movenet_estimator_model.pkl" 
```

```bash
python scripts/video_inference_fall_detector.py \
-i "../../data/videos/euge.mp4" \
-o "../../data/videos/euge_yolo_out.mp4" \
-m "yolo" \
-c "../../models/yolo_estimator_model.pkl" 
```

## WebCam Fall Detection

```bash
python scripts/webcam_inference_fall_detector.py \
-m "mediapipe" \
-c "../../models/mediapipe_knn_model.pkl" 
```

```bash
python scripts/webcam_inference_fall_detector.py \
-m "mediapipe" \
-c "../../models/mediapipe_estimator_model.pkl" 
```

```bash
python scripts/webcam_inference_fall_detector.py \
-m "movenet" \
-c "../../models/movenet_knn_model.pkl" 
```

```bash
python scripts/webcam_inference_fall_detector.py \
-m "movenet" \
-c "../../models/movenet_estimator_model.pkl" 
```

```bash
python scripts/webcam_inference_fall_detector.py \
-m "yolo" \
-c "../../models/yolo_knn_classification_model.pkl" 
```

```bash
python scripts/webcam_inference_fall_detector.py \
-m "yolo" \
-c "../../models/yolo_estimator_classification_model.pkl" 
```



# Steps to Evaluate new Dataset / Model

1. convert dataset to folder dataset
2. generate keypoints samples
3. train classifier
4. validate(TODO)
5. run inference on example image/video/webcam
   
## 1


## 2

```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/custom_dataset/train" \
-o "../../data/yolo_custom_out/train" \
-f "../../data/yolo_custom_csv_out/train" \
-m "yolo" \
--max-samples 3000
```

```bash
python scripts/generate_landmarks_dataset.py \
-i "../../data/custom_dataset/test" \
-o "../../data/yolo_custom_out/test" \
-f "../../data/yolo_custom_csv_out/test" \
-m "yolo" \
--max-samples 500
```