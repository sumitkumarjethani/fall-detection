# Realtime fall detection


## Datasets

### [Falldataset](https://falldataset.com/)

#### Requirements:
- wget
- tar

#### Code

- [Source](./src/datasets/falldataset.py)

#### Descarga del dataset

- [Script](./src/scripts/download_falldataset.py)

```bash
python src/scripts/download_falldataset.py -O ./data
```

#### Procesado del dataset

- [Script](./src/scripts/process_falldataset.py)

```bash
python src/scripts/process_falldataset.py -I ./data
```

#### Movenet Inference

```bash
python scripts/movenet_pose_inference.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/fall-sample.png" -o "/Users/vito/Documents/TFM-2023/fall-detection/data/fall-sample-output.jpg"
```

#### Mediaipe inference

```bash
python scripts/mediapipe_pose_inference.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/fall-sample.png" -o "/Users/vito/Documents/TFM-2023/fall-detection/data/fall-sample-output.jpg"
```

#### Yolo Pose Inferece

```bash
python scripts/yolo_pose_inference.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/fall-sample.png" -o "/Users/vito/Documents/TFM-2023/fall-detection/data/fall-sample-output.jpg" -m "/Users/vito/Documents/TFM-2023/fall-detection/models/yolo-pose/yolov7-w6-pose.pt"
```


#### Generate Landmarks Dataset

- Test small script
```bash
python scripts/generate_landmarks_dataset.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/test_dataset" -o "/Users/vito/Documents/TFM-2023/fall-detection/data/test_dataset_out" -f "/Users/vito/Documents/TFM-2023/fall-detection/data/test_dataset_csv" -m "mediapipe"
```

- Full Falldataset
```bash
python scripts/generate_landmarks_dataset.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/samples" -o "/Users/vito/Documents/TFM-2023/fall-detection/data/samples_out" -f "/Users/vito/Documents/TFM-2023/fall-detection/data/samples_csv_out" -m "mediapipe" --max-samples 1000
```

#### Train Pose Classification

- test train on small dataset


```bash
python scripts/train_pose_classifier.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/test_dataset_csv" -m "/Users/vito/Documents/TFM-2023/fall-detection/models/test_pose_classification_model.pkl"
```

-  train on falldataset dataset

```bash
python scripts/train_pose_classifier.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/samples_csv_out" -m "/Users/vito/Documents/TFM-2023/fall-detection/models/falldataset_classification_model.pkl"
```

- train linear model

```bash
python scripts/train_estimator_pose_classifier.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/samples_csv_out" -m "/Users/vito/Documents/TFM-2023/fall-detection/models/falldataset_estimator_classification_model.pkl"
```

#### Fall Detection

```bash
python scripts/video_inference_fall_detector.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/videos/uri.mp4" -o "/Users/vito/Documents/TFM-2023/fall-detection/data/videos/uri_out.mp4" -m "mediapipe" -c "/Users/vito/Documents/TFM-2023/fall-detection/models/falldataset_classification_model.pkl" 
```

```bash
python scripts/video_inference_fall_detector.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/videos/uri.mp4" -o "/Users/vito/Documents/TFM-2023/fall-detection/data/videos/uri_out.mp4" -m "mediapipe" -c "/Users/vito/Documents/TFM-2023/fall-detection/models/falldataset_estimator_classification_model.pkl" 
```

- Webcam 
```bash
python scripts/webcam_inference_fall_detector.py -m "mediapipe" -c "/Users/vito/Documents/TFM-2023/fall-detection/models/falldataset_estimator_classification_model.pkl" 
```

