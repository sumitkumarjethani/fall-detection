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
python scripts/movenet_pose_inference.py -i "/Users/vito/Documents/TFM-2023/fall-detection/data/samples/Fall/image_000000001.png" -o "/Users/vito/Documents/TFM-2023/fall-detection/data/output/image_000000001_output.jpg"
```