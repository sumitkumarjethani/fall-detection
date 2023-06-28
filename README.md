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