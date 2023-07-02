# Realtime fall detection

## Dataset ingestion

### Fall dataset scripts

Link to dataset: [Falldataset](https://falldataset.com/)

**Requirements MAC users**:
- wget
- tar

**Requirements Windows users**:
- Download [wget](https://eternallybored.org/misc/wget/) binaries
- Copy the wget.exe to the c:\Windows\System32 folder location

**Download fall dataset**
```bash
python src/scripts/download_fall_dataset.py -O ./output_directory_path
```

**Process fall dataset**
```bash
python src/scripts/process_fall_dataset.py -I ./output_directory_path -O ./final_datasets
```

## Key point dataset generation
```bash
python src/scripts/create_key_point_dataset.py [-M model_name] -I ./final_datasets -O ./final_data
```