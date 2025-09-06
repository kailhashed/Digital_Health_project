# 🎵 Dataset Setup Guide

This guide provides detailed instructions for downloading and setting up the emotional speech datasets used in this project.

## 📊 Dataset Overview

This project uses three publicly available emotional speech datasets:

| Dataset | Source | Files | Emotions | License | Size |
|---------|--------|-------|----------|---------|------|
| **RAVDESS** | Zenodo | 1,440 | 8 emotions | CC BY 4.0 | ~1.6 GB |
| **CREMA-D** | GitHub | 7,442 | 6 emotions | CC BY 4.0 | ~3.2 GB |
| **TESS** | Kaggle | 2,800 | 7 emotions | Open | ~400 MB |

**Total Combined**: 11,682 audio files (~5.2 GB)

---

## 🎯 Required Directory Structure

After downloading, organize your datasets in the following structure:

```
Project/
├── Dataset/
│   ├── RAVDESS/
│   │   ├── Actor_01/
│   │   │   ├── 03-01-01-01-01-01-01.wav
│   │   │   ├── 03-01-01-01-01-02-01.wav
│   │   │   └── ...
│   │   ├── Actor_02/
│   │   └── ... (Actor_01 through Actor_24)
│   ├── CREMA-D/
│   │   ├── AudioWAV/
│   │   │   ├── 1001_DFA_ANG_XX.wav
│   │   │   ├── 1001_DFA_DIS_XX.wav
│   │   │   └── ...
│   │   └── README.txt
│   └── TESS/
│       └── TESS Toronto emotional speech set data/
│           ├── OAF_angry/
│           ├── OAF_disgust/
│           ├── OAF_Fear/
│           └── ...
└── organized_by_emotion/          # Generated after preprocessing
    ├── angry/
    ├── calm/
    ├── disgust/
    ├── fearful/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprised/
```

---

## 📥 Dataset Download Instructions

### 1. RAVDESS Dataset

**The Ryerson Audio-Visual Database of Emotional Speech and Song**

#### Download Source
- **URL**: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)
- **License**: Creative Commons Attribution 4.0
- **Size**: ~1.6 GB

#### Download Steps
1. Visit the Zenodo link above
2. Download `Audio_Speech_Actors_01-24.zip`
3. Extract to `Project/Dataset/RAVDESS/`
4. The structure should be `Actor_01/` through `Actor_24/`

#### File Naming Convention
```
Modality-Vocal_channel-Emotion-Emotional_intensity-Statement-Repetition-Actor.wav
```

Example: `03-01-06-01-02-01-12.wav`
- `03` = Audio-only
- `01` = Speech
- `06` = Fearful
- `01` = Normal intensity
- `02` = Statement "Kids are talking by the door"
- `01` = 1st repetition
- `12` = Actor 12

#### Emotion Mapping
| Code | Emotion | Files per Actor |
|------|---------|-----------------|
| 01 | Neutral | 2 |
| 02 | Calm | 2 |
| 03 | Happy | 2 |
| 04 | Sad | 2 |
| 05 | Angry | 2 |
| 06 | Fearful | 2 |
| 07 | Disgust | 2 |
| 08 | Surprised | 2 |

---

### 2. CREMA-D Dataset

**Crowdsourced Emotional Multimodal Actors Dataset**

#### Download Source
- **URL**: [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- **License**: Creative Commons Attribution 4.0
- **Size**: ~3.2 GB

#### Download Steps
1. Visit the GitHub repository
2. Go to releases or download the full dataset
3. Alternative: Direct download from [University of Pennsylvania](https://web.eecs.umich.edu/~mihalcea/downloads/CREMA-D.zip)
4. Extract to `Project/Dataset/CREMA-D/`
5. Audio files should be in `AudioWAV/` folder

#### File Naming Convention
```
ActorID_Sentence_Emotion_Intensity.wav
```

Example: `1001_DFA_ANG_XX.wav`
- `1001` = Actor ID
- `DFA` = Sentence ("Dogs are sitting by the door")
- `ANG` = Angry
- `XX` = Intensity level

#### Emotion Mapping
| Code | Emotion | Count |
|------|---------|-------|
| ANG | Angry | ~1,271 |
| DIS | Disgust | ~1,271 |
| FEA | Fear | ~1,271 |
| HAP | Happy | ~1,271 |
| NEU | Neutral | ~1,087 |
| SAD | Sad | ~1,271 |

---

### 3. TESS Dataset

**Toronto Emotional Speech Set**

#### Download Source
- **URL**: [https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)
- **License**: Open (Academic Use)
- **Size**: ~400 MB

#### Download Steps
1. Create a Kaggle account if you don't have one
2. Visit the dataset URL
3. Click "Download" to get `toronto-emotional-speech-set-tess.zip`
4. Extract to `Project/Dataset/TESS/`
5. The main folder should be `TESS Toronto emotional speech set data/`

#### File Naming Convention
```
Speaker_Word_Emotion.wav
```

Example: `OAF_back_angry.wav`
- `OAF` = Older Adult Female
- `back` = Spoken word
- `angry` = Emotion

#### Emotion Mapping
| Emotion | OAF Files | YAF Files | Total |
|---------|-----------|-----------|-------|
| angry | 200 | 200 | 400 |
| disgust | 200 | 200 | 400 |
| fear | 200 | 200 | 400 |
| happy | 200 | 200 | 400 |
| neutral | 200 | 200 | 400 |
| pleasant_surprised | 200 | 200 | 400 |
| sad | 200 | 200 | 400 |

---

## 🔄 Data Preprocessing

After downloading all datasets, run the preprocessing script to organize files by emotion:

### Automatic Organization
```bash
# Navigate to project directory
cd Project/

# Run the organization script
python data_preprocessing.py

# Or use PowerShell script (Windows)
.\organize_emotion_dataset.ps1
```

This will create the `organized_by_emotion/` directory with the following structure:

```
organized_by_emotion/
├── angry/       # 1,863 files
├── calm/        # 192 files
├── disgust/     # 1,863 files
├── fearful/     # 1,863 files
├── happy/       # 1,863 files
├── neutral/     # 1,583 files
├── sad/         # 1,863 files
└── surprised/   # 592 files
```

### Manual Verification
```bash
# Count files in each emotion folder
ls organized_by_emotion/*/| wc -l

# Check total file count
find organized_by_emotion/ -name "*.wav" | wc -l
# Expected: 11,682 files
```

---

## 📋 Dataset Citations

### RAVDESS
```bibtex
@misc{livingstone2018ryerson,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)},
  author={Livingstone, Steven R and Russo, Frank A},
  year={2018},
  publisher={Zenodo},
  doi={10.5281/zenodo.1188976},
  url={https://zenodo.org/record/1188976}
}
```

### CREMA-D
```bibtex
@article{cao2014crema,
  title={CREMA-D: Crowd-sourced emotional multimodal actors dataset},
  author={Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Raquel C and Nenkova, Ani and Verma, Ragini},
  journal={IEEE transactions on affective computing},
  volume={5},
  number={4},
  pages={377--390},
  year={2014},
  publisher={IEEE}
}
```

### TESS
```bibtex
@misc{dupuis2010toronto,
  title={Toronto emotional speech set (TESS)},
  author={Dupuis, Kate and Pichora-Fuller, M Kathleen},
  year={2010},
  publisher={University of Toronto, Psychology Department},
  url={https://tspace.library.utoronto.ca/handle/1807/24487}
}
```

---

## ⚠️ Important Notes

### Legal Considerations
- **Academic Use**: All datasets are available for academic research
- **Commercial Use**: Check individual licenses before commercial use
- **Attribution**: Always cite the original datasets in publications

### Technical Requirements
- **Storage**: Minimum 6 GB free space
- **Format**: All files are in WAV format
- **Sample Rate**: Various (will be standardized to 16kHz during preprocessing)
- **Duration**: Various (will be normalized to 3 seconds during preprocessing)

### Common Issues

#### Download Problems
- **Slow Download**: Use a download manager for large files
- **Kaggle Authentication**: Ensure you're logged into Kaggle
- **Network Issues**: Some institutions block large downloads

#### Extraction Issues
- **Nested Folders**: Ensure proper directory structure after extraction
- **File Permissions**: Check read permissions on extracted files
- **Corrupt Files**: Re-download if extraction fails

#### Organization Problems
- **Missing Files**: Verify all datasets are downloaded completely
- **Permission Errors**: Run preprocessing with appropriate permissions
- **Disk Space**: Ensure sufficient space for organized copies

---

## 🔧 Troubleshooting

### Verify Dataset Integrity
```python
import os

def verify_datasets():
    """Verify all datasets are properly downloaded and organized."""
    
    # Expected file counts
    expected_counts = {
        'RAVDESS': 1440,    # 24 actors × 60 files each
        'CREMA-D': 7442,    # Various actors and sentences
        'TESS': 2800        # 2 speakers × 7 emotions × 200 words
    }
    
    # Check RAVDESS
    ravdess_path = "Dataset/RAVDESS"
    if os.path.exists(ravdess_path):
        wav_files = sum([len([f for f in os.listdir(f"{ravdess_path}/Actor_{i:02d}") 
                             if f.endswith('.wav')]) 
                        for i in range(1, 25)])
        print(f"RAVDESS: {wav_files}/{expected_counts['RAVDESS']} files")
    
    # Check CREMA-D
    crema_path = "Dataset/CREMA-D/AudioWAV"
    if os.path.exists(crema_path):
        wav_files = len([f for f in os.listdir(crema_path) if f.endswith('.wav')])
        print(f"CREMA-D: {wav_files}/{expected_counts['CREMA-D']} files")
    
    # Check TESS
    tess_path = "Dataset/TESS/TESS Toronto emotional speech set data"
    if os.path.exists(tess_path):
        wav_files = sum([len([f for f in os.listdir(f"{tess_path}/{emotion}")
                             if f.endswith('.wav')])
                        for emotion in os.listdir(tess_path)
                        if os.path.isdir(f"{tess_path}/{emotion}")])
        print(f"TESS: {wav_files}/{expected_counts['TESS']} files")

if __name__ == "__main__":
    verify_datasets()
```

### Fix Common Path Issues
```python
import os
import shutil

def fix_common_issues():
    """Fix common dataset organization issues."""
    
    # Create missing directories
    os.makedirs("Dataset/RAVDESS", exist_ok=True)
    os.makedirs("Dataset/CREMA-D/AudioWAV", exist_ok=True)
    os.makedirs("Dataset/TESS", exist_ok=True)
    os.makedirs("organized_by_emotion", exist_ok=True)
    
    # Create emotion subdirectories
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    for emotion in emotions:
        os.makedirs(f"organized_by_emotion/{emotion}", exist_ok=True)
    
    print("Directory structure created successfully!")

if __name__ == "__main__":
    fix_common_issues()
```

---

## 📞 Support

If you encounter issues with dataset setup:

1. **Check File Paths**: Ensure all datasets are in the correct directories
2. **Verify Downloads**: Check file sizes match expected values
3. **Run Verification**: Use the provided verification scripts
4. **Check Logs**: Review preprocessing logs for specific errors
5. **Contact Support**: Create an issue with detailed error information

---

## 🎯 Next Steps

After successfully setting up the datasets:

1. **Run Preprocessing**: `python data_preprocessing.py`
2. **Verify Organization**: Check `organized_by_emotion/` folders
3. **Start Training**: `python scripts/train_all_models.py`
4. **Monitor Progress**: Check `logs/` directory for training logs

The preprocessing step will standardize all audio files to:
- **Sample Rate**: 16kHz
- **Duration**: 3 seconds
- **Format**: WAV
- **Normalization**: Min-max scaling
- **Organization**: By emotion class

---

*Last updated: January 2025*
