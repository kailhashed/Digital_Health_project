# Ensemble Audio Emotion Analysis

A sophisticated audio sentiment analysis system that combines **Speech-to-Text (STT)**, **BERT text emotion classification**, and **ResNet vocal emotion recognition** to provide comprehensive emotion detection with a configurable bias towards vocal cues.

## 🎯 Key Features

- **Multi-Modal Analysis**: Combines vocal and textual emotion signals
- **Vocal Bias**: Configurable bias towards vocal emotion cues to detect when someone's tone doesn't match their words
- **Advanced Models**: 
  - Whisper for speech-to-text
  - BERT for text emotion classification
  - Trained ResNet for vocal emotion recognition
- **8 Emotion Classes**: angry, calm, disgust, fearful, happy, neutral, sad, surprised
- **High Confidence Detection**: Automatically increases vocal weight when vocal predictions are highly confident

## 🏗️ Architecture

```
Audio File Input
       ↓
   ┌─────────────────────────────────────┐
   │     ENSEMBLE PROCESSOR              │
   │                                     │
   │  ┌─────────────┐  ┌─────────────┐   │
   │  │   WHISPER   │  │   ResNet    │   │
   │  │    (STT)    │  │  (Vocal)    │   │
   │  └─────────────┘  └─────────────┘   │
   │         │                │          │
   │         ↓                │          │
   │  ┌─────────────┐         │          │
   │  │    BERT     │         │          │
   │  │   (Text)    │         │          │
   │  └─────────────┘         │          │
   │         │                │          │
   │         ↓                ↓          │
   │  ┌─────────────────────────────┐    │
   │  │    WEIGHTED ENSEMBLE        │    │
   │  │   (Vocal Bias: 0.6-0.8)     │    │
   │  └─────────────────────────────┘    │
   └─────────────────────────────────────┘
                   ↓
           Final Emotion Prediction
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install requirements
pip install -r requirements_ensemble.txt

# Or install individual packages
pip install torch librosa transformers openai-whisper
```

### 2. Basic Usage

```python
from src.models.ensemble_model import create_ensemble_model

# Create ensemble with 70% vocal bias
ensemble = create_ensemble_model(vocal_bias=0.7)

# Analyze an audio file
result = ensemble.predict_emotion("path/to/audio.wav")

print(f"Predicted Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Transcription: '{result['transcribed_text']}'")
```

### 3. Command Line Usage

```bash
# Analyze single audio file
python ensemble_emotion_analyzer.py audio_file.wav --vocal-bias 0.65

# Analyze directory of audio files
python ensemble_emotion_analyzer.py audio_directory/ --batch --detailed

# Save results to JSON
python ensemble_emotion_analyzer.py audio_file.wav --output results.json
```

### 4. Run Demo

```bash
# Interactive demonstration
python demo_ensemble.py
```

## 🎛️ Configuration Options

### Vocal Bias Settings

The vocal bias parameter controls how much weight is given to vocal vs. text emotion:

- **0.5**: Equal weight (50% vocal, 50% text)
- **0.65**: Moderate vocal bias (recommended default)
- **0.75**: Strong vocal bias 
- **0.85**: Very strong vocal bias
- **0.9+**: Almost pure vocal emotion detection

### When to Use Different Bias Settings

| Vocal Bias | Best For | Example Use Cases |
|------------|----------|-------------------|
| 0.5-0.6 | Clear, emotionally expressive text | Customer feedback analysis, written speech |
| 0.65-0.7 | General purpose (default) | Most audio analysis scenarios |
| 0.75-0.8 | Strong vocal cues, ambiguous text | Detecting sarcasm, emotional speech |
| 0.85-0.9 | Focus on vocal patterns | Mental health monitoring, stress detection |

## 🧠 Model Components

### 1. Speech-to-Text (Whisper)
- **Model**: OpenAI Whisper (base/small/medium)
- **Purpose**: Convert audio to text for linguistic analysis
- **Languages**: Multi-language support

### 2. Text Emotion Classification (BERT)
- **Model**: DistilRoBERTa fine-tuned on emotion data
- **Purpose**: Analyze emotional content of transcribed text
- **Classes**: Maps to our 8 emotion categories

### 3. Vocal Emotion Recognition (ResNet)
- **Model**: Your trained ResNet model
- **Purpose**: Analyze vocal characteristics (pitch, tone, prosody)
- **Input**: Mel spectrograms of audio

### 4. Ensemble Logic
- **Adaptive Weighting**: Increases vocal weight for high-confidence vocal predictions
- **Confidence Threshold**: 0.7 default for adaptive weighting
- **Bias Implementation**: Weighted combination with configurable vocal preference

## 📊 Example Results

```python
{
  "predicted_emotion": "sad",
  "confidence": 0.847,
  "transcribed_text": "I'm fine, everything is great",
  "probabilities": {
    "angry": 0.023,
    "calm": 0.089,
    "disgust": 0.012,
    "fearful": 0.156,
    "happy": 0.034,
    "neutral": 0.039,
    "sad": 0.847,
    "surprised": 0.000
  },
  "component_predictions": {
    "vocal": {
      "emotion": "sad",
      "confidence": 0.892
    },
    "text": {
      "emotion": "happy", 
      "confidence": 0.756
    }
  }
}
```

**Analysis**: The person says "I'm fine, everything is great" (text emotion: happy) but their vocal tone indicates sadness. The ensemble correctly identifies the true emotional state as sad due to vocal bias.

## 🔧 Advanced Usage

### Batch Processing

```python
# Process multiple files
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
results = ensemble.predict_batch(audio_files)

for result in results:
    print(f"File: {result['file_path']}")
    print(f"Emotion: {result['predicted_emotion']}")
```

### Custom Configuration

```python
from src.models.ensemble_model import EnsembleAudioSentimentAnalyzer

# Create with custom settings
ensemble = EnsembleAudioSentimentAnalyzer(
    resnet_model_path="custom/path/to/model.pth",
    whisper_model_size="medium",  # Better accuracy
    vocal_weight=0.8,             # Strong vocal bias
    text_weight=0.2,
    confidence_threshold=0.75     # Higher threshold for adaptive weighting
)
```

### Update Weights Dynamically

```python
# Start with moderate bias
ensemble = create_ensemble_model(vocal_bias=0.65)

# Increase vocal bias for specific analysis
ensemble.update_weights(vocal_weight=0.8, text_weight=0.2)
```

## 🎯 Use Cases

### 1. Sarcasm Detection
**Scenario**: Person says "That's just wonderful" in a sarcastic tone
- **Text Emotion**: Happy/Positive
- **Vocal Emotion**: Angry/Disgusted  
- **Ensemble Result**: Correctly identifies negative emotion

### 2. Emotional Suppression
**Scenario**: Person says "I'm okay" while crying
- **Text Emotion**: Neutral
- **Vocal Emotion**: Sad
- **Ensemble Result**: Identifies true sadness

### 3. Excitement Detection
**Scenario**: Person says "I got the job" with excited tone
- **Text Emotion**: Happy
- **Vocal Emotion**: Happy/Surprised
- **Ensemble Result**: High confidence happiness

## 🛠️ Troubleshooting

### Common Issues

1. **Whisper Not Loading**
   ```bash
   pip install openai-whisper
   # May require ffmpeg: apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)
   ```

2. **BERT Model Download Issues**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/
   ```

3. **ResNet Model Not Found**
   - Ensure the model exists at `models/resnet/best_ResNet.pth`
   - Train the ResNet model first using existing training scripts

4. **CUDA Memory Issues**
   - Use smaller Whisper model: `whisper_model_size="tiny"`
   - Reduce batch size for processing

### Performance Optimization

- **CPU Only**: Use `whisper_model_size="tiny"` or `"base"`
- **GPU Available**: Use `"small"` or `"medium"` for better accuracy
- **High Memory**: Use `"large"` for best performance

## 📈 Model Performance

The ensemble approach typically provides:
- **Better Accuracy**: Combines strengths of both vocal and text analysis
- **Robustness**: Works even when one modality fails (e.g., poor audio quality)
- **Context Awareness**: Detects when vocal and text emotions differ
- **Adaptability**: Configurable bias for different use cases

## 🤝 Contributing

To extend the ensemble model:

1. **Add New Text Models**: Modify `BERTEmotionClassifier` 
2. **Add New Vocal Models**: Extend `VocalEmotionClassifier`
3. **Custom Ensemble Logic**: Modify `EnsembleAudioSentimentAnalyzer`
4. **New Features**: Add to the main analyzer class

## 📄 License

This project extends the existing emotion recognition codebase. Please refer to the main project license.

---

**Note**: This ensemble model is designed to handle the complex scenario where verbal communication and vocal emotion may not align, providing a more nuanced understanding of true emotional state.
