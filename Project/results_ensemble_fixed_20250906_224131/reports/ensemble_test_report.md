# Ensemble Model Test Report (Fixed STT)

**Generated:** 2025-09-06 22:42:19

## Model Configuration

- **Architecture:** ResNet (Vocal) + Whisper (STT) + BERT (Text)
- **Vocal Weight:** Variable (adaptive based on confidence)
- **STT Model:** Whisper Base
- **Text Model:** DistilRoBERTa-base (emotion)
- **Audio Processing:** 16kHz, Mel-spectrogram

## Overall Performance

- **Accuracy:** 0.662
- **Precision:** 0.701
- **Recall:** 0.662
- **F1-Score:** 0.662
- **Average Confidence:** 0.579
- **Processing Time:** 0.18s per file
- **Total Samples:** 240

## Component Analysis

- **Vocal Component (ResNet):** 0.667
- **Text Component (BERT):** 0.104
- **STT Success Rate:** 1.000
- **Successful Transcriptions:** 240

## Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|----------|
| angry | 0.583 | 0.700 | 0.636 | 30 |
| calm | 0.966 | 0.933 | 0.949 | 30 |
| disgust | 0.400 | 0.800 | 0.533 | 30 |
| fearful | 0.588 | 0.333 | 0.426 | 30 |
| happy | 0.621 | 0.600 | 0.610 | 30 |
| neutral | 0.783 | 0.600 | 0.679 | 30 |
| sad | 0.667 | 0.400 | 0.500 | 30 |
| surprised | 1.000 | 0.933 | 0.966 | 30 |

## Key Findings

- **Best Performing Emotion:** surprised (F1: 0.966)
- **Worst Performing Emotion:** fearful (F1: 0.426)
- **STT Integration:** Successfully fixed Windows compatibility issues
- **Speech Recognition:** 100.0% success rate
- **Vocal Dominance:** Vocal component is strongest

## Recommendations

2. **Model Enhancement:** Consider additional training or different architectures
3. **Data Quality:** Ensure consistent audio quality across datasets
4. **Bias Tuning:** Experiment with different vocal/text weight ratios
5. **Preprocessing:** Consider advanced audio preprocessing techniques

## Generated Files

- `metrics/performance_metrics.json` - Detailed numerical metrics
- `metrics/per_class_metrics.csv` - Per-class performance data
- `visualizations/confusion_matrix.png` - Confusion matrix heatmap
- `visualizations/per_class_performance.png` - Per-class metrics charts
- `visualizations/component_comparison.png` - Component accuracy comparison
- `visualizations/confidence_analysis.png` - Confidence distribution analysis
- `visualizations/stt_analysis.png` - Speech-to-text analysis
- `predictions/raw_results.json` - Raw prediction results

