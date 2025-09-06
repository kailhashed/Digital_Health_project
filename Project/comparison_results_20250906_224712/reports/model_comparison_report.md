# Comprehensive Model Comparison Report

**Generated:** 2025-09-06 23:29:03

## Executive Summary

**Dataset Size:** 11682 audio files
**Models Compared:** Ensemble (ResNet+STT+BERT) vs ResNet Standalone

**Winner:** Tie

## Detailed Performance Comparison

| Metric | Ensemble | ResNet | Difference |
|--------|----------|--------|-----------|
| Accuracy | 0.691 | 0.690 | +0.001 |
| Precision | 0.728 | 0.730 | -0.001 |
| Recall | 0.691 | 0.690 | +0.001 |
| F1-Score | 0.686 | 0.686 | +0.000 |
| Avg Confidence | 0.652 | 0.738 | -0.086 |
| Processing Time | 0.206s | 0.008s | +0.199s |

## Per-Class Performance Analysis

### F1-Score Comparison by Emotion

| Emotion | Ensemble | ResNet | Difference | Winner |
|---------|----------|--------|-----------|---------|
| angry | 0.779 | 0.779 | +0.001 | Tie |
| calm | 0.691 | 0.695 | -0.004 | Tie |
| disgust | 0.643 | 0.640 | +0.003 | Tie |
| fearful | 0.659 | 0.659 | +0.000 | Tie |
| happy | 0.598 | 0.599 | -0.002 | Tie |
| neutral | 0.724 | 0.722 | +0.002 | Tie |
| sad | 0.660 | 0.662 | -0.001 | Tie |
| surprised | 0.870 | 0.871 | -0.001 | Tie |

**Per-Class Summary:** Ensemble wins: 0, ResNet wins: 0, Ties: 8

## Speech-to-Text Analysis

- **Transcription Success Rate:** 1.000
- **Successful Transcriptions:** 11681
- **Total Files:** 11682

## Key Insights

1. **Comparable Performance:** Both models show similar accuracy (difference: +0.001)
2. **Processing Time:** Ensemble is slower (+0.199s per file) due to STT and text processing
3. **Confidence:** ResNet shows higher confidence in predictions (+0.086)

## Recommendations

1. **Choice depends on requirements** - similar performance
2. **Use ResNet** for speed, **Ensemble** for interpretability
3. **Consider application context** - real-time vs batch processing
4. **Monitor STT quality** if using ensemble in production

## Generated Files

- `metrics/comparison_metrics.json` - Detailed numerical metrics
- `visualizations/model_comparison.png` - Overall performance comparison
- `visualizations/per_class_comparison.png` - Per-emotion analysis
- `visualizations/confusion_matrices.png` - Confusion matrix comparison
- `predictions/comparison_results.json` - Raw prediction results

