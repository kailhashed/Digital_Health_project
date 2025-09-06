# Ensemble Audio Emotion Analysis - Testing Report

**Generated:** 2025-09-06 22:16:57

## Dataset Information

- **Total Files:** 160
- **Training Files:** 112
- **Test Files:** 48
- **Emotion Classes:** 8

### Emotion Distribution
- **neutral:** 6 files
- **calm:** 6 files
- **surprised:** 6 files
- **sad:** 6 files
- **disgust:** 6 files
- **angry:** 6 files
- **happy:** 6 files
- **fearful:** 6 files

## Performance Summary

- **Mean Accuracy:** 0.562 ± nan
- **Mean F1-Score:** 0.536 ± nan

## Best Configurations

### Best Accuracy
- **Configuration:** optimal_configuration
- **Accuracy:** 0.562
- **F1-Score:** 0.536
- **Vocal Weight:** 0.7

### Best F1-Score
- **Configuration:** optimal_configuration
- **Accuracy:** 0.562
- **F1-Score:** 0.536
- **Vocal Weight:** 0.7

## Detailed Results

| Configuration | Accuracy | F1-Score | Vocal Weight | Text Weight | Whisper Model |
|---------------|----------|----------|--------------|-------------|---------------|
| optimal_configuration | 0.562 | 0.536 | 0.70 | 0.30 | tiny |

## Recommendations

Based on the testing results:

1. **Moderate vocal bias** configurations perform best, providing good balance between vocal and text cues.
2. **Recommended configuration:** optimal_configuration with 56.2% accuracy.
3. **Use cases:** This configuration is suitable for detecting vocal-text emotion mismatches.
