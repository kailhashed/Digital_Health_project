# Comprehensive Model Testing Report

## Executive Summary

This report presents comprehensive testing results for 2 emotion recognition models.

## Overall Performance

| Model | Accuracy | Macro AUC | Macro F1 | Weighted F1 |
|-------|----------|-----------|----------|-------------|
| ResNet | 0.7634 | 0.9685 | 0.7603 | 0.7621 |
| SimpleCNN | 0.6872 | 0.9488 | 0.7028 | 0.6855 |

## Best Performing Model: ResNet

- **Test Accuracy**: 0.7634 (76.34%)
- **Macro AUC**: 0.9685
- **Macro F1-Score**: 0.7603
- **Weighted F1-Score**: 0.7621

## Model Rankings

1. **ResNet**: 76.34% accuracy
2. **SimpleCNN**: 68.72% accuracy

## Recommendations

Based on the comprehensive testing results, **ResNet** is recommended for production deployment due to its superior performance across all metrics.

## Files Generated

- `model_performance_summary.csv` - Overall performance comparison
- `comprehensive_model_comparison.png` - Performance visualizations
- `simplecnn/` - Detailed results for SimpleCNN
- `resnet/` - Detailed results for ResNet
