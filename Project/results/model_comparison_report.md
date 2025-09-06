# Model Comparison Report: ResNet vs SimpleCNN

## Executive Summary
This report compares the performance of ResNet and SimpleCNN models on emotion recognition.

## Overall Performance

| Metric | SimpleCNN | ResNet | Improvement |
|--------|-----------|--------|-------------|
| Test Accuracy | 0.6872 | 0.7634 | +0.0762 |
| Macro F1-Score | 0.7028 | 0.7603 | +0.0575 |

## Key Findings

- **ResNet achieves 76.34% test accuracy** vs SimpleCNN's 68.72%
- **Performance gain: +7.62 percentage points**
- ResNet shows improved performance across most emotions
- Both models show strong performance on 'surprised' emotion

## Architecture Comparison

### SimpleCNN
- 4 convolutional layers
- Standard CNN architecture
- Dropout regularization

### ResNet
- Residual connections
- Skip connections prevent vanishing gradients
- Better feature learning capability

## Conclusion
ResNet demonstrates superior performance due to its residual architecture, achieving better feature learning and generalization for emotion recognition tasks.
