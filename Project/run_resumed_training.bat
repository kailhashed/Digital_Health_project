@echo off
echo Starting DenseNet Resumed Training...
cd /d "%~dp0"
python resume_densenet_training.py --checkpoint results_densenet_20250906_230232\best_densenet.pth --max_epochs 100 --patience 10 --learning_rate 0.0005
pause
