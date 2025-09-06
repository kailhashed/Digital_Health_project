#!/usr/bin/env python3
"""
Training Monitor Script
Monitors the progress of parallel model training
"""

import os
import time
import json
import psutil
import GPUtil
from datetime import datetime
import glob

def check_training_progress():
    """Check the progress of all training processes"""
    print("🔍 Training Progress Monitor")
    print("=" * 50)
    print(f"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check system resources
    print("💻 System Resources:")
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"   CPU: {cpu_usage:.1f}%")
    print(f"   RAM: {memory.percent:.1f}% ({memory.used/1e9:.1f}GB / {memory.total/1e9:.1f}GB)")
    
    # Check GPU usage
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.load*100:.1f}% load, {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        else:
            print("   GPU: No GPU detected")
    except:
        print("   GPU: Unable to detect GPU")
    
    print()
    
    # Check for model files
    models = ['densenet', 'crnn', 'adaboost', 'xgboost', 'naivebayes']
    
    print("📊 Model Training Status:")
    for model in models:
        model_dir = f"models/{model}"
        results_dir = f"results/{model}"
        
        # Check if model directory exists
        if os.path.exists(model_dir):
            model_files = glob.glob(f"{model_dir}/*.pth") + glob.glob(f"{model_dir}/*.pkl")
            if model_files:
                latest_file = max(model_files, key=os.path.getctime)
                file_size = os.path.getsize(latest_file) / 1e6  # MB
                mod_time = datetime.fromtimestamp(os.path.getctime(latest_file))
                print(f"   ✅ {model.upper()}: Model saved ({file_size:.1f}MB, {mod_time.strftime('%H:%M:%S')})")
            else:
                print(f"   🔄 {model.upper()}: Training in progress...")
        else:
            print(f"   ⏳ {model.upper()}: Not started yet")
        
        # Check for results
        if os.path.exists(results_dir):
            result_files = glob.glob(f"{results_dir}/*.json")
            if result_files:
                print(f"      📈 Results available")
    
    print()
    
    # Check for parallel training results
    if os.path.exists("results/parallel_training_results.json"):
        with open("results/parallel_training_results.json", "r") as f:
            results = json.load(f)
        
        print("📋 Parallel Training Summary:")
        print(f"   Start time: {results['start_time']}")
        print(f"   Successful: {results['summary']['successful']}")
        print(f"   Failed: {results['summary']['failed']}")
        print(f"   Total: {results['summary']['total']}")
        
        if results['summary']['successful'] == results['summary']['total']:
            print("   🎉 All models completed successfully!")
        elif results['summary']['failed'] > 0:
            print("   ⚠️ Some models failed - check logs")
    
    print()

def main():
    """Main monitoring function"""
    print("🎭 Emotion Recognition Training Monitor")
    print("=" * 60)
    
    try:
        while True:
            check_training_progress()
            print("⏰ Refreshing in 30 seconds... (Ctrl+C to stop)")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")

if __name__ == "__main__":
    main()
