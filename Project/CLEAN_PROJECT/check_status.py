#!/usr/bin/env python3
"""
Simple Status Checker for Training Progress
"""

import os
import glob
import time
from datetime import datetime

def check_status():
    """Check training status"""
    print("🔍 Training Status Check")
    print("=" * 40)
    print(f"📅 Time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    models = ['densenet', 'crnn', 'adaboost', 'xgboost', 'naivebayes']
    
    for model in models:
        model_dir = f"models/{model}"
        results_dir = f"results/{model}"
        
        # Check for model files
        if os.path.exists(model_dir):
            model_files = glob.glob(f"{model_dir}/*.pth") + glob.glob(f"{model_dir}/*.pkl")
            if model_files:
                latest_file = max(model_files, key=os.path.getctime)
                file_size = os.path.getsize(latest_file) / 1e6  # MB
                mod_time = datetime.fromtimestamp(os.path.getctime(latest_file))
                print(f"✅ {model.upper()}: Model saved ({file_size:.1f}MB, {mod_time.strftime('%H:%M:%S')})")
            else:
                print(f"🔄 {model.upper()}: Training...")
        else:
            print(f"⏳ {model.upper()}: Not started")
        
        # Check for results
        if os.path.exists(results_dir):
            result_files = glob.glob(f"{results_dir}/*.json")
            if result_files:
                print(f"   📊 Results available")
    
    print()

if __name__ == "__main__":
    check_status()
