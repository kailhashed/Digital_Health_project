#!/usr/bin/env python3
"""
Monitoring script to start resumed training with error handling
"""

import os
import sys
import subprocess
import traceback
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully!")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        traceback.print_exc()
        return False

def main():
    """Main monitoring function"""
    print("DenseNet Resumed Training Monitor")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Test imports
    print("\nStep 1: Testing imports and environment...")
    if not run_command("python test_resume_imports.py", "Import Test"):
        print("Import test failed. Cannot proceed with training.")
        return False
    
    # Step 2: Start resumed training
    print("\nStep 2: Starting resumed training...")
    cmd = ("python resume_densenet_training.py "
           "--checkpoint results_densenet_20250906_230232/best_densenet.pth "
           "--max_epochs 100 "
           "--patience 10 "
           "--learning_rate 0.0005")
    
    if run_command(cmd, "Resumed Training"):
        print("\n🎉 Training completed successfully!")
        
        # List results directories
        result_dirs = [d for d in os.listdir('.') if d.startswith('results_densenet_resumed_')]
        if result_dirs:
            latest_results = sorted(result_dirs)[-1]
            print(f"Results saved in: {latest_results}")
            
            # List contents of results directory
            results_path = os.path.join('.', latest_results)
            if os.path.exists(results_path):
                print(f"\nResults directory contents:")
                for item in os.listdir(results_path):
                    item_path = os.path.join(results_path, item)
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path)
                        print(f"  - {item} ({size:,} bytes)")
        
        return True
    else:
        print("\n❌ Training failed!")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ All operations completed successfully!")
        else:
            print("\n❌ Some operations failed!")
        
        input("\nPress Enter to continue...")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        input("\nPress Enter to continue...")
