#!/usr/bin/env python3
"""
Parallel Training Script for All Models
Runs DenseNet, CRNN, AdaBoost, XGBoost, and NaiveBayes training in parallel
"""

import os
import sys
import subprocess
import threading
import time
from datetime import datetime
import json
import psutil
import GPUtil
import warnings
warnings.filterwarnings('ignore')

class ParallelTrainer:
    """Manages parallel training of all models"""
    
    def __init__(self):
        self.models = [
            {'name': 'DenseNet', 'script': 'train_densenet.py', 'gpu': True},
            {'name': 'CRNN', 'script': 'train_crnn.py', 'gpu': True},
            {'name': 'AdaBoost', 'script': 'train_adaboost.py', 'gpu': False},
            {'name': 'XGBoost', 'script': 'train_xgboost.py', 'gpu': False},
            {'name': 'NaiveBayes', 'script': 'train_naivebayes.py', 'gpu': False}
        ]
        
        self.processes = {}
        self.results = {}
        self.start_time = None
        
    def check_system_resources(self):
        """Check system resources before starting"""
        print("🔍 Checking system resources...")
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"   CPU: {cpu_count} cores, {cpu_usage:.1f}% usage")
        
        # Check Memory
        memory = psutil.virtual_memory()
        print(f"   RAM: {memory.total / 1e9:.1f} GB total, {memory.percent:.1f}% used")
        
        # Check GPU
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}, {gpu.memoryTotal} MB total, {gpu.memoryUsed} MB used")
            else:
                print("   GPU: No GPU detected")
        except:
            print("   GPU: Unable to detect GPU")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        print(f"   Disk: {disk.free / 1e9:.1f} GB free")
        
        print()
    
    def create_directories(self):
        """Create necessary directories"""
        print("📁 Creating directories...")
        
        directories = [
            'models/densenet',
            'models/crnn', 
            'models/adaboost',
            'models/xgboost',
            'models/naivebayes',
            'results/densenet',
            'results/crnn',
            'results/adaboost', 
            'results/xgboost',
            'results/naivebayes'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"   ✓ {directory}")
        
        print()
    
    def run_model_training(self, model_info):
        """Run training for a single model"""
        model_name = model_info['name']
        script_name = model_info['script']
        use_gpu = model_info['gpu']
        
        print(f"🚀 Starting {model_name} training...")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            if use_gpu:
                env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
            
            # Run the training script
            process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            self.processes[model_name] = process
            
            # Monitor the process
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"✅ {model_name} training completed successfully!")
                self.results[model_name] = {
                    'status': 'success',
                    'stdout': stdout,
                    'stderr': stderr,
                    'returncode': process.returncode
                }
            else:
                print(f"❌ {model_name} training failed!")
                print(f"   Error: {stderr}")
                self.results[model_name] = {
                    'status': 'failed',
                    'stdout': stdout,
                    'stderr': stderr,
                    'returncode': process.returncode
                }
                
        except Exception as e:
            print(f"❌ {model_name} training crashed: {e}")
            self.results[model_name] = {
                'status': 'crashed',
                'error': str(e)
            }
    
    def run_parallel_training(self):
        """Run all models in parallel"""
        print("🎭 Starting Parallel Model Training")
        print("=" * 60)
        print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = datetime.now()
        
        # Check system resources
        self.check_system_resources()
        
        # Create directories
        self.create_directories()
        
        # Start all training processes
        threads = []
        for model_info in self.models:
            thread = threading.Thread(
                target=self.run_model_training,
                args=(model_info,)
            )
            thread.start()
            threads.append(thread)
            time.sleep(2)  # Small delay between starts
        
        # Wait for all threads to complete
        print("⏳ Waiting for all training processes to complete...")
        for thread in threads:
            thread.join()
        
        # Calculate total time
        total_time = datetime.now() - self.start_time
        
        print("\n" + "=" * 60)
        print("🎉 All training processes completed!")
        print(f"⏱️ Total time: {total_time}")
        print()
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print training summary"""
        print("📊 Training Summary:")
        print("-" * 40)
        
        successful = 0
        failed = 0
        
        for model_name, result in self.results.items():
            status = result['status']
            if status == 'success':
                print(f"✅ {model_name}: SUCCESS")
                successful += 1
            else:
                print(f"❌ {model_name}: {status.upper()}")
                failed += 1
        
        print(f"\n📈 Results: {successful} successful, {failed} failed")
        print()
    
    def save_results(self):
        """Save training results to file"""
        results_data = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_time': str(datetime.now() - self.start_time),
            'models': self.results,
            'summary': {
                'successful': sum(1 for r in self.results.values() if r['status'] == 'success'),
                'failed': sum(1 for r in self.results.values() if r['status'] != 'success'),
                'total': len(self.results)
            }
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/parallel_training_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to results/parallel_training_results.json")
    
    def monitor_resources(self):
        """Monitor system resources during training"""
        print("📊 Monitoring system resources...")
        
        while any(thread.is_alive() for thread in threading.enumerate() if thread != threading.current_thread()):
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # GPU usage (if available)
            gpu_info = ""
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info = f", GPU: {gpu.load*100:.1f}%"
            except:
                pass
            
            print(f"   CPU: {cpu_usage:.1f}%, RAM: {memory.percent:.1f}%{gpu_info}")
            time.sleep(30)  # Check every 30 seconds

def main():
    """Main function"""
    print("🎭 Parallel Emotion Recognition Model Training")
    print("=" * 60)
    print("Training models: DenseNet, CRNN, AdaBoost, XGBoost, NaiveBayes")
    print("Early stopping: 20 epochs patience, 200 epochs max")
    print("Data split: 80-10-10 (train-val-test)")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('data_loader.py'):
        print("❌ Error: data_loader.py not found!")
        print("   Please run this script from the CLEAN_PROJECT directory")
        return
    
    # Initialize and run parallel trainer
    trainer = ParallelTrainer()
    
    try:
        trainer.run_parallel_training()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("   Stopping all processes...")
        
        # Kill all running processes
        for model_name, process in trainer.processes.items():
            if process.poll() is None:  # Process is still running
                process.terminate()
                print(f"   Stopped {model_name}")
        
        print("   All processes stopped")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("   Stopping all processes...")
        
        # Kill all running processes
        for model_name, process in trainer.processes.items():
            if process.poll() is None:  # Process is still running
                process.terminate()
                print(f"   Stopped {model_name}")

if __name__ == "__main__":
    main()
