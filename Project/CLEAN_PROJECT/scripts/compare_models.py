#!/usr/bin/env python3
"""
Script to compare all trained models
Uses the organized codebase structure
"""

import os
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

from evaluation import ModelComparator
from utils import Config, setup_logger


def main():
    """Main comparison function"""
    print("🔍 COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    
    # Setup
    Config.create_directories()
    logger = setup_logger("model_comparison")
    
    # Create model comparator
    comparator = ModelComparator()
    
    # Run full comparison
    try:
        comparison_results = comparator.run_full_comparison(Config.RESULTS_DIR)
        
        logger.info("Model comparison completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        logger.error(f"Comparison failed: {e}")
        return False


if __name__ == "__main__":
    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    success = main()
    if success:
        print("\n✨ Model comparison completed successfully!")
    else:
        print("\n💥 Model comparison failed!")

