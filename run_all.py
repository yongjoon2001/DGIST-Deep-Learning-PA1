#!/usr/bin/env python3

import subprocess
import sys
import os

def run_script(script_name, description):
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              cwd=os.getcwd(), 
                              capture_output=False, 
                              text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
    except Exception as e:
        print(f"❌ Error running {description}: {e}")

def main():
    print("PA1 - Running all implementations")
    print("This will run all 4 implementations sequentially.")
    print("Make sure you have the required dependencies installed:")
    print("- numpy, matplotlib, seaborn")
    print("- torch (for PyTorch implementations)")
    
    scripts = [
        ("nn_pure_python.py", "3-layer Neural Network (Pure Python)"),
        ("nn_framework.py", "3-layer Neural Network (PyTorch)"),
        ("cnn_pure_python.py", "3-layer CNN (Pure Python)"),
        ("cnn_framework.py", "3-layer CNN (PyTorch)")
    ]
    
    for script, description in scripts:
        if os.path.exists(script):
            run_script(script, description)
        else:
            print(f"❌ Script {script} not found!")
    
    print("\n" + "="*60)
    print("All implementations completed!")
    print("Check the results/ directory for outputs.")
    print("="*60)

if __name__ == "__main__":
    main()