#!/usr/bin/env python3
"""
Setup script for migrating EDICT to Python 3.12
This script helps automate the migration process and checks compatibility.
"""

import sys
import subprocess
import os
from pathlib import Path
import shutil

def check_python_version():
    """Check if we're running Python 3.12+"""
    version = sys.version_info
    if version.major != 3 or version.minor < 12:
        print(f"Warning: This script is designed for Python 3.12+, but you're running {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install requirements from the updated requirements file"""
    requirements_file = Path("requirements_py312.txt")
    if not requirements_file.exists():
        print("❌ requirements_py312.txt not found!")
        return False
    
    print("📦 Installing Python 3.12 compatible requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def backup_original_files():
    """Create backups of original files"""
    files_to_backup = [
        "edict_functions.py",
        "environment.yaml"
    ]
    
    backup_dir = Path("backup_original")
    backup_dir.mkdir(exist_ok=True)
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            shutil.copy2(file_path, backup_dir / file_path)
            print(f"✓ Backed up {file_path}")
    
    print(f"✓ Original files backed up to {backup_dir}")

def update_files():
    """Update files to Python 3.12 compatible versions"""
    updates = [
        ("edict_functions_py312.py", "edict_functions.py"),
        ("environment_py312.yaml", "environment.yaml")
    ]
    
    for src, dst in updates:
        if Path(src).exists():
            if Path(dst).exists():
                # Create backup if not already done
                backup_path = Path(f"{dst}.backup")
                if not backup_path.exists():
                    shutil.copy2(dst, backup_path)
            
            shutil.copy2(src, dst)
            print(f"✓ Updated {dst}")
        else:
            print(f"⚠️  {src} not found, skipping update of {dst}")

def check_cuda_availability():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available - will use CPU (much slower)")
    except ImportError:
        print("⚠️  PyTorch not installed yet")

def verify_huggingface_token():
    """Check if HuggingFace token exists"""
    hf_auth_file = Path("hf_auth")
    if hf_auth_file.exists():
        print("✓ HuggingFace auth token found")
    else:
        print("⚠️  HuggingFace auth token not found")
        print("   Create 'hf_auth' file with your HuggingFace token")
        print("   Get token from: https://huggingface.co/settings/tokens")

def run_compatibility_test():
    """Run a basic compatibility test"""
    print("\n🧪 Running compatibility test...")
    try:
        # Test basic imports
        import torch
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        
        print("✓ Core libraries import successfully")
        
        # Test HuggingFace imports
        from transformers import CLIPTokenizer
        print("✓ Transformers library works")
        
        # Test diffusers
        try:
            from diffusers import AutoencoderKL
            print("✓ Diffusers library works")
        except ImportError as e:
            print(f"⚠️  Diffusers import issue: {e}")
        
        print("✓ Basic compatibility test passed")
        return True
        
    except ImportError as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 EDICT Python 3.12 Migration Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        print("Consider using Python 3.12+ for best compatibility")
    
    # Create backups
    print("\n📁 Creating backups...")
    backup_original_files()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        print("❌ Setup failed during requirements installation")
        return False
    
    # Update files
    print("\n📝 Updating files...")
    update_files()
    
    # Check CUDA
    print("\n🔍 Checking system...")
    check_cuda_availability()
    verify_huggingface_token()
    
    # Run compatibility test
    if run_compatibility_test():
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Ensure your HuggingFace token is in the 'hf_auth' file")
        print("2. Test the updated code with: python -c 'from edict_functions import *'")
        print("3. Run the Jupyter notebook to test full functionality")
    else:
        print("\n❌ Setup completed with warnings")
        print("Please check the error messages above and resolve any issues")

if __name__ == "__main__":
    main()
