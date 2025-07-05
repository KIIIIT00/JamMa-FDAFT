#!/usr/bin/env python3
"""
JamMa-FDAFT Setup Script

This script automates the setup process for JamMa-FDAFT including:
- Environment validation
- Dependency installation
- Directory structure creation
- Pre-trained model downloads
- Configuration validation

Usage:
    python setup.py [--mode {basic,full,dev}] [--force] [--no-download]
"""

import os
import sys
import argparse
import subprocess
import urllib.request
import gzip
import shutil
from pathlib import Path
import tempfile
import hashlib
import json


def print_banner():
    """Print JamMa-FDAFT setup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           JamMa-FDAFT Setup                                  ║
║                                                                              ║
║  Integrated Planetary Remote Sensing Image Matching                         ║
║  Architecture: FDAFT Encoder → Joint Mamba → C2F Matching                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")


def check_cuda_availability():
    """Check CUDA availability"""
    print("🔥 Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            
            print(f"✅ CUDA {cuda_version} available")
            print(f"   GPUs: {gpu_count} ({gpu_name})")
            return True
        else:
            print("⚠️  CUDA not available - CPU-only mode")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet - will check CUDA after installation")
        return None


def create_directory_structure():
    """Create necessary directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "data/megadepth/train",
        "data/megadepth/test", 
        "data/megadepth/index",
        "data/scannet/train",
        "data/scannet/test",
        "data/scannet/index",
        "assets/structured_forests",
        "weight",
        "jamma_fdaft_logs",
        "dump",
        "test_visualizations",
        "scripts/reproduce_train",
        "scripts/reproduce_test"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ Directory structure created")


def install_dependencies(mode="basic"):
    """Install required dependencies"""
    print(f"📦 Installing dependencies (mode: {mode})...")
    
    # Basic requirements
    basic_packages = [
        "torch>=2.0.1",
        "torchvision>=0.15.2", 
        "pytorch-lightning==1.3.5",
        "opencv-contrib-python==3.4.8.29",
        "opencv-python==4.4.0.46",
        "numpy==1.24.4",
        "scipy==1.10.1",
        "scikit-image==0.21.0",
        "matplotlib==3.7.5",
        "einops==0.8.1",
        "loguru==0.7.3",
        "yacs==0.1.8",
        "kornia==0.7.0",
        "pillow==10.4.0",
        "tqdm==4.67.1",
        "poselib==2.0.4"
    ]
    
    # Development packages
    dev_packages = [
        "pytest>=6.0.0",
        "black>=22.0.0",
        "isort>=5.0.0",
        "flake8>=4.0.0",
        "jupyter>=1.0.0",
        "tensorboard>=2.14.0"
    ]
    
    packages_to_install = basic_packages
    if mode in ["full", "dev"]:
        packages_to_install.extend(dev_packages)
    
    # Install packages
    for package in packages_to_install:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            continue
    
    print("✅ Dependencies installed")


def download_structured_forests():
    """Download pre-trained Structured Forests model"""
    print("🌲 Downloading Structured Forests model...")
    
    model_dir = Path("assets/structured_forests")
    model_path = model_dir / "model.yml"
    model_url = "https://github.com/pdollar/edges/raw/master/models/forest/modelBsds.yml.gz"
    
    if model_path.exists():
        print("   ✅ Structured Forests model already exists")
        return True
    
    try:
        # Download compressed model
        print("   📥 Downloading model...")
        with tempfile.NamedTemporaryFile(suffix=".yml.gz", delete=False) as tmp_file:
            urllib.request.urlretrieve(model_url, tmp_file.name)
            
            # Decompress
            print("   📂 Extracting model...")
            with gzip.open(tmp_file.name, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        # Cleanup
        os.unlink(tmp_file.name)
        
        print(f"   ✅ Structured Forests model saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download Structured Forests model: {e}")
        print("   💡 You can manually download from:")
        print(f"      {model_url}")
        return False


def create_sample_configs():
    """Create sample configuration files"""
    print("⚙️  Creating sample configuration files...")
    
    # Sample training script
    train_script = """#!/bin/bash
# JamMa-FDAFT Training Script

# Outdoor training (MegaDepth)
python train_jamma_fdaft.py \\
    configs/data/megadepth_trainval_832.py \\
    configs/jamma_fdaft/outdoor/final.py \\
    --exp_name jamma_fdaft_outdoor \\
    --batch_size 2 \\
    --gpus 1 \\
    --max_epochs 30

# Indoor training (ScanNet)  
python train_jamma_fdaft.py \\
    configs/data/scannet_trainval.py \\
    configs/jamma_fdaft/indoor/final.py \\
    --exp_name jamma_fdaft_indoor \\
    --batch_size 2 \\
    --gpus 1 \\
    --max_epochs 25
"""
    
    # Sample test script
    test_script = """#!/bin/bash
# JamMa-FDAFT Testing Script

# Test on MegaDepth
python test_jamma_fdaft.py \\
    configs/data/megadepth_test_1500.py \\
    configs/jamma_fdaft/outdoor/test.py \\
    --ckpt_path weight/jamma_fdaft_weight.ckpt \\
    --detailed_analysis \\
    --save_visualizations

# Test on ScanNet
python test_jamma_fdaft.py \\
    configs/data/scannet_test_1500.py \\
    configs/jamma_fdaft/indoor/test.py \\
    --ckpt_path weight/jamma_fdaft_weight.ckpt \\
    --detailed_analysis
"""
    
    # Write scripts
    train_script_path = Path("scripts/reproduce_train/jamma_fdaft_train.sh")
    test_script_path = Path("scripts/reproduce_test/jamma_fdaft_test.sh")
    
    with open(train_script_path, 'w') as f:
        f.write(train_script)
    os.chmod(train_script_path, 0o755)
    
    with open(test_script_path, 'w') as f:
        f.write(test_script)
    os.chmod(test_script_path, 0o755)
    
    print(f"   ✅ Training script: {train_script_path}")
    print(f"   ✅ Testing script: {test_script_path}")


def validate_installation():
    """Validate the installation"""
    print("🔍 Validating installation...")
    
    # Check key imports
    imports_to_check = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("kornia", "Kornia"),
        ("einops", "Einops"),
        ("loguru", "Loguru"),
    ]
    
    failed_imports = []
    
    for module, name in imports_to_check:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        print("💡 Try reinstalling with: pip install -r requirements.txt")
        return False
    
    # Check directory structure
    required_dirs = ["data", "assets", "weight", "configs"]
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"   ✅ {directory}/ directory")
        else:
            print(f"   ❌ {directory}/ directory missing")
            return False
    
    print("✅ Installation validation passed")
    return True


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*80)
    print("🎉 JamMa-FDAFT Setup Complete!")
    print("="*80)
    
    print("\n📋 Next Steps:")
    print("1. Download datasets (see docs/TRAINING.md):")
    print("   - MegaDepth for outdoor scenes")
    print("   - ScanNet for indoor scenes")
    
    print("\n2. Run the demo:")
    print("   python demo_jamma_fdaft.py")
    
    print("\n3. Start training:")
    print("   bash scripts/reproduce_train/jamma_fdaft_train.sh")
    
    print("\n4. Or test with pre-trained model:")
    print("   bash scripts/reproduce_test/jamma_fdaft_test.sh")
    
    print("\n📚 Documentation:")
    print("   - README.md - Complete project overview")
    print("   - docs/TRAINING.md - Dataset setup and training guide")
    print("   - docs/ARCHITECTURE.md - Detailed architecture docs")
    
    print("\n🔗 Useful Commands:")
    print("   python train_jamma_fdaft.py --help    # Training options")
    print("   python test_jamma_fdaft.py --help     # Testing options")
    print("   python demo_jamma_fdaft.py            # Run demo")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="JamMa-FDAFT Setup Script")
    parser.add_argument("--mode", choices=["basic", "full", "dev"], default="basic",
                       help="Installation mode (basic/full/dev)")
    parser.add_argument("--force", action="store_true",
                       help="Force reinstallation of existing components")
    parser.add_argument("--no-download", action="store_true",
                       help="Skip downloading pre-trained models")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only run validation checks")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # If validation only
    if args.validate_only:
        check_python_version()
        check_cuda_availability()
        success = validate_installation()
        sys.exit(0 if success else 1)
    
    # Full setup process
    try:
        # Step 1: Environment checks
        check_python_version()
        check_cuda_availability()
        
        # Step 2: Create directories
        create_directory_structure()
        
        # Step 3: Install dependencies
        if args.force or args.mode != "basic":
            install_dependencies(args.mode)
        
        # Step 4: Download models
        if not args.no_download:
            download_structured_forests()
        
        # Step 5: Create sample configs
        create_sample_configs()
        
        # Step 6: Validate installation
        if not validate_installation():
            print("\n❌ Setup completed with errors")
            sys.exit(1)
        
        # Step 7: Print next steps
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()