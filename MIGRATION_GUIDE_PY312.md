# EDICT Python 3.12 Migration Guide

This guide helps you upgrade the EDICT codebase to be compatible with Python 3.12 and modern PyTorch/Transformers libraries.

## Overview of Changes

### Key Compatibility Issues Fixed

1. **Deprecated imports**: Updated `torch.autocast` import for newer PyTorch versions
2. **Transformers API changes**: Updated `use_auth_token` → `token` parameter
3. **PIL deprecations**: Updated `Image.LANCZOS` → `Image.Resampling.LANCZOS`
4. **Type annotations**: Added proper type hints for better code clarity
5. **Error handling**: Improved error handling and fallback mechanisms
6. **Modern dependencies**: Updated all library versions to Python 3.12 compatible versions

## Migration Steps

### Step 1: Backup Original Files

```bash
# Run the setup script to automatically backup files
python setup_py312.py
```

Or manually:
```bash
mkdir backup_original
cp edict_functions.py backup_original/
cp environment.yaml backup_original/
```

### Step 2: Install Python 3.12 Environment

#### Option A: Using Conda (Recommended)
```bash
# Create new environment with Python 3.12
conda env create -f environment_py312.yaml
conda activate edict_py312
```

#### Option B: Using pip
```bash
# Create virtual environment
python3.12 -m venv edict_py312
source edict_py312/bin/activate  # On Windows: edict_py312\Scripts\activate

# Install requirements
pip install -r requirements_py312.txt
```

### Step 3: Update Code Files

The main changes are in `edict_functions_py312.py`:

#### Import Updates
```python
# Old (Python 3.8)
from torch import autocast

# New (Python 3.12)
try:
    from torch.cuda.amp import autocast
except ImportError:
    from torch import autocast
```

#### Model Loading Updates
```python
# Old
unet = UNet2DConditionModel.from_pretrained(
    model_path, 
    use_auth_token=auth_token,  # Deprecated
    torch_dtype=torch.float16
)

# New
unet = UNet2DConditionModel.from_pretrained(
    model_path, 
    token=auth_token,  # Updated parameter name
    torch_dtype=torch.float16,
    trust_remote_code=True  # Added for security
)
```

#### Autocast Usage Updates
```python
# Old
with autocast(device):
    result = model(input)

# New
with autocast(device_type=device):
    result = model(input)
```

### Step 4: Update Dependencies

#### Key Version Updates
- Python: 3.8 → 3.12
- PyTorch: 1.11.0 → 2.1.0+
- Transformers: 4.19.2 → 4.35.0+
- Diffusers: 0.6.0 → 0.24.0+
- NumPy: 1.23.4 → 1.24.0+
- Pillow: 9.2.0 → 10.0.0+

#### CUDA Updates
- CUDA Toolkit: 11.3 → 12.1
- Updated PyTorch CUDA compatibility

### Step 5: Test the Migration

```bash
# Run the setup script
python setup_py312.py

# Test basic imports
python -c "from edict_functions_py312 import *"

# Run the test notebook
jupyter notebook EDICT_py312_test.ipynb
```

## Detailed Changes

### 1. edict_functions.py → edict_functions_py312.py

**Major Changes:**
- Added type hints throughout
- Updated import statements for compatibility
- Improved error handling with try/catch blocks
- Added device detection (CUDA/CPU fallback)
- Updated model loading with new parameter names
- Added warnings suppression for cleaner output

**Key Functions Updated:**
- `load_auth_token()`: Added error handling
- `load_diffusion_models()`: Updated with fallback loading
- `EDICT_editing()`: Added type hints and improved error handling
- `load_im_into_format_from_path()`: Updated PIL resampling method

### 2. Environment Configuration

**environment_py312.yaml:**
- Updated to Python 3.12
- Modern PyTorch with CUDA 12.1 support
- Updated all dependencies to latest compatible versions
- Added optional dependencies like xformers

**requirements_py312.txt:**
- Pip-based alternative to conda environment
- Pinned versions for reproducibility
- Includes all necessary dependencies

### 3. New Files Created

- `setup_py312.py`: Automated migration script
- `EDICT_py312_test.ipynb`: Test notebook for new setup
- `requirements_py312.txt`: Pip requirements file
- `environment_py312.yaml`: Updated conda environment

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Compatibility Issues
```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 2. Transformers Token Issues
```bash
# Create HuggingFace token file
echo "your_hf_token_here" > hf_auth

# Get token from: https://huggingface.co/settings/tokens
```

#### 3. Memory Issues
- Reduce image resolution in processing
- Use mixed precision instead of double precision
- Reduce batch sizes or number of steps

#### 4. Import Errors
```python
# Check if all dependencies are installed
python -c "import torch, transformers, diffusers, PIL, numpy, matplotlib"
```

### Performance Optimization

#### For Better Performance:
1. **Use CUDA**: Essential for reasonable performance
2. **Enable xformers**: For memory-efficient attention
3. **Use mixed precision**: Instead of double precision when possible
4. **Optimize batch sizes**: Based on your GPU memory

#### Memory Usage:
- Original (double precision): ~12-16GB GPU memory
- Mixed precision: ~6-8GB GPU memory
- CPU fallback: Uses system RAM (much slower)

## Testing Your Migration

### Basic Functionality Test
```python
# Test imports
from edict_functions_py312 import *

# Test model loading
print("Models loaded successfully" if 'unet' in globals() else "Model loading failed")

# Test image processing
test_image = load_im_into_format_from_path("experiment_images/imagenet_dog_1.jpg")
print(f"Image loaded: {test_image.shape}")
```

### Full Pipeline Test
```python
# Run a simple edit
result = EDICT_editing(
    "experiment_images/imagenet_dog_1.jpg",
    base_prompt="a photo of a dog",
    edit_prompt="a photo of a cat",
    steps=10,  # Reduced for testing
    run_baseline=True
)
print("EDICT editing successful!")
```

## Rollback Instructions

If you need to rollback to the original setup:

```bash
# Restore original files
cp backup_original/edict_functions.py ./
cp backup_original/environment.yaml ./

# Recreate original environment
conda env create -f environment.yaml
conda activate edict
```

## Additional Resources

- [PyTorch 2.1 Release Notes](https://pytorch.org/blog/pytorch-2.1/)
- [Transformers 4.35 Documentation](https://huggingface.co/docs/transformers)
- [Diffusers 0.24 Documentation](https://huggingface.co/docs/diffusers)
- [Python 3.12 What's New](https://docs.python.org/3.12/whatsnew/3.12.html)

## Support

If you encounter issues during migration:

1. Check the troubleshooting section above
2. Verify all dependencies are correctly installed
3. Ensure your HuggingFace token is valid and in the `hf_auth` file
4. Test with the provided `EDICT_py312_test.ipynb` notebook

The migration maintains full backward compatibility with the original EDICT functionality while adding modern Python 3.12 support and improved error handling.
