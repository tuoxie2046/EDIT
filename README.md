# Official Implementation of EDICT: Exact Diffusion Inversion via Coupled Transformations

[Arxiv](https://arxiv.org/abs/2211.12446)

The original framework for this code is based off of Github user bloc97's [implementation of Prompt-to-Prompt](https://github.com/bloc97/CrossAttentionControl). 

# What is EDICT?

EDICT (Exact Diffusion Inversion via Coupled Transformations) is an algorithm that closely mirrors a typical generative diffusion process but in an invertible fashion. We achieve this by tracking a *pair* of intermediate representations instead of just one. This exact invertibility enables edits that remain extremely faithful to the original image. Check out our [paper](https://arxiv.org/abs/2211.12446) for more details and don't hesitate to reach out with questions!

# Setup

## 🆕 Python 3.12 Setup (Recommended)

For the latest Python 3.12 compatible version with modern libraries:

### Quick Setup
```bash
# Create Python 3.12 environment
conda env create -f environment_py312_macos.yaml
conda activate edict_py312

# Fix PyTorch if you encounter segmentation faults
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Add your HuggingFace token
echo "your_hf_token_here" > hf_auth

# Test the setup
python -c "from edict_functions_py312 import *"
```

### Alternative pip installation
```bash
# Create virtual environment
python3.12 -m venv edict_py312
source edict_py312/bin/activate  # On Windows: edict_py312\Scripts\activate

# Install requirements
pip install -r requirements_py312_macos.txt

# Fix PyTorch if needed
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

### What's Updated in Python 3.12 Version
- ✅ **Python 3.12** compatibility with modern libraries
- ✅ **PyTorch 2.1+** with improved performance
- ✅ **Transformers 4.35+** with latest features
- ✅ **Diffusers 0.24+** with enhanced stability
- ✅ **Improved error handling** and type annotations
- ✅ **Better device detection** (CUDA/CPU fallback)
- ✅ **Updated API calls** for deprecated functions

### Python 3.12 Usage
```python
# Import the updated functions
from edict_functions_py312 import EDICT_editing

# Use exactly like the original
edited_image, original_recon = EDICT_editing(
    "path/to/image.jpg",
    base_prompt="a photo of a dog",
    edit_prompt="a photo of a cat",
    steps=50,
    run_baseline=True
)
```

### Test Notebook
Check out [EDICT_py312_test.ipynb](EDICT_py312_test.ipynb) for examples with the Python 3.12 setup.

---

## Original Python 3.8 Setup

### HF Auth token

Paste a copy of a suitable [HF Auth Token](https://huggingface.co/docs/hub/security-tokens) into [hf_auth](hf_auth) with no new line (to be read by both `edict_functions.py` and `edict_functions_py312.py`)
```
with open('hf_auth', 'r') as f:
    auth_token = f.readlines()[0].strip()
    
```

Example file at `./hf_auth`
```
abc123abc123
```

### Environment

Run  `conda env create -f environment.yaml`, activate that conda env (`conda activate edict`). Run jupyter with that conda env active

# Experimentation

## Python 3.12 Version
Check out [EDICT_py312_test.ipynb](EDICT_py312_test.ipynb) for examples using the updated Python 3.12 compatible code.

## Original Version
Check out [this notebook](EDICT.ipynb) for examples of how to use EDICT; including edits on randomly selected in-the-wild imagery ([alternative to render in browser](EDICT_no_images.ipynb)).

# Example Results

## Changing Dog Breeds

![Dogs](figs/edits_dogs.png)

## Miscellaneous


![Some edits](figs/edits_1.png)

![Some more edits](figs/edits_2.png)

# Other Files

## Core Files
[edict_functions.py](edict_functions.py) has the core functionality of EDICT for Python 3.8. [edict_functions_py312.py](edict_functions_py312.py) contains the updated version compatible with Python 3.12 and modern libraries.

[my_diffusers](my_diffusers) is a very slightly changed version of the [HF Diffusers repo](https://github.com/huggingface/diffusers) to work in double precision and avoid floating-point errors in our inversion/reconstruction experiments. EDICT editing also works at lower precisions, but to allow all experiments in a unified setting we present the double-precision floating point version of the code.

## Python 3.12 Migration Files
- `edict_functions_py312.py` - Updated main functions with Python 3.12 compatibility
- `environment_py312_macos.yaml` - Modern conda environment for macOS
- `requirements_py312_macos.txt` - Pip requirements for Python 3.12 setup
- `setup_py312.py` - Automated migration script
- `EDICT_py312_test.ipynb` - Test notebook for the new setup
- `MIGRATION_GUIDE_PY312.md` - Comprehensive migration guide
- `PYTHON312_MIGRATION_STATUS.md` - Migration status and troubleshooting

## Troubleshooting

### Common Issues with Python 3.12 Setup
1. **PyTorch Segmentation Fault**: Downgrade to stable version:
   ```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
   ```

2. **xformers Compilation Error**: This is optional and can be skipped on macOS:
   ```bash
   # xformers is excluded from macOS requirements due to compilation issues
   # The code works fine without it, just with slightly higher memory usage
   ```

3. **CUDA Not Available**: On macOS, this is expected. The code will automatically use CPU:
   ```bash
   # CPU fallback is automatic - performance will be slower but functional
   ```

For detailed troubleshooting, see [MIGRATION_GUIDE_PY312.md](MIGRATION_GUIDE_PY312.md).

# Quick Reference

## Python 3.12 (Recommended)
```bash
# Setup
conda env create -f environment_py312_macos.yaml
conda activate edict_py312
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
echo "your_hf_token" > hf_auth

# Usage
python -c "from edict_functions_py312 import EDICT_editing"
jupyter notebook EDICT_py312_test.ipynb
```

## Python 3.8 (Original)
```bash
# Setup
conda env create -f environment.yaml
conda activate edict
echo "your_hf_token" > hf_auth

# Usage
python -c "from edict_functions import EDICT_editing"
jupyter notebook EDICT.ipynb
```

# Citation

If you find our work useful in your research, please cite:

```
@article{wallace2022edict,
  title={EDICT: Exact Diffusion Inversion via Coupled Transformations},
  author={Wallace, Bram and Gokul, Akash and Naik, Nikhil},
  journal={arXiv preprint arXiv:2211.12446},
  year={2022}
}
```

# License

Our code is BSD-3 licensed. See LICENSE.txt for details.

