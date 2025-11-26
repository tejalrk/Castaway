# DreamBooth - Custom Stable Diffusion Model Training

This project implements a DreamBooth pipeline for fine-tuning Stable Diffusion models on custom image datasets. It enables you to generate AI images of specific people, objects, or concepts by training the model with just a few sample images.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [APIs and Technologies](#apis-and-technologies)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training Process](#training-process)
- [Inference](#inference)
- [Memory Optimization](#memory-optimization)
- [Output Formats](#output-formats)
- [Important Notes](#important-notes)

## Overview

This notebook implements the DreamBooth technique to personalize text-to-image diffusion models. By training on 3-5 images of a subject with a unique identifier, the model learns to generate new images of that subject in various contexts, poses, and styles.

**Example Use Case:** Train on photos of "jennaor person" to generate new images like "photo of jennaor person in a dream" or "jennaor person as an astronaut".

## Features

- **Custom Model Training**: Fine-tune Stable Diffusion v1.5 on your own images
- **Prior Preservation**: Maintains model's ability to generate diverse outputs
- **Memory Efficient**: Optimized for Tesla T4 GPU (15GB VRAM)
- **Multiple Output Formats**: Generates both diffusers and checkpoint (.ckpt) formats
- **Interactive UI**: Gradio interface for easy image generation
- **Batch Inference**: Generate multiple images simultaneously
- **Google Drive Integration**: Optional model storage on Google Drive

## Requirements

### Hardware
- GPU: NVIDIA Tesla T4 (or equivalent with 9-16GB VRAM)
- RAM: 12GB+ recommended
- Storage: ~5GB for model weights

### Software
- Python 3.8+
- CUDA 11.8 or 12.1
- Google Colab environment (recommended)

## Installation

### 1. Clone Training Scripts
```bash
wget https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py
wget https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py
```

### 2. Install Dependencies
```bash
pip install git+https://github.com/ShivamShrirao/diffusers
pip install -U --pre triton
pip install accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers
```

### Key Libraries
- **diffusers**: Hugging Face's diffusion models library
- **transformers**: For text encoder components
- **accelerate**: Distributed training and optimization
- **bitsandbytes**: 8-bit Adam optimizer for memory efficiency
- **xformers**: Memory-efficient attention mechanisms
- **gradio**: Web UI for inference
- **safetensors**: Safe tensor serialization format

## APIs and Technologies

### 1. Hugging Face Hub API
**Purpose**: Model downloading and authentication

**Usage**:
- Downloads pre-trained Stable Diffusion v1.5 model
- Requires authentication token from https://huggingface.co/settings/tokens
- Access to licensed models (runwayml/stable-diffusion-v1.5)

**Configuration**:
```python
HUGGINGFACE_TOKEN = "hf_your_token_here"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
```

### 2. Google Drive API
**Purpose**: Cloud storage for model weights and training data

**Features**:
- Mount Google Drive to Colab environment
- Save trained models (~4-5GB) directly to Drive
- Persist training data between sessions

**Usage**:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Stable Diffusion Pipeline
**Components**:
- **Text Encoder**: CLIP ViT-L/14 (transforms text prompts to embeddings)
- **VAE**: Variational Autoencoder (stabilityai/sd-vae-ft-mse)
- **U-Net**: Denoising diffusion model
- **Scheduler**: DDIMScheduler for inference

**Model Architecture**:
```
Input Text → CLIP Encoder → U-Net (with VAE) → Generated Image
```

### 4. Training Framework
**DreamBooth Training Script**: Custom implementation with features:
- Prior preservation loss (prevents language drift)
- Mixed precision training (fp16)
- Gradient checkpointing
- 8-bit Adam optimizer
- Text encoder fine-tuning

### 5. Gradio Web Interface
**Purpose**: Interactive image generation UI

**Features**:
- Real-time prompt input
- Parameter adjustment (steps, guidance scale, dimensions)
- Negative prompt support
- Batch generation
- Image gallery display

### 6. NVIDIA CUDA & cuDNN
**Purpose**: GPU acceleration

**APIs Used**:
- CUDA kernels for tensor operations
- cuDNN for optimized neural network primitives
- Mixed precision (FP16/FP32) computation

### 7. PyTorch
**Core Framework**: Deep learning operations

**Key Features**:
- Autocast for automatic mixed precision
- Inference mode for optimization
- CUDA memory management
- Model serialization

## Configuration

### Training Parameters

#### Basic Settings
```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "stable_diffusion_weights/your_name"
```

#### Concept Configuration
```python
concepts_list = [
    {
        "instance_prompt": "photo of jennaor person",
        "class_prompt": "photo of a person",
        "instance_data_dir": "/content/data/jennaor",
        "class_data_dir": "/content/data/person"
    }
]
```

**Parameters Explained**:
- `instance_prompt`: Unique identifier for your subject (e.g., "zwx person", "jennaor person")
- `class_prompt`: Generic class description (e.g., "a person", "a dog")
- `instance_data_dir`: Your training images (3-5 images recommended)
- `class_data_dir`: Regularization images (auto-generated)

#### Advanced Training Parameters
```python
--pretrained_model_name_or_path: Base model to fine-tune
--pretrained_vae_name_or_path: "stabilityai/sd-vae-ft-mse"
--resolution: 512 (image resolution)
--train_batch_size: 1
--mixed_precision: "fp16"
--use_8bit_adam: Enables 8-bit optimizer
--gradient_accumulation_steps: 1
--learning_rate: 1e-6
--lr_scheduler: "constant"
--max_train_steps: 800
--num_class_images: 50 (regularization images)
--prior_loss_weight: 1.0
--train_text_encoder: Fine-tunes text encoder
--gradient_checkpointing: Reduces memory usage
```

## Usage

### Step 1: GPU Setup
Check available GPU and VRAM:
```python
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```

### Step 2: Mount Google Drive (Optional)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Authenticate with Hugging Face
```python
HUGGINGFACE_TOKEN = "your_token_here"
```

### Step 4: Prepare Training Images
- Upload 3-5 high-quality images of your subject
- Place in `instance_data_dir`
- Images should be clear, well-lit, and varied

### Step 5: Start Training
Run the training cell with configured parameters. Training takes approximately 30-60 minutes for 800 steps.

### Step 6: Generate Images
Use either:
- **Direct Python API**: For programmatic generation
- **Gradio UI**: For interactive experimentation

## Training Process

### DreamBooth Algorithm

1. **Prior Preservation**:
   - Generates class images using the base model
   - Prevents model from forgetting the general class concept
   - Regularization loss: `L_prior = E[||f(x_prior) - x_prior||²]`

2. **Fine-tuning**:
   - Trains on instance images with unique identifier
   - Updates both U-Net and text encoder weights
   - Loss: `L_total = L_instance + λ * L_prior`

3. **Optimization**:
   - 8-bit Adam optimizer reduces memory footprint
   - Gradient checkpointing trades compute for memory
   - Mixed precision (FP16) speeds up training

### Training Flow
```
Base Model → Load Weights
    ↓
Generate Class Images (Prior Preservation)
    ↓
Train on Instance Images (800 steps)
    ↓
Save Checkpoints
    ↓
Convert to .ckpt format
```

## Inference

### Python API
```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    safety_checker=None,
    torch_dtype=torch.float16
).to("cuda")

images = pipe(
    prompt="photo of jennaor person in a dream",
    height=512,
    width=512,
    num_images_per_prompt=4,
    num_inference_steps=24,
    guidance_scale=7.5,
    generator=torch.Generator(device='cuda').manual_seed(52362)
).images
```

### Inference Parameters
- `prompt`: Text description of desired image
- `negative_prompt`: What to avoid in generation
- `num_inference_steps`: Quality vs speed (24-50 recommended)
- `guidance_scale`: Adherence to prompt (7.5 default, higher = stricter)
- `height/width`: Output dimensions (512x512 default)
- `seed`: For reproducible results

### Gradio UI
Interactive interface with:
- Text input for prompts
- Slider controls for parameters
- Real-time image generation
- Gallery view for outputs

## Memory Optimization

### Configuration Options

| Configuration | fp16 | batch_size | gradient_accum | gradient_checkpoint | 8bit_adam | VRAM (GB) | Speed (it/s) |
|--------------|------|------------|----------------|---------------------|-----------|-----------|--------------|
| Minimum      | ✓    | 1          | 1              | ✓                   | ✓         | 9.92      | 0.93         |
| Recommended  | ✓    | 1          | 1              | ✓                   | ✓         | 9.92      | 0.93         |
| High Speed   | ✓    | 1          | 1              | ✗                   | ✓         | 11.17     | 1.14         |
| Full Precision | ✗  | 1          | 1              | ✓                   | ✗         | 15.79     | 0.77         |

### Memory-Saving Techniques

1. **Gradient Checkpointing** (`--gradient_checkpointing`):
   - Recomputes activations during backward pass
   - Saves ~2-3GB VRAM
   - 20-30% slower training

2. **8-bit Adam** (`--use_8bit_adam`):
   - Reduces optimizer memory by 75%
   - Minimal impact on quality
   - Saves ~4GB VRAM

3. **Mixed Precision** (`--mixed_precision="fp16"`):
   - FP16 computation with FP32 accumulation
   - 2x faster, 50% less memory
   - Requires modern GPU

4. **XFormers** (`pipe.enable_xformers_memory_efficient_attention()`):
   - Memory-efficient attention mechanism
   - Reduces inference memory by 30-40%
   - Faster generation

## Output Formats

### 1. Diffusers Format
**Location**: `OUTPUT_DIR/`
**Structure**:
```
stable_diffusion_weights/jennaor/
├── model_index.json
├── scheduler/
├── text_encoder/
├── tokenizer/
├── unet/
├── vae/
└── samples/
```

**Usage**: Direct loading with diffusers library
```python
pipe = StableDiffusionPipeline.from_pretrained(model_path)
```

### 2. Checkpoint Format (.ckpt)
**Location**: `OUTPUT_DIR/model.ckpt`
**Size**: ~2GB (fp16) or ~4GB (fp32)

**Usage**: Compatible with:
- AUTOMATIC1111 WebUI
- ComfyUI
- InvokeAI
- Other Stable Diffusion UIs

**Conversion**:
```bash
python convert_diffusers_to_original_stable_diffusion.py \
  --model_path $WEIGHTS_DIR \
  --checkpoint_path $ckpt_path \
  --half
```

### 3. Intermediate Samples
**Location**: `OUTPUT_DIR/samples/`
**Purpose**: Visual progress during training
**Frequency**: Set by `--save_interval`

## Important Notes

### Best Practices

1. **Training Images**:
   - Use 3-5 high-quality images
   - Vary poses, backgrounds, and lighting
   - Avoid group photos or heavy filters
   - Square aspect ratio recommended

2. **Prompts**:
   - Always include the unique identifier ("jennaor person")
   - Be specific with descriptions
   - Use negative prompts to avoid unwanted elements

3. **Training Steps**:
   - 800-1200 steps for people
   - 400-600 steps for objects/styles
   - More steps ≠ better (overfitting risk)

4. **Learning Rate**:
   - 1e-6 for text encoder training
   - 2e-6 without text encoder
   - Lower = safer but slower

### Common Issues

**CUDA Version Mismatch**:
```
RuntimeError: PyTorch and torchvision compiled with different CUDA versions
```
**Solution**: Reinstall matching versions or use Colab's default environment

**Out of Memory**:
- Enable gradient checkpointing
- Reduce batch size
- Disable text encoder training (`remove --train_text_encoder`)

**Poor Quality Results**:
- Increase training steps
- Use better quality training images
- Adjust learning rate
- Increase guidance scale during inference

### License & Attribution

- **Base Model**: [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) (CreativeML Open RAIL-M)
- **Training Script**: [ShivamShrirao/diffusers](https://github.com/ShivamShrirao/diffusers)
- **DreamBooth**: Original paper by Google Research

### Security Notes

**WARNING**: The notebook contains a hardcoded Hugging Face token:
```python
HUGGINGFACE_TOKEN = "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"
```

**Action Required**:
- **Immediately revoke** this token at https://huggingface.co/settings/tokens
- Generate a new token
- Never commit tokens to version control
- Use environment variables or secure vaults

### Storage Management

Free up space after training:
```python
# Delete intermediate checkpoints, keep only final .ckpt
# Run cleanup cell (Cell 22)
```

**Space Usage**:
- Diffusers format: ~4-5GB
- Checkpoint format: ~2GB (fp16)
- Training images: <100MB
- Regularization images: ~500MB

## Workflow Summary

```
1. Setup Environment → Install dependencies, authenticate
2. Configure Training → Set model name, output dir, concepts
3. Upload Images → Add 3-5 training images
4. Train Model → Run for 800 steps (~45 min)
5. Convert Format → Generate .ckpt file
6. Generate Images → Use Python API or Gradio UI
7. Export & Share → Download model or share on HuggingFace
```

## Support & Resources

- [DreamBooth Paper](https://arxiv.org/abs/2208.12242)
- [Hugging Face Diffusers Docs](https://huggingface.co/docs/diffusers)
- [Stable Diffusion Guide](https://stable-diffusion-art.com/)
- [Training Repository](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth)

## Example Prompts

After training on "jennaor person":
- "photo of jennaor person in a dream"
- "jennaor person as an astronaut in space"
- "portrait of jennaor person, oil painting"
- "jennaor person wearing a red dress in Paris"
- "professional headshot of jennaor person"

Use negative prompts to improve quality:
```
Negative: "blurry, low quality, distorted, ugly, duplicate"
```

---

**Note**: This is an educational project demonstrating DreamBooth fine-tuning. Ensure you have rights to use any images for training and follow ethical AI generation practices.
