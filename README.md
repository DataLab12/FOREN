# ViSIR: Vision Transformer Meets SIREN for Image Super-Resolution

**ViSIR** combines Vision Transformers (ViT) with Sinusoidal Representation Networks (SIRENs) to enhance image super-resolution tasks. The model introduces SIREN-based feedforward layers within the transformer architecture, enabling superior recovery of high-frequency details in the output image.

---

## 🔍 Overview

ViSIR consists of the following components:

- **Patch Embedding**: Converts input images into non-overlapping patches for transformer processing.
- **Transformer Encoder**: Applies multi-head self-attention to model relationships between patches.
- **SIREN-Driven Feedforward**: Replaces the traditional MLP block with sinusoidal activation layers for better frequency learning.
- **SIREN Decoder**: Maps the processed representation to a high-resolution output image.

---

## 📁 Directory Structure

├── LR/  Folder with low-resolution input images
├── HR/  Folder with high-resolution ground truth images
├── FinalScoresVitSIRENF*.xlsx # Training results (loss, PSNR, SSIM, etc.)
├── ViTSIREN2comp_image#*.png # Visualization comparisons of output vs ground truth
├── visir_main.py # Main training script (contains model and pipeline)


## 🧠 Model Highlights

- Integrates a ViT encoder with a SIREN-based decoder.
- Uses positional embeddings for patch ordering.
- Enhances reconstruction quality using sine activation functions (SIREN).
- Outputs a high-resolution image that is 3× larger than the input.

---

## 🧪 Getting Started

### 🔧 Requirements

Install dependencies with pip:

```bash
pip install torch torchvision matplotlib numpy pillow openpyxl scikit-image
```

🖼 Dataset Preparation
Place your image pairs in the following folders:

LR/: Low-resolution images (e.g., 0.png)

HR/: High-resolution (3×) images with the same filenames (e.g., 0.png)

▶️ Run Training
Update the visir_main.py script as needed, then run:
