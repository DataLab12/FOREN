## [Frequency-aware vision transformers for high-fidelity super-resolution of Earth system models]([https://arxiv.org/abs/2502.06741](https://www.nature.com/articles/s41598-026-41020-5)) [ArXiv](https://arxiv.org/abs/2502.06741)


**ViSIR** combines Vision Transformers (ViT) with Sinusoidal Representation Networks (SIRENs) to enhance image super-resolution tasks. The model introduces SIREN-based feedforward layers within the transformer architecture, enabling superior recovery of high-frequency details in the output image.

---

##  Overview

ViSIR consists of the following components:

- **Patch Embedding**: Converts input images into non-overlapping patches for transformer processing.
- **Transformer Encoder**: Applies multi-head self-attention to model relationships between patches.
- **SIREN-Driven Feedforward**: Replaces the traditional MLP block with sinusoidal activation layers for better frequency learning.
- **SIREN Decoder**: Maps the processed representation to a high-resolution output image.

---

##  Directory Structure
FinalScoresVitSIRENF*.xlsx                    # Training results (loss, PSNR, SSIM, etc.)

ViTSIREN2comp_image.png                     # Visualization comparisons of output vs ground truth

visir_main.py                                 # Main training script (contains model and pipeline)

├── LR/  Folder with low-resolution input images

├── HR/  Folder with high-resolution ground truth images




## Model Highlights

- Integrates a ViT encoder with a SIREN-based decoder.
- Uses positional embeddings for patch ordering.
- Enhances reconstruction quality using sine activation functions (SIREN).
- Outputs a high-resolution image that is 3× larger than the input.

---

## Getting Started

### 🔧 Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

🖼 Dataset Preparation
Place your image pairs in the following folders:

LR/: Low-resolution images (e.g., 0.png)

HR/: High-resolution (3×) images with the same filenames (e.g., 0.png)

---

## Run Training
Update the visir_main.py script as needed, then run:

```
python3 VISIR.py
python3 FOREN.py
```
---
Inside the script, you can modify:

num_of_images: Total number of image pairs to train on

EPOCHS: Number of training epochs (default: 1000)

Freq: Omega₀ value for SIREN's sine activation

Layer: Number of hidden layers in the SIREN block

---
## Output Metrics
After training, the model will generate:

PSNR and SSIM scores

Loss curves and epoch tracking

Visual side-by-side comparisons (ViTSIREN2comp_image#*.png)

An Excel summary of results (FinalScoresVitSIRENF{Freq}H{Layer}.xlsx)

### Metrics
MSE: Pixel-wise error

PSNR: Measures peak signal quality

SSIM: Evaluates structural similarity of images

---
Find a warmup example at Example folder

---


Citation
Zeraatkar, E., Faroughi, S.A. & Tešić, J. Frequency-aware vision transformers for high-fidelity super-resolution of Earth system models. Sci Rep 16, 10363 (2026). https://doi.org/10.1038/s41598-026-41020-5


Acknowledgments
ViSIR builds on ideas from:

[SIREN: Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929)

---
Contact
For inquiries or collaboration opportunities, feel free to reach out to Ehsan Zeraatkar via [email](mailto:ezeraatkar@gmail.com).




