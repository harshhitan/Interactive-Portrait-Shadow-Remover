# Interactive Portrait Shadow Remover

## Overview
An end-to-end deep learning pipeline for removing harsh, asymmetrical shadows from facial portraits. This project utilizes a custom **4-Channel Pix2Pix U-Net** Architecture optimized with a deep **VGG16 Perceptual Loss** to recover facial morphology while maintaining biological identity. 

Rather than relying on closed-source GANs or pre-trained behemoths, this architecture was built from the ground up, featuring a self-contained synthetic data generation pipeline and a production-grade interactive Streamlit web application.

## Key Features
- **Synthetic Shadow Engine**: Procedurally generates complex algorithmic illumination drop-offs, creating perfect `[Shadow, Mask, Clean]` training triplets from high-resolution CelebA-HQ and FFHQ datasets.
- **4-Channel Neural Architecture**: Bypasses traditional RGB mapping by passing the user's explicit shadow mask into the spatial tensor block directly, massively improving localized convergence.
- **Cloud-Ready Web Interface**: Includes a responsive Streamlit application capable of dynamic user-drawn masks. The UI is hard-coded for Streamlit Community Cloud, fetching compiled model weights invisibly via the **Hugging Face Hub**.

---

## Directory Architecture

```text
Interactive_Shadow_Remover/
├── src/
│   ├── download_data.py       # Pulls FFHQ/CelebA-HQ from Hugging Face Datasets
│   ├── synthetic_data_gen.py  # Generates synthetic shadows & mathematical baselines
│   └── train_unet.py          # PyTorch training loop + VGG Perceptual Loss definition
├── evaluate/
│   ├── ablation_study.py      # Computes PSNR, SSIM, and Identity Error metrics
│   └── visual_comparisons/    # Generated multi-row visual grids
├── ui/
│   └── app.py                 # Streamlit UI (fetches HF weights + physical drawing canvas)
├── checkpoints/               # Target directory for locally trained models (.pth)
├── project.ipynb              # Complete end-to-end Orchestrator notebook for Google Colab
└── requirements.txt           # Unified project dependency tree
```

---

## Getting Started

### Method 1: Local Virtual Environment
If you possess a sufficient local GPU and wish to execute the complete pipeline natively:
1. **Initialize Virtual Environment**
   ```bash
   python3 -m venv venv
   
   # Activate on Linux/macOS:
   source venv/bin/activate
   # Activate on Windows:
   venv\Scripts\activate
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute Pipeline**
   ```bash
   python src/download_data.py           # Fetch models
   python src/synthetic_data_gen.py      # Build synthetic triplets
   python src/train_unet.py --epochs 20  # Train U-Net
   python evaluate/ablation_study.py     # Compute evaluation metrics
   ```
4. **Local Web Interface**
   ```bash
   streamlit run ui/app.py
   ```

### Method 2: Google Colab (Recommended)
The entire deep learning lifecycle is explicitly orchestrated into a single ready-to-run notebook.
1. Upload this localized codebase directory to your Google Drive.
2. Open `project.ipynb` in Google Colab.
3. Run the cells sequentially to install dependencies, formulate the synthetic triplets natively, train the U-Net via T4 GPUs, and evaluate the visual comparisons.
