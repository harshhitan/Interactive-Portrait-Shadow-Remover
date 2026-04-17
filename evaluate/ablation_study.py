import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except ImportError:
    print("Install 'rich': pip install rich")
    sys.exit(1)

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    print("CRITICAL: Install facenet-pytorch")
    sys.exit(1)

# Project Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_unet import ShadowRemoverUNet


# Helper Functions
def load_image(path, mode="RGB"):
    return Image.open(path).convert(mode)

def to_numpy(tensor):
    return tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

def compute_metrics(gt, pred):
    data_range = pred.max() - pred.min()
    return {
        "ssim": ssim(gt, pred, data_range=data_range, channel_axis=-1),
        "psnr": psnr(gt, pred, data_range=data_range)
    }

# Main Evaluation 
def evaluate_metrics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console = Console()
    console.print(f"Running evaluation on [bold]{device}[/bold]")

    # Model 
    model = ShadowRemoverUNet()
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_shadow_unet.pth')

    if not os.path.exists(ckpt_path):
        console.print("[red]Checkpoint not found.[/red]")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_shadows')
    paths = {
        "shadow": os.path.join(base_dir, "images"),
        "mask": os.path.join(base_dir, "masks"),
        "clean": os.path.join(base_dir, "clean"),
        "baseline": os.path.join(base_dir, "baseline_gan"),
        "vis": os.path.join(os.path.dirname(__file__), "visual_comparisons")
    }

    os.makedirs(paths["vis"], exist_ok=True)

    if not os.path.exists(paths["shadow"]):
        console.print("[red]Dataset not found[/red]")
        return

    files = [f for f in os.listdir(paths["shadow"]) if f.endswith((".png", ".jpg"))][:50]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Metrics 
    metrics = {
        "ours": {"psnr": 0, "ssim": 0, "id": 0, "n": 0},
        "baseline": {"psnr": 0, "ssim": 0, "id": 0, "n": 0}
    }
    
    vis_data = []
    
    console.print(f"Evaluating {len(files)} samples...")

    for idx, name in enumerate(tqdm(files)):
        mask_path = os.path.join(paths["mask"], name.split('.')[0] + "_mask.png")
        if not os.path.exists(mask_path):
            continue

        # Load images
        shadow = load_image(os.path.join(paths["shadow"], name))
        clean = load_image(os.path.join(paths["clean"], name))

        # Tensors
        shadow_t = transform(shadow)
        mask_t = transform(load_image(mask_path, "L"))
        clean_t = transform(clean)

        input_t = torch.cat((shadow_t, mask_t), dim=0).unsqueeze(0).to(device)
        target_t = clean_t.unsqueeze(0).to(device)

        # Baseline
        baseline_path = os.path.join(paths["baseline"], name)
        has_baseline = os.path.exists(baseline_path)

        # ---------- Inference ----------
        with torch.no_grad():
            output_t = model(input_t)
            tgt_feat = face_model((target_t * 2) - 1)
            out_feat = face_model((output_t * 2) - 1)

            metrics["ours"]["id"] += nn.functional.mse_loss(out_feat, tgt_feat).item()

            if has_baseline:
                baseline_t = transform(load_image(baseline_path)).unsqueeze(0).to(device)
                base_feat = face_model((baseline_t * 2) - 1)
                metrics["baseline"]["id"] += nn.functional.mse_loss(base_feat, tgt_feat).item()

        # ---------- Image Metrics ----------
        out_np = to_numpy(output_t)
        tgt_np = to_numpy(target_t)

        # Calculate Ours
        ours_m = compute_metrics(tgt_np, out_np)
        metrics["ours"]["ssim"] += ours_m["ssim"]
        metrics["ours"]["psnr"] += ours_m["psnr"]
        metrics["ours"]["n"] += 1

        if has_baseline:
            base_np = to_numpy(baseline_t)
            base_m = compute_metrics(tgt_np, base_np)
            metrics["baseline"]["ssim"] += base_m["ssim"]
            metrics["baseline"]["psnr"] += base_m["psnr"]
            metrics["baseline"]["n"] += 1

        # ---------- Visualization Collection ----------
        if len(vis_data) < 4:  # Collect 4 rows for the final presentation grid
            vis_data.append({
                "input": np.array(shadow.resize((256, 256))),
                "mask": np.array(load_image(mask_path, "L").resize((256, 256))),
                "baseline": np.clip(base_np, 0, 1) if has_baseline else None,
                "ours": np.clip(out_np, 0, 1),
                "gt": np.array(clean.resize((256, 256)))
            })

    # ---------- Render Final Presentation Grid ----------
    if vis_data:
        n_rows = len(vis_data)
        has_baseline = vis_data[0]["baseline"] is not None
        cols = 5 if has_baseline else 4
        
        # Calculate figure size to maintain aspect ratio with tightly packed squares
        fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 3, n_rows * 3))
        plt.subplots_adjust(wspace=0.03, hspace=0.03) # Minimal spacing like the paper
        
        col_labels = ["Input", "Mask", "Baseline", "Ours", "GT"] if has_baseline else ["Input", "Mask", "Ours", "GT"]
        
        # Ensure 'axes' is explicitly 2D even if n_rows == 1
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for r in range(n_rows):
            d = vis_data[r]
            row_images = [d["input"], d["mask"]]
            if has_baseline:
                row_images.append(d["baseline"])
            row_images.append(d["ours"])
            row_images.append(d["gt"])
            
            for c in range(cols):
                ax = axes[r, c]
                if c == 1:
                    ax.imshow(row_images[c], cmap="gray")
                else:
                    ax.imshow(row_images[c])
                
                # Remove all ticks and spines to create edge-to-edge look
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                    
                # Add titles ONLY to the very bottom row, just like the reference paper
                if r == n_rows - 1:
                    ax.set_xlabel(col_labels[c], fontsize=16, labelpad=10)

        out_path = os.path.join(paths["vis"], "publication_grid.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # ---------- Results Table ----------
    table = Table(title="Quantitative Analysis: Shadow Removal Performance", box=box.HEAVY_HEAD)
    table.add_column("Configuration", style="cyan")
    table.add_column("PSNR (dB) ↑", justify="center", style="green")
    table.add_column("SSIM ↑", justify="center", style="green")
    table.add_column("ID Error ↓", justify="center", style="yellow")

    def avg(m, key):
        return m[key] / m["n"] if m["n"] > 0 else 0

    if metrics["baseline"]["n"] > 0:
        table.add_row(
            "Baseline (StyleGAN)",
            f"{avg(metrics['baseline'], 'psnr'):.2f}",
            f"{avg(metrics['baseline'], 'ssim'):.4f}",
            f"{avg(metrics['baseline'], 'id'):.5f}"
        )
    else:
        table.add_row("Baseline (StyleGAN)", "-", "-", "-")

    table.add_row(
        "Ours (U-Net Restoration)",
        f"{avg(metrics['ours'], 'psnr'):.2f}",
        f"{avg(metrics['ours'], 'ssim'):.4f}",
        f"{avg(metrics['ours'], 'id'):.5f}"
    )

    console.print("\n")
    console.print(table)
    console.print("\nVisuals saved to /visual_comparisons/")

if __name__ == "__main__":
    evaluate_metrics()