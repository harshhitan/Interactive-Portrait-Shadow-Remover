import os
from pathlib import Path
import random
import time

try:
    from datasets import load_dataset
except ImportError:
    print("CRITICAL: 'datasets' package missing. Run `!pip install datasets` in Colab.")
    exit(1)

def download_dataset_safely(dataset_name, output_dir, num_samples=250, prefix="img"):
    print(f"\nFetching {num_samples} images from '{dataset_name}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load streaming dataset using the endpoints that have been proven to work
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= num_samples:
                break
                
            # Safely extract image matrix regardless of column naming schema
            img = item.get('image') or item.get('img')
            if img:
                # Save out to designated directory
                img.save(output_dir / f"{prefix}_{count:04d}.png")
                count += 1
                
            if count % 50 == 0:
                print(f"  -> Downloaded {count}/{num_samples} images...")
                
        print(f"✅ Successfully completed! Saved {count} samples from {dataset_name}.")
        return count
    except Exception as e:
        print(f"❌ Failed to fetch from {dataset_name}. Error: {e}")
        return 0

def download_images():
    base_dir = Path(__file__).parent.parent
    output_ffhq = base_dir / 'data' / 'ffhq_samples'
    output_celeba = base_dir / 'data' / 'celeba_hq_samples'
    
    print("Initializing Robust Dataset Collection (Target: 500 Faces)...")
    
    # Target 1: Verified FFHQ Distribution
    # ('bitmind/ffhq-256' or 'marcosv/ffhq-dataset')
    download_dataset_safely("bitmind/ffhq-256", output_ffhq, num_samples=250, prefix="ffhq")
    
    # Target 2: Verified CelebA-HQ Distribution
    download_dataset_safely("korexyz/celeba-hq-256x256", output_celeba, num_samples=250, prefix="celeba")
            
    total = len(list(output_ffhq.glob("*.png"))) + len(list(output_celeba.glob("*.png")))
    total += len(list(output_ffhq.glob("*.jpg"))) + len(list(output_celeba.glob("*.jpg")))
    print(f"\nDATA COLLECTION COMPLETE! Total combined portraits ready for synthesis: {total}")

if __name__ == "__main__":
    download_images()
