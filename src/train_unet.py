import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import argparse
from torchvision.models import vgg16, VGG16_Weights

# Check for crucial Face-specific dependency
try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    print("CRITICAL: facenet_pytorch is missing. Please run `!pip install facenet-pytorch` in Colab.")
    sys.exit(1)
    
# VGG16 Perceptual Loss
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval().to(device)
        self.slice = nn.Sequential(*list(vgg.children())[:16])
        for param in self.slice.parameters():
            param.requires_grad = False
            
    def forward(self, generated_img, target_img):
        # Normalize inputs as required by VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(generated_img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(generated_img.device)
        gen_norm = (generated_img - mean) / std
        target_norm = (target_img - mean) / std
        gen_f = self.slice(gen_norm)
        with torch.no_grad():
            target_f = self.slice(target_norm).detach()
        return nn.functional.mse_loss(gen_f, target_f)

# Basic convolution block used in decoder
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# Main Model (FaceNet Encoder + U-Net Decoder)
class ShadowRemoverUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained FaceNet encoder (captures facial features)
        self.face_encoder = InceptionResnetV1(pretrained='vggface2').eval()
        for param in self.face_encoder.parameters():
            param.requires_grad = False

        # Small network to process mask before merging    
        self.mask_processor = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 3, kernel_size=1)
        )

        # Decoder (U-Net style with skip connections)
        self.upconv3 = nn.ConvTranspose2d(896, 256, kernel_size=2, stride=2) 
        self.decoder3 = DoubleConv(256 + 256, 256) # 256 (upconv) + 256 (x3)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128 + 64, 128) # 128 (upconv) + 64 (x2)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(64 + 32, 64) # 64 (upconv) + 32 (x1)
        
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Split 4-channel input -> RGB image + mask
        img = x[:, :3, :, :]  
        mask = x[:, 3:4, :, :]

        # Inject mask information into image
        processed_mask = self.mask_processor(mask)
        input_face = img + (processed_mask * 0.1) 
        
        # Encoder (FaceNet feature extraction)
        x1 = self.face_encoder.conv2d_1a(input_face) 
        x = self.face_encoder.conv2d_2a(x1)
        x2 = self.face_encoder.conv2d_2b(x)          
        x = self.face_encoder.maxpool_3a(x2)
        x = self.face_encoder.conv2d_3b(x)
        x = self.face_encoder.conv2d_4a(x)
        x3 = self.face_encoder.conv2d_4b(x)          
        x = self.face_encoder.repeat_1(x3)
        bottleneck = self.face_encoder.mixed_6a(x)   

        # Fix size mismatch before concatenation
        def pad_to_match(tensor, target_tensor):
            diffY = target_tensor.size()[2] - tensor.size()[2]
            diffX = target_tensor.size()[3] - tensor.size()[3]
            return nn.functional.pad(tensor, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Decoder Path
        d3 = self.upconv3(bottleneck) # [B, 256, 28, 28]
        d3 = pad_to_match(d3, x3)
        d3 = torch.cat((d3, x3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3) 
        d2 = pad_to_match(d2, x2)
        d2 = torch.cat((d2, x2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2) 
        d1 = pad_to_match(d1, x1)
        d1 = torch.cat((d1, x1), dim=1)
        d1 = self.decoder1(d1)
        
        # Final reconstruction to 256×256 RGB image
        out = self.final_up(d1)
        out = nn.functional.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
        out = self.final_conv(out)
        return self.sigmoid(out)


# Dataset Loader (handles input, target, and mask)
class ShadowDataset(Dataset):
    def __init__(self, shadow_dir, mask_dir, clean_dir, img_size=256):
        self.shadow_dir = shadow_dir
        self.mask_dir = mask_dir
        self.clean_dir = clean_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(shadow_dir) if f.lower().endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        shadow_path = os.path.join(self.shadow_dir, img_name)
        shadow_img = Image.open(shadow_path).convert("RGB")
        
        clean_name = img_name.replace("_shadow", "") 
        clean_path = os.path.join(self.clean_dir, clean_name)
        if not os.path.exists(clean_path): clean_path = os.path.join(self.clean_dir, img_name)
        clean_img = Image.open(clean_path).convert("RGB")
        
        mask_name = img_name.split('.')[0] + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
        else:
            mask_img = Image.new('L', (shadow_img.size[0], shadow_img.size[1]), 0)

        # Resize all inputs
        shadow_img = shadow_img.resize((self.img_size, self.img_size))
        clean_img = clean_img.resize((self.img_size, self.img_size))
        mask_img = mask_img.resize((self.img_size, self.img_size))

        # Apply same augmentation to maintain alignment
        if random.random() > 0.5:
            shadow_img = TF.hflip(shadow_img)
            clean_img = TF.hflip(clean_img)
            mask_img = TF.hflip(mask_img)
            
        angle = random.uniform(-15, 15)
        shadow_img = TF.rotate(shadow_img, angle)
        clean_img = TF.rotate(clean_img, angle)
        mask_img = TF.rotate(mask_img, angle)

        color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)

        # Slight variation only on input image
        shadow_tensor = TF.to_tensor(color_jitter(shadow_img))
        clean_tensor = TF.to_tensor(clean_img)
        mask_tensor = TF.to_tensor(mask_img)

        # Create 4-channel input (RGB + mask)
        input_tensor = torch.cat((shadow_tensor, mask_tensor), dim=0)
        return input_tensor, clean_tensor

# Training Loop
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}")
    dataset = ShadowDataset(args.shadow_dir, args.mask_dir, args.clean_dir, img_size=256)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = ShadowRemoverUNet().to(device)
    criterion_l1 = nn.L1Loss() 

    try:
        perceptual_loss_fn = VGGPerceptualLoss(device)
    except Exception as e:
        print(f"Falling back to pure L1 due to error: {e}")
        perceptual_loss_fn = None
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler for smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Starting Training for {args.epochs} Epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        loop = tqdm(dataloader, leave=False)
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Combined loss (pixel + perceptual)
            loss_pixels = criterion_l1(outputs, targets)
            if perceptual_loss_fn is not None:
                loss_perceptual = perceptual_loss_fn(outputs, targets)
                loss = loss_pixels + (0.1 * loss_perceptual)
            else:
                loss = loss_pixels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] -> Average Compound Loss: {avg_loss:.4f}")

        # Save Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_shadow_unet.pth'))
            print(">>> Saved New Best Model!")
            
# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow_dir", type=str, default="data/synthetic_shadows/images")
    parser.add_argument("--mask_dir", type=str, default="data/synthetic_shadows/masks")
    parser.add_argument("--clean_dir", type=str, default="data/synthetic_shadows/clean") 
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    args = parser.parse_args()
    
    train(args)