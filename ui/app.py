import streamlit as st                   
import cv2                                 
import numpy as np                         
import os                                 
import sys                                 
from PIL import Image                      
import torch                               
import torchvision.transforms as transforms  

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import trained U-Net model
try:
    from src.train_unet import ShadowRemoverUNet
except ImportError:
    st.error("Failed to import ShadowRemoverUNet from src.train_unet. Please ensure project structure is intact.")
    st.stop()

from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide", page_title="EE655 Custom UNet Predictor")

# Caching the model to avoid reloading 
@st.cache_resource
def load_custom_unet():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner("Fetching weights from Hugging Face..."):
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id="harshhitan/interactive-shadow-remover", filename="best_shadow_unet.pth")
            
    with st.spinner("Initializing Neural Architecture (this may take a minute)..."):
        model = ShadowRemoverUNet()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
    return model, device

st.title("Interactive Portrait Shadow Removal")
st.markdown("Upload a portrait, paint over the shadowed region, and our custom **4-Channel U-Net** will instantly recover the true facial features underneath.")

model_data = load_custom_unet()

if model_data is None:
    st.error(f"**Model Checkpoint Not Found!**\nYou must train the model first by running:\n`!python src/train_unet.py --epochs 20` inside your Colab notebook before using the UI.")
    st.stop()
model, device = model_data
col1, col2 = st.columns(2)

# LEFT SIDE (INPUT + DRAWING)
with col1:
    uploaded_file = st.file_uploader("Upload Shadowed Portrait", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_raw = Image.open(uploaded_file).convert("RGB")
        image = image_raw.resize((256, 256))
        st.markdown("### Your Portrait")
        st.image(image, use_column_width=True)

        st.markdown("### Draw Shadow Mask Below")
        st.caption("Paint white over the shadow regions. Use the face above as reference.")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=20,
            stroke_color="white",
            background_color="#000000",
            update_streamlit=True,
            height=256,
            width=256,
            drawing_mode="freedraw",
            key="canvas",
        )

# RIGHT SIDE (OUTPUT)
with col2:
    if uploaded_file is not None:
        st.markdown("### 2. Custom Model Output")
        if st.button("Run UNet Inference (Instant)", type="primary"):

            # Extract binary mask from canvas
            if canvas_result.image_data is not None:
                mask_data = np.array(canvas_result.image_data, dtype=np.uint8)
                mask_alpha = mask_data[:, :, 3]   
                binary_mask = (mask_alpha > 5).astype(np.uint8) * 255
                mask_pil = Image.fromarray(binary_mask, mode="L")
            else:
                binary_mask = np.zeros((256, 256), dtype=np.uint8)
                mask_pil = None  

            # Ensure user has drawn mask
            if mask_pil is None or np.sum(binary_mask) == 0:
                st.warning("Please draw a mask over the shadow first!")
            else:
                with st.spinner("Executing Forward Pass on local U-Net..."):
                    
                    # Convert images to tensors
                    transform_fn = transforms.Compose([
                        transforms.ToTensor()
                    ])
                    img_tensor = transform_fn(image)
                    mask_tensor = transform_fn(mask_pil)
                    
                    # Create 4-channel input (RGB + mask)
                    input_tensor = torch.cat((img_tensor, mask_tensor), dim=0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output_tensor = model(input_tensor)
                    
                    # Convert output to image format
                    output_np = output_tensor.squeeze(0).cpu().numpy()
                    output_np = np.transpose(output_np, (1, 2, 0))
                    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.float32)

                    # Smooth mask edges
                    soft_mask = cv2.GaussianBlur(binary_mask, (41, 41), 0).astype(np.float32) / 255.0
                    soft_mask = np.expand_dims(soft_mask, -1)

                    # Blend model output with original image
                    original_np = np.array(image).astype(np.float32)
                    blended_np = (output_np * soft_mask) + (original_np * (1.0 - soft_mask))
                    
                    blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)
                    result_image = Image.fromarray(blended_np).resize((512, 512))
                
                # Show result
                st.success("High-Fidelity Alpha-Blended Inference Complete!")
                st.image(result_image, caption="U-Net Image Translation Output", use_column_width=True)