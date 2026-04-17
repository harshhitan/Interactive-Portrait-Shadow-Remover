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

from src.train_unet import ShadowRemoverUNet
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide", page_title="Interactive Portrait Shadow Removal")

@st.cache_resource
def load_custom_unet():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id="harshhitan/interactive-shadow-remover", filename="best_shadow_unet.pth")
    model = ShadowRemoverUNet()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

st.title("Interactive Portrait Shadow Removal")
st.markdown("Upload a portrait, paint over the shadowed region, and our custom **4-Channel U-Net** will instantly recover the true facial features underneath.")

model, device = load_custom_unet()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Shadowed Portrait", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_raw = Image.open(uploaded_file).convert("RGB")
        image = image_raw.resize((256, 256))
        # Convert to RGBA specifically for streamlit-drawable-canvas robustness
        display_image = image_raw.resize((512, 512)).convert("RGB")
        
        st.markdown("### 1. Highlight the Shadow Area")
        st.caption("Draw over the shadow. The red mask helps you see the underlying image.")

        # Dynamically generate key based on file to prevent caching issues in st_canvas
        canvas_key = f"canvas_{uploaded_file.name}-{uploaded_file.size}"

        st.image(display_image, use_column_width=True)

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=20,
            stroke_color="rgba(255, 0, 0, 0.6)",
            background_color="rgba(0,0,0,0)",  # transparent canvas
            height=512,
            width=512,
            drawing_mode="freedraw",
            key=canvas_key,
        )

with col2:
    if uploaded_file is not None:
        st.markdown("### 2. Shadow Removed Output")
        if st.button("Run Shadow Removal", type="primary"):

            if canvas_result.image_data is not None:
                mask_data = np.array(canvas_result.image_data, dtype=np.uint8)
                mask_alpha = mask_data[:, :, 3]
                binary_mask_512 = (mask_alpha > 5).astype(np.uint8) * 255
                mask_pil_512 = Image.fromarray(binary_mask_512, mode="L")
                mask_pil = mask_pil_512.resize((256, 256), Image.NEAREST)
                binary_mask = np.array(mask_pil)
            else:
                binary_mask = np.zeros((256, 256), dtype=np.uint8)
                mask_pil = None

            if mask_pil is None or np.sum(binary_mask) == 0:
                st.warning("Please draw a mask over the shadow first!")
            else:
                with st.spinner("Running U-Net inference..."):
                    transform_fn = transforms.Compose([transforms.ToTensor()])
                    img_tensor = transform_fn(image)
                    mask_tensor = transform_fn(mask_pil)

                    input_tensor = torch.cat((img_tensor, mask_tensor), dim=0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output_tensor = model(input_tensor)

                    output_np = output_tensor.squeeze(0).cpu().numpy()
                    output_np = np.transpose(output_np, (1, 2, 0))
                    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.float32)

                    soft_mask = cv2.GaussianBlur(binary_mask, (41, 41), 0).astype(np.float32) / 255.0
                    soft_mask = np.expand_dims(soft_mask, -1)

                    original_np = np.array(image).astype(np.float32)
                    blended_np = (output_np * soft_mask) + (original_np * (1.0 - soft_mask))
                    blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)
                    result_image = Image.fromarray(blended_np).resize((512, 512))

                st.success("Shadow Removal Complete!")
                st.image(result_image, caption="U-Net Output", use_column_width=True)