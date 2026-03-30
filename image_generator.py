import os
import torch
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import gc
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load

# --- Configuration  for SDXL and LayerDiffusion ---
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_CONTROLNET_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"
LAYER_DIFFUSION_REPO = "LayerDiffusion/layerdiffusion-v1"
LAYER_DIFFUSION_LORA_NAME = "layer_xl_transparent_attn.safetensors"
LAYER_DIFFUSION_VAE_NAME = "vae_transparent_decoder.safetensors"


# --- Utility Functions ---

def match_reference_image(ref_img_dir, prompt_text=""):
    """Dynamically selects the best reference image based on activity in the prompt."""
    prompt_text = prompt_text.lower()
    valid_images = [f for f in os.listdir(ref_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not valid_images:
        # In a stable environment, you'd handle this better, but for testing:
        print(f"❌ No image files found in {ref_img_dir}. Cannot proceed.")
        return None 

    # --- MATCHING LOGIC (Using the logic from your provided cell) ---
    match_key, activity_name = None, None
    if "running" in prompt_text or "runner" in prompt_text:
        match_key, activity_name = "run", "running"
    elif "walking" in prompt_text or "walk" in prompt_text:
        match_key, activity_name = "walk", "walking"
    elif "standing" in prompt_text:
        match_key, activity_name = "standing", "standing"
    elif "respiratory" in prompt_text:
        match_key, activity_name = "respirator", "Respiratory"
    elif "stand" in prompt_text:
        match_key, activity_name = "stand", "stand"
    else:
        print("⚠️ No strong activity match found. Using first available image as fallback.")
        return os.path.join(ref_img_dir, valid_images[0])

    for file in valid_images:
        if match_key in file.lower():
            if match_key == "stand" and "standing" in file.lower():
                continue
            print(f"🎯 Matched activity: '{activity_name}' to file {file}")
            return os.path.join(ref_img_dir, file)

    print("⚠️ Fallback: No matching image found for detected activity. Using first available image.")
    return os.path.join(ref_img_dir, valid_images[0])


def make_canny_image(image_path: str, size=(1024, 1024)) -> Image.Image:
    """Converts a reference image into a Canny edge map for ControlNet."""
    # Dummy image creation for environments without CV2/image files (VS Code only)
    if not os.path.exists(image_path):
        print(f"⚠️ DUMMY: Generating black canvas for Canny since {image_path} not found.")
        return Image.fromarray(np.zeros((*size, 3), dtype=np.uint8))
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Failed to read reference image: {image_path}")
        
    image_resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edge_rgb)

# ==========================================================
# === LOAD ALL MODELS ONCE (STABLE MEMORY VERSION) ===
# ==========================================================

def load_pipeline_and_models(device="cuda"):
    """Loads and configures the SDXL/ControlNet/LayerDiffusion pipeline ONCE."""

    print("🔄 Loading models: SDXL, ControlNet, and LayerDiffusion...")
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load SDXL ControlNet - IMPORTANT: Remove low_cpu_mem_usage
    controlnet = ControlNetModel.from_pretrained(
        SDXL_CONTROLNET_MODEL_ID,
        torch_dtype=dtype,
    )

    # Load SDXL ControlNet Pipeline - IMPORTANT: Remove low_cpu_mem_usage
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # 🌟 CRITICAL MEMORY FIX: Load everything directly to the device (VRAM)
    if device == "cuda":
        # Ensure we are not using CPU offload for stability
        pipe.to(device)
    else:
        pipe.to(device)

    # 1. Load the LayerDiffusion LoRA
    print("  - Loading LayerDiffusion LoRA...")
    lora_path = hf_hub_download(LAYER_DIFFUSION_REPO, LAYER_DIFFUSION_LORA_NAME)
    old_state_dict = safe_load(lora_path)
    pipe.unet.load_state_dict(old_state_dict, strict=False)
    print("  - LoRA weights applied to UNet.")

    # 2. Load the LayerDiffusion VAE Decoder
    print("  - Loading LayerDiffusion VAE Decoder...")
    vae_decoder_path = hf_hub_download(LAYER_DIFFUSION_REPO, LAYER_DIFFUSION_VAE_NAME)
    vae_state_dict = safe_load(vae_decoder_path)
    pipe.vae.load_state_dict(vae_state_dict, strict=False)
    pipe.vae.to(pipe.device).to(dtype)

    print("LayerDiffusion components loaded successfully. Ready for generation loop.")
    return pipe, device


# ==========================================================
# === GENERATE AND SAVE ===
# ==========================================================

def generate_image_and_save(pipe, prompt: str, ref_img_dir: str, output_dir: str, title: str):
    """Generates a single image and saves it with a unique name."""

    negative_prompt = (
        "blurry, low quality, disfigured, unrealistic anatomy, opaque background, cutout, border, jpeg artifacts, "
        "jagged edges, pixelated, blurry outlines, (bright highlights):1.5, (intense color):1.5,"
        "unwanted light bleed, blurrsy focus, showing glow on hands, green color on head and hands, blue color on head and hands, orange color on hand and hands"
    )
    
    # --- Image Optimization Settings (PNG) ---
    MAX_DIM = 480 

    os.makedirs(output_dir, exist_ok=True)

    # --- Pre-processing ---
    ref_image_path = match_reference_image(ref_img_dir, prompt)
    if ref_image_path is None:
        print("🛑 Skipping generation due to missing reference image path.")
        return None
        
    print(f"📸 Using reference image: {ref_image_path}")
    control_image = make_canny_image(ref_image_path, size=(1024, 1024))

    # --- Image Generation ---
    print(f"🎨 Generating image for: {title}...")
    output = pipe(
        prompt=prompt,
        image=control_image,
        negative_prompt=negative_prompt,
        num_inference_steps=15,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        width=1024,
        height=1024,
    )
    generated_img = output.images[0] 
    
    # --- 🌟 PNG COMPRESSION STEP 🌟 ---
    resized_img = generated_img.resize((MAX_DIM, MAX_DIM)) 

    # 2. Define the output path for the optimized PNG
    clean_title = title.replace(" ", "_").replace(",", "").replace(":", "").replace("__", "_")
    out_path_optimized = os.path.join(output_dir, f"gait_insight_{clean_title}_{datetime.now().strftime('%M%S')}.png")
    
    # 3. Save as optimized PNG
    resized_img.save(
        out_path_optimized,
        format="PNG",
        optimize=True
    )

    print(f"✅ Image optimized ({MAX_DIM}x{MAX_DIM}) and saved as PNG at {out_path_optimized}")
    
    # --- CRITICAL: Aggressive Memory Cleanup ---
    try:
        del output
        del generated_img
        del resized_img
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("    -> Memory cleanup complete.")

    except Exception as e:
        print(f"Warning: Failed to perform aggressive memory cleanup. Error: {e}")

    return out_path_optimized
    

 # ... [Your entire existing code remains unchanged above this line] ...

# ==========================================================
# === ADDITIONAL: Minimal Flask server to run app on port 8000 ===
# ==========================================================
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global pipe variable (reuse your pipeline if needed)
pipe, device = load_pipeline_and_models(device="cuda" if torch.cuda.is_available() else "cpu")

@app.route("/generate", methods=["POST"])
def generate_endpoint():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        ref_dir = data.get("ref_img_dir", "./ref_images")
        output_dir = data.get("output_dir", "./outputs")
        title = data.get("title", "default_title")
        
        result_path = generate_image_and_save(pipe, prompt, ref_dir, output_dir, title)
        if result_path:
            return jsonify({"status": "success", "file": result_path})
        else:
            return jsonify({"status": "failed", "message": "No image generated"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Run Flask app on all interfaces so EC2 can access it
    app.run(host="0.0.0.0", port=8000)
