# auto_hallucination_detector.py
import streamlit as st
from PIL import Image
import torch
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # Load BLIP (captioning model)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    return clip_model, clip_preprocess, blip_model, blip_processor, device

clip_model, clip_preprocess, blip_model, blip_processor, device = load_models()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Hallucination Detector", page_icon="🧠", layout="centered")
st.title("🧠 AI Multimodal Hallucination Detector")
st.markdown("Upload an image. The AI will generate a caption and check if it **matches** the image or if it is a **hallucination**.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        # Generate caption with BLIP
        inputs = blip_processor(image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_new_tokens=20)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

    st.subheader("Caption Mode")
    mode = st.radio("Choose how to get the caption:", ["AI Generated (BLIP)", "Custom (User Input)"])

    caption = None
    if mode == "AI Generated (BLIP)":
        with st.spinner("Generating caption with BLIP..."):
            inputs = blip_processor(image, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs, max_new_tokens=20)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
        st.success(f"📝 AI Generated Caption: **{caption}**")

    else:
        caption = st.text_input("✍️ Enter your custom caption here")

    with st.spinner("Analyzing consistency with CLIP..."):
        # Preprocess image + caption for CLIP
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize([caption]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Similarity score
        similarity = float((image_features @ text_features.T).cpu().numpy()[0][0])
        
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider("Set Confidence Threshold (%)", min_value=10, max_value=90, value=30)

    confidence = similarity * 100 

    st.subheader("🔍 Results")
    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

    if similarity > 0.30:
        st.success(f"✅ The caption matches the image (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"⚠️ Possible Hallucination (Confidence: {confidence:.2f}%)")
