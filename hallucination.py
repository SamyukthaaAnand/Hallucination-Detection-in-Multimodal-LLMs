# auto_hallucination_detector.py
import streamlit as st
from PIL import Image
import torch
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # Load BLIP (captioning model)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    # Semantic similarity model
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    return clip_model, clip_preprocess, blip_model, blip_processor, semantic_model, device

clip_model, clip_preprocess, blip_model, blip_processor, semantic_model, device = load_models()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Hallucination Detector", page_icon="🧠", layout="centered")
st.title("🧠 AI Multimodal Hallucination Detector")
st.markdown("Upload an image. The system will check if your caption is consistent with AI-generated caption.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------- Caption Mode ----------------
    st.subheader("Caption Mode")
    mode = st.radio("Choose caption type:", ["AI Generated (BLIP)", "Custom (User Input)"])

    # Generate BLIP caption
    with st.spinner("Generating AI caption..."):
        inputs = blip_processor(image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_new_tokens=50)
        blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)

    if mode == "AI Generated (BLIP)":
        user_caption = blip_caption
        st.success(f"📝 AI Generated Caption: **{user_caption}**")
    else:
        user_caption = st.text_input("✍️ Enter your caption here")
        if not user_caption:
            st.warning("Please enter a caption to proceed.")
            st.stop()

    # ---------------- CLIP Similarity (caption vs caption) ----------------
    text_input = clip.tokenize([user_caption, blip_caption]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity between the two text vectors safely
    clip_similarity = torch.cosine_similarity(text_features[0:1], text_features[1:2]).item()

    # ---------------- Semantic Similarity ----------------
    embeddings = semantic_model.encode([user_caption, blip_caption], convert_to_tensor=True)
    semantic_similarity = util.cos_sim(embeddings[0:1], embeddings[1:2]).item()

    # ---------------- Combined Confidence ----------------
    combined_confidence = 0.7 * clip_similarity + 0.3 * semantic_similarity
    combined_confidence = min(combined_confidence, 1.0)
    confidence_percent = combined_confidence * 100

    # ---------------- Results ----------------
    st.subheader("🔍 Results")
    st.metric(label="Confidence Score", value=f"{confidence_percent:.2f}%")

    # ---------------- Hallucination Check ----------------
    threshold = 0.7  # default threshold for hallucination detection
    if combined_confidence > threshold:
        st.success(f"✅ The caption matches the image (Confidence: {confidence_percent:.2f}%)")
    else:
        st.error(f"⚠️ Possible Hallucination (Confidence: {confidence_percent:.2f}%)")
