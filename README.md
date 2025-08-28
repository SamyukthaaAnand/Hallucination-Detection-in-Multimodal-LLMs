# 🧠 Hallucination Detection in Multimodal LLMs

This project implements an **AI-powered hallucination detector** for multimodal models.  
It checks whether an **image-caption pair** is consistent or potentially hallucinated, using a combination of **BLIP (image captioning), CLIP (vision-language similarity) and semantic similarity models**.

---

## 📌 Features
- 🖼️ Upload an image and test AI-generated or custom captions.  
- 🤖 Generates captions automatically using **BLIP**.  
- 🔗 Measures similarity between user/AI captions with **CLIP** and **Sentence Transformers**.  
- 📊 Outputs a **confidence score** for consistency.  
- ⚠️ Flags possible **hallucinations** when captions do not align with the image.  
- 🌐 Streamlit-based interactive web app.  

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – UI for interaction  
- [PyTorch](https://pytorch.org/) – Deep learning framework  
- [CLIP](https://github.com/openai/CLIP) – Vision-language model  
- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) – Image captioning  
- [Sentence Transformers](https://www.sbert.net/) – Semantic similarity  

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SamyukthaaAnand/Hallucination-Detection-in-Multimodal-LLMs.git
   cd Hallucination-Detection-in-Multimodal-LLMs
2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
---

## ▶️ Usage

Run the Streamlit app:
   ```bash
   streamlit run hallucination.py
   ```
---

## 🧪 Example Workflow

1. Upload an image (JPG/PNG).  
2. Choose caption type:  
   - **AI Generated** (via BLIP)  
   - **Custom** (enter your own)  
3. The system compares **caption vs. image** using CLIP + semantic similarity.  
4. A **confidence score** is displayed:  
   - ✅ High score → Caption likely matches image.  
   - ⚠️ Low score → Possible hallucination.  

---

## 📜 License

MIT License – feel free to use and modify for research purposes.  
