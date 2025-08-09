import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import emoji
import re
import shap
from ui_config import apply_femsafe_theme, render_sidebar

# -------------------------------
# 🔧 CONFIGURATION
# -------------------------------
MODEL_PATH = "distilbert_model_v2/"
LABELS = ["Safe", "At Risk", "Immediate Danger"]
PANIC_THRESHOLD = 2
LOG_FILE = "panic_log.txt"

# -------------------------------
# 🧼 TEXT PREPROCESSING
# -------------------------------
def preprocess_text(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    emoji_map = {
        ":crying_face:": ":distress:",
        ":broken_heart:": ":distress:",
        ":face_with_symbols_on_mouth:": ":anger:",
        ":sos_button:": ":panic:",
        ":knife:": ":threat:",
        ":skull:": ":danger:",
        ":face_with_tears_of_joy:": ":masking_pain:"
    }
    for raw, norm in emoji_map.items():
        text = text.replace(raw, norm)
    return re.sub(r"\s+", " ", text).strip()

# -------------------------------
# 📁 PANIC LOGGING
# -------------------------------
def log_panic_event(text, label):
    with open(LOG_FILE, "a") as f:
        f.write(f"[PANIC] {label} | {text}\n")

# -------------------------------
# 📦 LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# 🧠 SHAP EXPLAINER
# -------------------------------
def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, (np.ndarray, torch.Tensor)):
        texts = [str(t) for t in texts]
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encodings)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.numpy()

masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict_proba, masker)

# -------------------------------
# 🌐 STREAMLIT UI
# -------------------------------
apply_femsafe_theme()
render_sidebar()

st.title("🛡️ FemSafe GBV Risk Detector")
st.markdown("Enter a message or report below to assess femicide/GBV risk. Emojis are supported.")

# -------------------------------
# ✍️ SINGLE TEXT INPUT
# -------------------------------
user_input = st.text_area("✍️ Type a message or report:", height=150)

if user_input.strip():
    clean_text = preprocess_text(user_input)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.squeeze().tolist()
    pred_idx = int(torch.argmax(outputs.logits, dim=1).item())

    # 🔍 Show Preprocessed Text
    with st.expander("🔍 View Preprocessed Text"):
        st.code(clean_text, language="text")

    # 🚨 PANIC BUTTON
    if pred_idx == PANIC_THRESHOLD:
        st.error("🚨 Immediate Danger Detected!")
        if st.button("🔴 Trigger Panic Response"):
            st.warning("Panic protocol activated. Authorities or support services will be notified.")
            log_panic_event(user_input, LABELS[pred_idx])

    # 📊 RISK LEVEL DISPLAY
    st.subheader("🔎 Risk Assessment")
    for i, label in enumerate(LABELS):
        st.write(f"**{label}**")
        st.progress(min(max(logits[i], 0), 1))
    st.markdown(f"### 🧾 **Predicted Risk Level:** `{LABELS[pred_idx]}`")

    # 🧠 SHAP EXPLANATION
    st.subheader("🧠 Model Explanation (SHAP)")
    shap_values = explainer([clean_text])
    st.write("SHAP values (token-level impact):")
    st.json(shap_values.data)

    # 📚 SHAP Visualization
    st.subheader("📚 SHAP Visualization")
    shap.plots.text(shap_values[0])

# -------------------------------
# 📁 BATCH UPLOAD
# -------------------------------
st.markdown("---")
st.header("📁 Batch Triage")
uploaded_file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        st.success(f"Loaded {len(df)} reports.")
        results = []
        for text in df["text"]:
            clean = preprocess_text(str(text))
            inputs = tokenizer(clean, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits.squeeze().tolist()
            pred = int(torch.argmax(outputs.logits, dim=1).item())
            results.append({
                "Original Text": text,
                "Cleaned Text": clean,
                "Predicted Risk": LABELS[pred],
                "Immediate Danger": "🚨" if pred == PANIC_THRESHOLD else ""
            })
        st.dataframe(pd.DataFrame(results))

# -------------------------------
# 📊 SHAP SUMMARY (Optional)
# -------------------------------
st.markdown("---")
with st.expander("📊 SHAP Summary Plot (Experimental)"):
    try:
        shap.summary_plot(shap_values.values, feature_names=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
    except Exception as e:
        st.warning("Summary plot not available for this input.")