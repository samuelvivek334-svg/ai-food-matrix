import os
import io
import json
from typing import Optional

import streamlit as st
from PIL import Image
import numpy as np

try:
    from transformers import pipeline
except Exception:
    pipeline = None

st.set_page_config(page_title="AI Food Matrix | Indian Diet Bio-Predictor", page_icon="🍛", layout="wide")

CUSTOM_CSS = """
<style>
.main {background: #fffaf4;}
.block-container {padding-top: 1.6rem; max-width: 1180px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 22px; background: linear-gradient(120deg,#2e251f,#6e3e24); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:2.1rem;}
.hero p {color:#f9e9d8; font-size:1rem; margin:.5rem 0 0;}
.metric-card {padding:1rem; border:1px solid #ead9c8; border-radius:16px; background:white;}
.small-note {font-size:.88rem; color:#6b625c;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🍛 AI Food Matrix</h1>
  <p>Indian Diet Bio-Predictor · Upload a food photo, identify the likely dish, and explore nutrition estimates.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Research prototype: a photo cannot reliably reveal exact ingredients, portion weight, recipe, or glycemic index. "
    "Nutrition values are rough estimates. This app does not measure blood glucose or provide medical advice."
)

@st.cache_resource(show_spinner="Loading image recognition model (first run may take a few minutes)...")
def load_classifier():
    if pipeline is None:
        raise RuntimeError("Transformers is not installed. Install requirements.txt and restart the app.")
    # General-purpose zero-shot image classifier; candidate labels include common Indian dishes.
    return pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )

# Approximate nutrition references per 100 g of prepared food. Values are illustrative,
# broad estimates for UI demonstration—not a validated nutrient database or measured GI labels.
# Must be replaced with source-verified recipe data before research/public use.
FOOD_DATA = {
    "idli": {"kcal": 146, "carbs": 29.0, "protein": 4.5, "fat": 1.0, "fiber": 2.0, "gi": None, "note": "Varies by rice:urad ratio, fermentation, size and recipe."},
    "dosa": {"kcal": 184, "carbs": 27.0, "protein": 4.5, "fat": 6.0, "fiber": 1.5, "gi": None, "note": "Varies by batter, oil, thickness and fillings."},
    "plain rice": {"kcal": 130, "carbs": 28.0, "protein": 2.7, "fat": 0.3, "fiber": 0.4, "gi": None, "note": "Varies substantially by rice variety and cooking."},
    "chapati": {"kcal": 297, "carbs": 50.0, "protein": 9.0, "fat": 7.0, "fiber": 7.0, "gi": None, "note": "Values depend on flour, oil/ghee and thickness."},
    "poha": {"kcal": 180, "carbs": 30.0, "protein": 3.5, "fat": 5.0, "fiber": 2.0, "gi": None, "note": "Varies with rice flakes, peanuts, vegetables and portion."},
    "upma": {"kcal": 150, "carbs": 24.0, "protein": 4.0, "fat": 4.0, "fiber": 2.0, "gi": None, "note": "Varies by semolina, oil and vegetables."},
    "pongal": {"kcal": 165, "carbs": 25.0, "protein": 4.5, "fat": 5.0, "fiber": 2.0, "gi": None, "note": "Varies by rice/dal ratio and ghee."},
    "biryani": {"kcal": 180, "carbs": 24.0, "protein": 7.0, "fat": 6.0, "fiber": 1.5, "gi": None, "note": "Large recipe variation; meat/vegetables/oil and rice type matter."},
    "dal": {"kcal": 116, "carbs": 20.0, "protein": 7.5, "fat": 1.5, "fiber": 7.0, "gi": None, "note": "Depends on lentil type, thickness and added oil."},
    "sambar": {"kcal": 65, "carbs": 9.0, "protein": 3.0, "fat": 2.0, "fiber": 3.0, "gi": None, "note": "Varies with dal, vegetables and recipe."},
    "paneer curry": {"kcal": 190, "carbs": 8.0, "protein": 10.0, "fat": 14.0, "fiber": 2.0, "gi": None, "note": "Varies with paneer quantity, gravy, cream and oil."},
    "chicken curry": {"kcal": 160, "carbs": 5.0, "protein": 17.0, "fat": 8.0, "fiber": 1.0, "gi": None, "note": "Varies by cut, gravy, skin and oil."},
    "chole": {"kcal": 165, "carbs": 22.0, "protein": 8.0, "fat": 5.0, "fiber": 7.0, "gi": None, "note": "Varies by chickpea amount, gravy and added oil."},
    "samosa": {"kcal": 300, "carbs": 32.0, "protein": 5.0, "fat": 17.0, "fiber": 3.0, "gi": None, "note": "Varies by size, filling and frying oil."},
    "vada": {"kcal": 280, "carbs": 28.0, "protein": 8.0, "fat": 15.0, "fiber": 4.0, "gi": None, "note": "Varies by dal type, size and absorbed oil."},
    "parotta": {"kcal": 320, "carbs": 47.0, "protein": 7.0, "fat": 11.0, "fiber": 2.0, "gi": None, "note": "Varies by flour, layered fat and serving size."},
    "curd rice": {"kcal": 130, "carbs": 20.0, "protein": 4.0, "fat": 3.0, "fiber": 0.5, "gi": None, "note": "Varies by rice-to-curd ratio, tempering and additions."},
    "vegetable curry": {"kcal": 90, "carbs": 10.0, "protein": 3.0, "fat": 4.0, "fiber": 4.0, "gi": None, "note": "Depends on vegetables and oil."},
    "salad": {"kcal": 35, "carbs": 7.0, "protein": 2.0, "fat": 0.5, "fiber": 3.0, "gi": None, "note": "Depends on ingredients and dressing."},
    "banana": {"kcal": 89, "carbs": 23.0, "protein": 1.1, "fat": 0.3, "fiber": 2.6, "gi": None, "note": "Ripeness and variety affect carbohydrate and response."},
    "apple": {"kcal": 52, "carbs": 14.0, "protein": 0.3, "fat": 0.2, "fiber": 2.4, "gi": None, "note": "Variety and serving size matter."},
    "unknown": {"kcal": None, "carbs": None, "protein": None, "fat": None, "fiber": None, "gi": None, "note": "No suitable nutrition reference selected."}
}

LABELS = [
    "idli", "dosa", "plain rice", "chapati", "poha", "upma", "pongal", "biryani",
    "dal", "sambar", "paneer curry", "chicken curry", "chole", "samosa", "vada",
    "parotta", "curd rice", "vegetable curry", "salad", "banana", "apple",
    "fried rice", "noodles", "pizza", "burger", "sandwich", "other food"
]

def classify_image(image: Image.Image):
    clf = load_classifier()
    candidates = [f"a photo of {label}" for label in LABELS]
    out = clf(image, candidate_labels=candidates)
    cleaned = []
    for item in out[:5]:
        label = item["label"].replace("a photo of ", "").strip().lower()
        cleaned.append((label, float(item["score"])))
    return cleaned

def map_to_food(label: str):
    label = label.lower()
    for key in FOOD_DATA:
        if key != "unknown" and key in label:
            return key
    # A few broad aliases
    aliases = {
        "fried rice": "plain rice",
        "rice": "plain rice",
        "noodles": "unknown",
        "other food": "unknown",
        "burger": "unknown",
        "pizza": "unknown",
        "sandwich": "unknown"
    }
    for token, key in aliases.items():
        if token in label:
            return key
    return "unknown"

def fmt(x, suffix=""):
    return "Not available" if x is None else f"{x:g}{suffix}"

uploaded_file = st.file_uploader("Upload a clear photo of one food item or dish", type=["jpg","jpeg","png","webp"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        col_img, col_result = st.columns([1, 1.2], gap="large")
        with col_img:
            st.image(image, caption="Uploaded food photo", use_container_width=True)
        with st.spinner("Recognizing likely food... first use downloads the vision model."):
            predictions = classify_image(image)
        best_label, confidence = predictions[0]
        mapped = map_to_food(best_label)

        with col_result:
            st.subheader("AI image recognition")
            st.metric("Likely dish", best_label.title(), f"Model score: {confidence:.1%}")
            st.caption("Scores are model similarities, not calibrated probabilities.")
            st.write("Other possible matches:")
            for label, score in predictions[1:4]:
                st.write(f"• {label.title()} — {score:.1%}")

        st.divider()
        st.subheader("Confirm the food before estimating nutrition")
        options = [k for k in FOOD_DATA.keys() if k != "unknown"]
        default_index = options.index(mapped) if mapped in options else 0
        selected_food = st.selectbox(
            "Choose the closest matching dish (correct the AI if needed)",
            options=options,
            index=default_index,
            format_func=lambda x: x.title()
        )
        portion = st.number_input("Estimated portion eaten (grams)", min_value=1, max_value=2000, value=150, step=10)
        d = FOOD_DATA[selected_food]
        st.subheader("Approximate nutrition estimate")
        st.caption(f"Reference values are rough demonstration estimates per 100 g, scaled to your entered portion. Confirm against a reliable recipe/nutrition source.")
        a, b, c = st.columns(3)
        a.metric("Energy", fmt(None if d["kcal"] is None else round(d["kcal"]*portion/100), " kcal"))
        b.metric("Carbohydrates", fmt(None if d["carbs"] is None else round(d["carbs"]*portion/100, 1), " g"))
        c.metric("Protein", fmt(None if d["protein"] is None else round(d["protein"]*portion/100, 1), " g"))
        d1, d2 = st.columns(2)
        d1.metric("Fat", fmt(None if d["fat"] is None else round(d["fat"]*portion/100, 1), " g"))
        d2.metric("Fiber", fmt(None if d["fiber"] is None else round(d["fiber"]*portion/100, 1), " g"))
        st.info(d["note"])
        st.subheader("Glycemic index / post-meal response")
        st.write("**GI prediction: Not provided.** A photo classifier plus generic nutrition estimates is not enough to produce a scientifically valid GI value.")
        st.write("**Personal post-meal glucose curve: Not available.** That requires actual time-stamped glucose measurements and other data.")
        st.download_button(
            "Download this result as JSON",
            data=json.dumps({
                "recognized_label": best_label,
                "recognition_score_not_probability": confidence,
                "selected_food": selected_food,
                "portion_g": portion,
                "approximate_nutrition_scaled_from_demo_reference_per_100g": {
                    "energy_kcal": None if d["kcal"] is None else round(d["kcal"]*portion/100,1),
                    "carbs_g": None if d["carbs"] is None else round(d["carbs"]*portion/100,1),
                    "protein_g": None if d["protein"] is None else round(d["protein"]*portion/100,1),
                    "fat_g": None if d["fat"] is None else round(d["fat"]*portion/100,1),
                    "fiber_g": None if d["fiber"] is None else round(d["fiber"]*portion/100,1)
                },
                "gi": None,
                "limitations": d["note"]
            }, indent=2),
            file_name="food_analysis.json",
            mime="application/json"
        )
    except Exception as e:
        st.error("The image model could not run. Check internet access and requirements, then restart the app.")
        st.exception(e)
else:
    st.markdown("### How it works")
    st.markdown("1. Upload a food image.  2. The vision model suggests likely dish names.  3. Confirm the dish and portion.  4. Review a rough nutrition estimate.")
    st.markdown("### Current supported examples")
    st.write("Idli, dosa, plain rice, chapati, poha, upma, pongal, biryani, dal, sambar, paneer curry, chicken curry, chole, samosa, vada, parotta, curd rice, vegetable curry, salad, banana and apple.")
    st.caption("For academic use: replace the illustrative nutrient table with a cited dataset and validate it before reporting scientific conclusions.")
