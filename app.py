
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

st.set_page_config(page_title="AI Food Matrix", page_icon="🥗", layout="wide")

MODEL_ID = "openai/clip-vit-base-patch32"
DATA_FILE = Path("data/food_reference.csv")
USER_AGENT = "AIFoodMatrix/1.0 (educational research prototype)"

FOOD_LABELS = [
    "pizza", "hamburger", "sandwich", "salad", "soup",
    "fried rice", "noodles", "pasta", "grilled chicken",
    "chicken curry", "chicken biryani", "vegetable biryani",
    "plain cooked rice", "dal lentil curry", "chickpea curry",
    "paneer curry", "idli", "dosa", "masala dosa", "uttapam",
    "sambar", "pongal", "curd rice", "chapati roti", "parotta",
    "poori", "vada", "samosa", "pav bhaji", "vegetable curry",
    "fish curry", "grilled fish", "omelette", "boiled eggs",
    "fruit salad", "cake", "ice cream", "bread", "oatmeal",
    "stir fried vegetables", "fried chicken", "vegetable noodles",
    "hakka noodles", "egg fried rice", "chicken 65",
    "vegetable manchurian", "mushroom manchurian",
]

NUTRIENT_IDS = {
    "Energy (kcal)": 1008,
    "Protein (g)": 1003,
    "Carbohydrate (g)": 1005,
    "Fat (g)": 1004,
    "Fibre (g)": 1079,
}


@st.cache_resource(show_spinner="Loading image model...")
def load_model(offline: bool):
    processor = CLIPProcessor.from_pretrained(
        MODEL_ID, local_files_only=offline
    )
    model = CLIPModel.from_pretrained(
        MODEL_ID, local_files_only=offline
    )
    model.eval()
    return processor, model


def identify_food(image, offline):
    processor, model = load_model(offline)
    prompts = [f"a photo of {food}" for food in FOOD_LABELS]
    inputs = processor(
        text=prompts, images=image, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits_per_image.softmax(dim=1)[0]
    top = torch.argsort(scores, descending=True)[:5].tolist()
    return [(FOOD_LABELS[i], float(scores[i])) for i in top]


def search_usda(food_name, api_key):
    """Search USDA FoodData Central. Requires a private API key."""
    if not api_key:
        return None

    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    try:
        response = requests.post(
            url,
            params={"api_key": api_key},
            json={"query": food_name, "pageSize": 5},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        response.raise_for_status()
        foods = response.json().get("foods", [])
    except (requests.RequestException, ValueError):
        return None

    if not foods:
        return None

    # Show the best search match for user review; it is not guaranteed
    # to represent the exact recipe or cooking method.
    food = foods[0]
    values = {}
    for item in food.get("foodNutrients", []):
        nutrient_id = item.get("nutrientId")
        for label, wanted_id in NUTRIENT_IDS.items():
            if nutrient_id == wanted_id:
                values[label] = item.get("value")

    return {
        "food_name": food.get("description", food_name),
        "source": "USDA FoodData Central",
        "source_url": (
            "https://fdc.nal.usda.gov/food-details/"
            + str(food.get("fdcId", ""))
            + "/nutrients"
        ),
        "data_type": food.get("dataType", "Not specified"),
        "values": values,
        "basis": "USDA search result; confirm the food and nutrient basis.",
    }


def lookup_off_barcode(barcode):
    """Look up packaged food by barcode in Open Food Facts."""
    if not barcode.strip():
        return None

    url = (
        "https://world.openfoodfacts.org/api/v3/product/"
        + requests.utils.quote(barcode.strip(), safe="")
        + ".json"
    )
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    product = data.get("product")
    if data.get("status") != 1 or not product:
        return None

    nutriments = product.get("nutriments", {})
    mapping = {
        "Energy (kcal)": "energy-kcal_100g",
        "Protein (g)": "proteins_100g",
        "Carbohydrate (g)": "carbohydrates_100g",
        "Fat (g)": "fat_100g",
        "Fibre (g)": "fiber_100g",
    }
    values = {
        label: nutriments[key]
        for label, key in mapping.items()
        if nutriments.get(key) is not None
    }

    return {
        "food_name": product.get("product_name", "Packaged food"),
        "source": "Open Food Facts",
        "source_url": (
            "https://world.openfoodfacts.org/product/"
            + barcode.strip()
        ),
        "data_type": "Community-submitted packaged-food label",
        "values": values,
        "basis": "Reported per 100 g on the product record.",
    }


def lookup_local(food_name):
    """Use a locally saved CSV table when internet data is unavailable."""
    if not DATA_FILE.exists():
        return None

    try:
        df = pd.read_csv(DATA_FILE)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None

    if "food_name" not in df.columns or df.empty:
        return None

    match = df[
        df["food_name"].astype(str).str.casefold()
        == food_name.casefold()
    ]
    if match.empty:
        # Fallback: simple partial match, still show as a candidate.
        match = df[
            df["food_name"].astype(str).str.contains(
                food_name, case=False, regex=False, na=False
            )
        ]
    if match.empty:
        return None

    row = match.iloc[0]
    columns = {
        "Energy (kcal)": "energy_kcal",
        "Protein (g)": "protein_g",
        "Carbohydrate (g)": "carbohydrate_g",
        "Fat (g)": "fat_g",
        "Fibre (g)": "fiber_g",
    }
    values = {
        label: row[column]
        for label, column in columns.items()
        if column in df.columns and pd.notna(row.get(column))
    }

    return {
        "food_name": str(row["food_name"]),
        "source": str(row.get("source", "Local CSV; source unspecified")),
        "source_url": str(row.get("source_url", "")),
        "data_type": "Saved local reference",
        "values": values,
        "basis": "Check the CSV for the stated units and per-100-g basis.",
    }


def display_result(result, portion_g):
    st.subheader("Nutrition reference")
    st.write("**Matched record:**", result["food_name"])
    st.write("**Source:**", result["source"])
    st.write("**Record type:**", result["data_type"])
    st.write("**Basis:**", result["basis"])

    if result.get("source_url"):
        st.markdown(f"[Open source record]({result['source_url']})")

    if not result["values"]:
        st.warning("No usable nutrient values were returned for this record.")
        return

    st.caption(
        "Values below are scaled only if the source values are per 100 g. "
        "Verify the source basis before using the numbers."
    )

    per_100g = result["values"]
    scaled = {}
    for label, value in per_100g.items():
        try:
            scaled[label] = float(value) * portion_g / 100
        except (TypeError, ValueError):
            continue

    cols = st.columns(min(len(scaled), 5) or 1)
    for i, (label, value) in enumerate(scaled.items()):
        cols[i % len(cols)].metric(label, f"{value:.1f}")

    st.warning(
        "These values describe the matched database record, not measurements "
        "made from your photo. If the food or serving differs, the result "
        "may not match what you ate."
    )


st.title("🥗 AI Food Matrix")
st.subheader("Live camera + image upload + online/offline food references")

st.info(
    "Educational research prototype. AI recognition is a suggestion, not proof. "
    "Exact ingredients, serving weight, recipe, GI and personal blood-glucose "
    "response cannot be determined reliably from a photo alone."
)

with st.sidebar:
    st.header("Mode")
    offline = st.toggle(
        "Offline mode",
        value=False,
        help="Requires model weights already downloaded and a local CSV reference."
    )
    st.caption(
        "For first-time setup, connect to the internet and run the model once "
        "before enabling offline mode."
    )

    st.header("USDA lookup")
    st.caption("Optional. Enter a USDA API key in Streamlit Secrets.")
    try:
        usda_key = st.secrets.get("USDA_API_KEY", "")
    except Exception:
        usda_key = os.getenv("USDA_API_KEY", "")

    st.markdown("[Get a USDA API key](https://fdc.nal.usda.gov/api-key-signup/)")

    st.header("Image source")
    source_mode = st.radio(
        "Choose photo input",
        ["Take a photo", "Upload a photo"],
    )

uploaded_image = None
if source_mode == "Take a photo":
    uploaded_image = st.camera_input("Take a food photo")
else:
    uploaded_image = st.file_uploader(
        "Choose a food image",
        type=["jpg", "jpeg", "png", "webp"],
    )

barcode = st.text_input(
    "Optional: packaged food barcode",
    help="Enter the numbers printed below the barcode. This looks up a packaged product, not a cooked meal."
)

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")
    except Exception:
        st.error("Unable to read this image. Please try another JPG, PNG or WEBP file.")
        st.stop()

    st.image(image, caption="Selected food image", use_container_width=True)

    if st.button("Identify food", type="primary", use_container_width=True):
        try:
            with st.spinner("Analysing photo..."):
                matches = identify_food(image, offline)
            st.session_state["food_matches"] = matches
        except Exception as exc:
            st.error(
                "Could not load or run the AI model. In offline mode, ensure "
                "the model was downloaded previously. Check the app logs for details."
            )
            st.caption(f"Technical detail: {type(exc).__name__}: {exc}")

matches = st.session_state.get("food_matches")
if matches:
    st.subheader("Possible food matches")
    st.caption(
        "Scores are relative across the model's candidate labels, not calibrated "
        "probabilities. Similar-looking foods may be confused."
    )

    st.dataframe(
        pd.DataFrame([
            {"Candidate": name, "Relative score": f"{score * 100:.1f}%"}
            for name, score in matches
        ]),
        hide_index=True,
        use_container_width=True,
    )

    default_name = matches[0][0]
    selected_food = st.selectbox(
        "Confirm or correct the food",
        FOOD_LABELS,
        index=FOOD_LABELS.index(default_name),
    )
    portion_g = st.number_input(
        "Portion weight in grams",
        min_value=1,
        max_value=3000,
        value=250,
        step=10,
    )

    if st.button("Find nutrition reference", use_container_width=True):
        result = None

        if barcode.strip():
            if offline:
                st.info("Barcode lookup needs internet; trying the local food table instead.")
            else:
                with st.spinner("Looking up packaged food..."):
                    result = lookup_off_barcode(barcode)

        if result is None and not offline and usda_key:
            with st.spinner("Searching USDA FoodData Central..."):
                result = search_usda(selected_food, usda_key)

        if result is None:
            result = lookup_local(selected_food)

        st.session_state["nutrition_result"] = result
        st.session_state["portion_g"] = portion_g
        st.session_state["selected_food"] = selected_food

    result = st.session_state.get("nutrition_result")
    if result:
        display_result(result, st.session_state.get("portion_g", 250))

        export = {
            "selected_food": st.session_state.get("selected_food", ""),
            "portion_g_entered": st.session_state.get("portion_g", 250),
            "reference_food": result["food_name"],
            "source": result["source"],
            "source_url": result.get("source_url", ""),
            "data_type": result["data_type"],
            "basis": result["basis"],
            "GI": "Not predicted from photo",
        }
        for key, value in result["values"].items():
            export[key + " source value"] = value

        st.download_button(
            "Download analysis summary",
            data=pd.DataFrame([export]).to_csv(index=False).encode("utf-8"),
            file_name="ai_food_matrix_result.csv",
            mime="text/csv",
        )
    elif st.session_state.get("selected_food"):
        st.warning(
            "No matching online or local nutrient record was found. "
            "Do not substitute an invented value. Try a more specific food name "
            "or add a verified record to data/food_reference.csv."
        )

st.markdown("---")
st.subheader("What this version does not claim")
st.write(
    "- It does not calculate a scientifically validated GI from the image.\n"
    "- It does not estimate exact food weight from pixels.\n"
    "- It does not treat AI similarity scores as calibrated confidence.\n"
    "- It does not replace nutrition assessment or medical advice."
)
