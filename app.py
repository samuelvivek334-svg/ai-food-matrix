
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from datetime import date, timedelta
import zxingcpp


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


# ============================================================
# AI FOOD MATRIX - TRACKER, PORTIONS, BARCODE, REPORTS & MEALS
# ============================================================

NUTRIENT_FIELDS = {
    "Calories (kcal)": "energy_kcal",
    "Protein (g)": "protein_g",
    "Carbohydrates (g)": "carbohydrate_g",
    "Fat (g)": "fat_g",
    "Fiber (g)": "fiber_g",
    "Sugar (g)": "sugar_g",
    "Saturated fat (g)": "saturated_fat_g",
    "Sodium (mg)": "sodium_mg",
}

GOAL_OPTIONS = [
    "Gym / muscle building",
    "Weight management",
    "Diabetes / blood sugar awareness",
    "General healthy eating",
]


def safe_number(value):
    try:
        number = float(value)
        if pd.isna(number):
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0


def get_off_product(barcode):
    """Look up a packaged product using the Open Food Facts API."""
    url = (
        "https://world.openfoodfacts.org/api/v2/product/"
        f"{barcode}.json"
    )
    headers = {"User-Agent": "AIFoodMatrix/1.0 (nutrition research app)"}

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != 1:
            return None, "Product not found in Open Food Facts."

        product = payload.get("product", {})
        nutrients = product.get("nutriments", {})

        def nutrient(*keys):
            for key in keys:
                value = nutrients.get(key)
                if value is not None:
                    return safe_number(value)
            return None

        record = {
            "food_name": (
                product.get("product_name")
                or product.get("product_name_en")
                or f"Product {barcode}"
            ),
            "energy_kcal": nutrient("energy-kcal_100g", "energy-kcal_value"),
            "protein_g": nutrient("proteins_100g"),
            "carbohydrate_g": nutrient("carbohydrates_100g"),
            "fat_g": nutrient("fat_100g"),
            "fiber_g": nutrient("fiber_100g"),
            "sugar_g": nutrient("sugars_100g"),
            "saturated_fat_g": nutrient("saturated-fat_100g"),
            "sodium_mg": (
                nutrient("sodium_100g") * 1000
                if nutrient("sodium_100g") is not None
                else None
            ),
            "source": "Open Food Facts",
            "source_url": (
                "https://world.openfoodfacts.org/product/"
                f"{barcode}"
            ),
        }
        return record, None

    except requests.RequestException:
        return None, "Online lookup failed. Check your internet connection."
    except (ValueError, TypeError):
        return None, "The product data could not be read."



def barcode_from_image(image):
    """Decode a barcode from an uploaded image."""
    try:
        results = zxingcpp.read_barcodes(np.array(image.convert("RGB")))
        for result in results:
            if result.text:
                return result.text
        return None
    except Exception:
        return None



def display_food_nutrients(record, grams):
    """Calculate nutrient amounts from a per-100-g database record."""
    multiplier = safe_number(grams) / 100.0

    st.caption(
        f"Calculated for {grams:g} g, using database values per 100 g."
    )

    cols = st.columns(4)
    keys = [
        ("Calories", "energy_kcal", "kcal"),
        ("Protein", "protein_g", "g"),
        ("Carbohydrates", "carbohydrate_g", "g"),
        ("Fat", "fat_g", "g"),
        ("Fiber", "fiber_g", "g"),
        ("Sugar", "sugar_g", "g"),
        ("Saturated fat", "saturated_fat_g", "g"),
        ("Sodium", "sodium_mg", "mg"),
    ]

    for index, (label, key, unit) in enumerate(keys):
        value = record.get(key)
        if value is None or pd.isna(value):
            display = "Not available"
        else:
            display = f"{safe_number(value) * multiplier:.1f} {unit}"
        cols[index % 4].metric(label, display)


# Keep food logs in this Streamlit session.
if "food_log" not in st.session_state:
    st.session_state["food_log"] = []

if "nutrition_records" not in st.session_state:
    st.session_state["nutrition_records"] = {}


st.markdown("---")
st.header("🎯 Your nutrition goal")

selected_goal = st.selectbox(
    "Choose your goal",
    GOAL_OPTIONS,
    key="nutrition_goal",
)

if selected_goal == "Gym / muscle building":
    st.info(
        "Focus on adequate energy, protein, balanced carbohydrates, "
        "healthy fats, and resistance training. Individual protein needs vary."
    )
elif selected_goal == "Weight management":
    st.info(
        "Track portions and overall intake. Sustainable changes are generally "
        "more useful than extreme calorie restriction."
    )
elif selected_goal == "Diabetes / blood sugar awareness":
    st.info(
        "Pay attention to total carbohydrates, serving size, fiber, and "
        "added sugars when available. Follow your healthcare professional's "
        "individual advice."
    )
else:
    st.info(
        "Aim for a varied diet with vegetables, fruits, protein sources, "
        "whole grains, and suitable portions."
    )


# ---------------- ONLINE BARCODE LOOKUP ----------------

st.markdown("---")
st.header("📦 Packaged-food barcode lookup")

barcode_image = st.file_uploader(
    "Upload a clear photo of the barcode",
    type=["png", "jpg", "jpeg", "webp"],
    key="barcode_image",
)

if barcode_image is not None:
    try:
        barcode_picture = Image.open(barcode_image).convert("RGB")
        st.image(
            barcode_picture,
            caption="Uploaded barcode",
            width=300,
        )
        if st.button("Read barcode image"):
            detected_barcode = barcode_from_image(barcode_picture)
            if detected_barcode:
                st.session_state["detected_barcode"] = detected_barcode
                st.success(f"Barcode detected: {detected_barcode}")
            else:
                st.warning(
                    "No barcode detected. Try a sharper, well-lit image "
                    "with the entire barcode visible."
                )
    except Exception:
        st.error("The uploaded image could not be opened.")

barcode_value = st.text_input(
    "Or enter the barcode number",
    value=st.session_state.get("detected_barcode", ""),
    key="barcode_number",
)

if st.button("Search packaged food online"):
    if not barcode_value.strip().isdigit():
        st.warning("Enter a valid numeric barcode.")
    else:
        with st.spinner("Looking up the packaged food..."):
            product_record, product_error = get_off_product(
                barcode_value.strip()
            )

        if product_record:
            st.session_state["barcode_result"] = product_record
            st.success("Product found.")
        else:
            st.session_state.pop("barcode_result", None)
            st.warning(product_error)

barcode_record = st.session_state.get("barcode_result")

if barcode_record:
    st.subheader(str(barcode_record["food_name"]))
    st.write("Source: Open Food Facts")
    st.write("Nutrient figures are reported per 100 g when available.")

    product_grams = st.number_input(
        "Amount eaten (g)",
        min_value=1.0,
        max_value=5000.0,
        value=100.0,
        step=10.0,
        key="barcode_portion",
    )
    display_food_nutrients(barcode_record, product_grams)

    if st.button("Add packaged food to today's log"):
        st.session_state["food_log"].append({
            "date": date.today().isoformat(),
            "meal": "Packaged food",
            "food_name": barcode_record["food_name"],
            "grams": product_grams,
            **{
                key: (
                    safe_number(barcode_record[key]) * product_grams / 100
                    if barcode_record.get(key) is not None
                    else None
                )
                for key in NUTRIENT_FIELDS.values()
            },
            "source": "Open Food Facts",
        })
        st.success("Added to this session's food log.")


# ---------------- PORTION CALCULATOR ----------------

st.markdown("---")
st.header("⚖️ Portion-size calculator")


def load_local_food_data():
    if not DATA_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(DATA_FILE)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()

if not local_records.empty and "food_name" in local_records.columns:
    available_foods.extend(
        local_records["food_name"].dropna().astype(str).tolist()
    )

available_foods.extend(
    list(st.session_state["nutrition_records"].keys())
)

if not available_foods:
    st.caption(
        "No local nutrition records are available yet. Search for a food "
        "online below, or use the barcode lookup above."
    )
else:
    portion_food = st.selectbox(
        "Select a saved food",
        sorted(set(available_foods)),
        key="portion_food",
    )
    portion_grams = st.number_input(
        "Portion size in grams",
        min_value=1.0,
        max_value=5000.0,
        value=150.0,
        step=10.0,
        key="portion_grams",
    )

    local_record = local_nutrition(portion_food)
    portion_record = (
        st.session_state["nutrition_records"].get(portion_food)
        or local_record
    )

    if portion_record:
        display_food_nutrients(portion_record, portion_grams)

        if st.button("Add portion to daily log"):
            st.session_state["food_log"].append({
                "date": date.today().isoformat(),
                "meal": "Meal",
                "food_name": portion_food,
                "grams": portion_grams,
                **{
                    key: (
                        safe_number(portion_record[key])
                        * portion_grams / 100
                        if portion_record.get(key) is not None
                        else None
                    )
                    for key in NUTRIENT_FIELDS.values()
                },
                "source": portion_record.get("source", "Local database"),
            })
            st.success("Portion added to your log.")


# ---------------- ONLINE FOOD SEARCH FOR ALL NUTRIENTS ----------------

st.markdown("---")
st.header("🔎 Find nutrition online")

online_food_query = st.text_input(
    "Search a food name",
    placeholder="e.g. cooked lentils, apple, chicken breast",
    key="extended_food_search",
)

if st.button("Look up food nutrition"):
    if not online_food_query.strip():
        st.warning("Enter a food name.")
    else:
        with st.spinner("Searching the online nutrition database..."):
            online_record, online_error = online_nutrition(
                online_food_query.strip()
            )

        if online_record:
            # USDA records may have fewer nutrients than packaged-food data.
            online_record.setdefault("sugar_g", None)
            online_record.setdefault("saturated_fat_g", None)
            online_record.setdefault("sodium_mg", None)

            # Store only in session memory until the user saves/logs it.
            st.session_state["nutrition_records"][
                online_food_query.strip()
            ] = online_record
            st.session_state["latest_online_record"] = online_record
            st.success("Nutrition record retrieved.")
        else:
            st.warning(
                online_error or "No nutrition record was returned."
            )

latest_record = st.session_state.get("latest_online_record")

if latest_record:
    st.write("**Matched database record:**", latest_record.get("food_name"))
    st.write("**Source:**", latest_record.get("source", "USDA FoodData Central"))

    online_grams = st.number_input(
        "Portion to calculate (g)",
        min_value=1.0,
        max_value=5000.0,
        value=100.0,
        step=10.0,
        key="online_portion_grams",
    )
    display_food_nutrients(latest_record, online_grams)

    if st.button("Add online food to daily log"):
        st.session_state["food_log"].append({
            "date": date.today().isoformat(),
            "meal": "Meal",
            "food_name": latest_record.get("food_name", online_food_query),
            "grams": online_grams,
            **{
                key: (
                    safe_number(latest_record[key]) * online_grams / 100
                    if latest_record.get(key) is not None
                    else None
                )
                for key in NUTRIENT_FIELDS.values()
            },
            "source": latest_record.get("source", "Online database"),
        })
        st.success("Food added to your daily log.")


# ---------------- HIGH-PROTEIN FOOD FINDER ----------------

st.markdown("---")
st.header("💪 High-protein food finder")

protein_threshold = st.slider(
    "Minimum protein per 100 g",
    min_value=1,
    max_value=30,
    value=10,
)

protein_foods = []

if not local_records.empty:
    for _, row in local_records.iterrows():
        protein = row.get("protein_g")
        if pd.notna(protein) and safe_number(protein) >= protein_threshold:
            protein_foods.append({
                "Food": row.get("food_name", "Unknown"),
                "Protein (g/100 g)": safe_number(protein),
                "Calories (kcal/100 g)": row.get("energy_kcal"),
                "Source": row.get("source", "Local database"),
            })

for food_name, record in st.session_state["nutrition_records"].items():
    protein = record.get("protein_g")
    if protein is not None and safe_number(protein) >= protein_threshold:
        protein_foods.append({
            "Food": record.get("food_name", food_name),
            "Protein (g/100 g)": safe_number(protein),
            "Calories (kcal/100 g)": record.get("energy_kcal"),
            "Source": record.get("source", "Online database"),
        })

if protein_foods:
    protein_table = pd.DataFrame(protein_foods).drop_duplicates(subset=["Food"])
    protein_table = protein_table.sort_values(
        "Protein (g/100 g)", ascending=False
    )
    st.dataframe(protein_table, use_container_width=True)
else:
    st.info(
        "Search foods online or add verified local nutrition records to build "
        "your high-protein list. The finder does not invent missing values."
    )


# ---------------- DAILY FOOD LOG ----------------

st.markdown("---")
st.header("📅 Daily nutrition tracker")

meal_type = st.selectbox(
    "Meal category for the next manual log",
    ["Breakfast", "Lunch", "Dinner", "Snack"],
    key="manual_meal_type",
)

with st.expander("Add a food manually using a known nutrition record"):
    manual_name = st.text_input("Food name", key="manual_food_name")
    manual_grams = st.number_input(
        "Amount eaten (g)",
        min_value=1.0,
        max_value=5000.0,
        value=100.0,
        step=10.0,
        key="manual_food_grams",
    )
    manual_calories = st.number_input(
        "Calories for your portion (kcal)",
        min_value=0.0,
        max_value=20000.0,
        value=0.0,
        key="manual_calories",
    )
    manual_protein = st.number_input(
        "Protein for your portion (g)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        key="manual_protein",
    )
    manual_carbs = st.number_input(
        "Carbohydrates for your portion (g)",
        min_value=0.0,
        max_value=2000.0,
        value=0.0,
        key="manual_carbs",
    )
    manual_fat = st.number_input(
        "Fat for your portion (g)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        key="manual_fat",
    )
    manual_fiber = st.number_input(
        "Fiber for your portion (g)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        key="manual_fiber",
    )
    manual_sugar = st.number_input(
        "Sugar for your portion (g)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        key="manual_sugar",
    )

    if st.button("Add manual entry"):
        if not manual_name.strip():
            st.warning("Enter a food name.")
        else:
            st.session_state["food_log"].append({
                "date": date.today().isoformat(),
                "meal": meal_type,
                "food_name": manual_name.strip(),
                "grams": manual_grams,
                "energy_kcal": manual_calories,
                "protein_g": manual_protein,
                "carbohydrate_g": manual_carbs,
                "fat_g": manual_fat,
                "fiber_g": manual_fiber,
                "sugar_g": manual_sugar,
                "saturated_fat_g": None,
                "sodium_mg": None,
                "source": "User-entered values",
            })
            st.success("Entry added.")

if st.session_state["food_log"]:
    logs = pd.DataFrame(st.session_state["food_log"])
    logs["date"] = logs["date"].astype(str)

    today_logs = logs[logs["date"] == date.today().isoformat()]
    st.subheader("Today's foods")

    if not today_logs.empty:
        st.dataframe(today_logs, use_container_width=True)

        today_totals = today_logs[
            ["energy_kcal", "protein_g", "carbohydrate_g",
             "fat_g", "fiber_g", "sugar_g"]
        ].sum(numeric_only=True)

        metric_columns = st.columns(3)
        for col, (label, key) in zip(
            metric_columns,
            [
                ("Calories (kcal)", "energy_kcal"),
                ("Protein (g)", "protein_g"),
                ("Carbohydrates (g)", "carbohydrate_g"),
            ],
        ):
            col.metric(label, f"{today_totals.get(key, 0):.1f}")

        metric_columns_2 = st.columns(3)
        for col, (label, key) in zip(
            metric_columns_2,
            [
                ("Fat (g)", "fat_g"),
                ("Fiber (g)", "fiber_g"),
                ("Sugar (g)", "sugar_g"),
            ],
        ):
            col.metric(label, f"{today_totals.get(key, 0):.1f}")

    st.download_button(
        "Download all food logs as CSV",
        data=logs.to_csv(index=False).encode("utf-8"),
        file_name="ai_food_matrix_food_log.csv",
        mime="text/csv",
    )

    if st.button("Clear all session food logs"):
        st.session_state["food_log"] = []
        st.rerun()
else:
    st.info("No meals logged in this session yet.")


# ---------------- WEEKLY REPORT ----------------

st.markdown("---")
st.header("📈 Weekly progress report")

if st.session_state["food_log"]:
    logs = pd.DataFrame(st.session_state["food_log"])
    logs["date"] = pd.to_datetime(logs["date"], errors="coerce")

    week_start = date.today() - timedelta(days=6)
    weekly = logs[
        (logs["date"].dt.date >= week_start)
        & (logs["date"].dt.date <= date.today())
    ].copy()

    if weekly.empty:
        st.info("No meals have been logged in the last seven days.")
    else:
        numeric_columns = [
            "energy_kcal", "protein_g", "carbohydrate_g",
            "fat_g", "fiber_g", "sugar_g",
        ]
        for column in numeric_columns:
            if column not in weekly.columns:
                weekly[column] = 0.0

        weekly[numeric_columns] = weekly[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)

        daily = weekly.groupby(weekly["date"].dt.date)[numeric_columns].sum()
        st.write("Daily totals for the last seven days")
        st.dataframe(daily, use_container_width=True)

        st.line_chart(daily[["energy_kcal", "protein_g"]])

        st.caption(
            "Reports include only logged foods. Missing nutrient values are "
            "not proof of zero intake. Download your logs to keep a copy."
        )
else:
    st.info("Log some meals first to generate a weekly report.")


# ---------------- MEAL SUGGESTIONS ----------------

st.markdown("---")
st.header("🍽️ Meal ideas for your goal")

meal_ideas = {
    "Gym / muscle building": [
        "Eggs with whole-grain toast and fruit",
        "Dal, rice, vegetables, and curd",
        "Paneer or tofu with chapati and vegetables",
        "Chicken or fish with rice and vegetables",
        "Yogurt with oats and nuts",
    ],
    "Weight management": [
        "Vegetable omelette with fruit",
        "Dal with vegetables and a suitable rice portion",
        "Grilled tofu, paneer, chicken, or fish with vegetables",
        "Bean salad with whole grains",
        "Plain yogurt with fruit",
    ],
    "Diabetes / blood sugar awareness": [
        "Eggs or tofu with non-starchy vegetables",
        "Dal with vegetables and a suitable portion of whole grains",
        "Unsweetened yogurt with nuts, if appropriate",
        "Fish, chicken, or tofu with vegetables",
        "Beans with salad and a measured grain portion",
    ],
    "General healthy eating": [
        "Oats with fruit and unsweetened yogurt",
        "Idli with sambar",
        "Rice, dal, and a variety of vegetables",
        "Chapati with beans or paneer and vegetables",
        "Fruit with nuts or plain yogurt",
    ],
}

st.write(f"Suggestions for: **{selected_goal}**")
for idea in meal_ideas[selected_goal]:
    st.markdown(f"- {idea}")

st.caption(
    "These are general meal ideas, not personalized medical prescriptions. "
    "Adjust for allergies, dietary preferences, medications, and individual "
    "health needs. For diabetes, follow your care team's guidance."
)
