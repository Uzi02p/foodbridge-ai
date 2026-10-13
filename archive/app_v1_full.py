"""
app.py
------
The FoodBridge AI web app, built with Streamlit.  Start it with:

    streamlit run app.py

It has three tabs, one per AI part:
  1. Freshness Check  - upload a food photo -> Fresh / Stale + confidence
  2. Surplus Forecast - predict tomorrow's leftover food in kg
  3. Redistribute     - the agent matches fresh surplus to the nearest NGO

Results from tabs 1 and 2 are remembered (st.session_state) and used in tab 3.
"""

import datetime as dt

import pandas as pd
import streamlit as st
from PIL import Image

from src import freshness_model, redistribution_agent, sample_data, surplus_model

st.set_page_config(page_title="FoodBridge AI", page_icon="🍲", layout="wide")


# ---------------------------------------------------------------------------
# Load models once and keep them in memory (st.cache_resource), so the app
# doesn't reload them every time a button is clicked.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_freshness_model():
    return freshness_model.load_model()


@st.cache_resource
def get_surplus_model():
    return surplus_model.load_model()


# Stop with a helpful message if the models haven't been trained yet
if not (freshness_model.MODEL_PATH.exists() and surplus_model.MODEL_PATH.exists()):
    st.error("Models not found. Please run `python train_models.py` first, then restart the app.")
    st.stop()


# ---------------------------------------------------------------------------
# Header and sidebar (restaurant details)
# ---------------------------------------------------------------------------
st.title("🍲 FoodBridge AI")
st.caption("Reducing food waste for UN SDG 2: Zero Hunger. "
           "Check freshness → forecast surplus → redistribute to people in need.")
st.warning("⚠️ This demo uses **SAMPLE (synthetic) data**: made-up restaurant records, "
           "computer-drawn food images and fictional NGOs.")

with st.sidebar:
    st.header("🏨 Your restaurant")
    restaurant_name = st.text_input("Name", "Sample Grand Hotel")
    restaurant_lat = st.number_input("Latitude", value=12.9700, format="%.4f")
    restaurant_lon = st.number_input("Longitude", value=77.6000, format="%.4f")
    st.caption("Default location is a sample point near the sample NGOs.")


tab1, tab2, tab3 = st.tabs(["1️⃣ Freshness Check", "2️⃣ Surplus Forecast", "3️⃣ Redistribute"])


# ---------------------------------------------------------------------------
# Tab 1: Freshness classifier
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Is this food still fresh?")
    st.write("Upload a photo of the food. The CNN model predicts **Fresh** or **Stale**.")

    uploaded = st.file_uploader("Food photo", type=["jpg", "jpeg", "png"])

    # Let people try it without their own photo, using one of the sample images
    sample_choice = st.selectbox("...or try a sample image", ["(none)", "Sample fresh", "Sample stale"])

    image = None
    if uploaded is not None:
        image = Image.open(uploaded)
    elif sample_choice != "(none)":
        folder = "fresh" if sample_choice == "Sample fresh" else "stale"
        first_file = sorted((sample_data.IMAGES_DIR / folder).glob("*.png"))[0]
        image = Image.open(first_file)

    if image is not None:
        col_img, col_result = st.columns([1, 2])
        col_img.image(image, caption="Your food", width=220)

        label, confidence = freshness_model.predict(get_freshness_model(), image)
        st.session_state["freshness"] = (label, confidence)  # remember for tab 3

        with col_result:
            if label == "Fresh":
                st.success(f"### ✅ Fresh\nConfidence: **{confidence:.0%}**")
            else:
                st.error(f"### ❌ Stale\nConfidence: **{confidence:.0%}**")
            st.progress(confidence)
            st.caption("Note: trained on simple synthetic images, so real photos may be misjudged. "
                       "Retrain with real photos for real use.")


# ---------------------------------------------------------------------------
# Tab 2: Surplus predictor
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("How much food will be left over tomorrow?")

    history = pd.read_csv(sample_data.SURPLUS_CSV)
    tomorrow = dt.date.today() + dt.timedelta(days=1)

    col_a, col_b, col_c = st.columns(3)
    day_name = col_a.selectbox("Day of week", surplus_model.DAY_NAMES, index=tomorrow.weekday())
    customers = col_b.number_input("Expected customers", min_value=0, value=150, step=10)
    prepared = col_c.number_input("Food to prepare (kg)", min_value=0.0, value=80.0, step=5.0)

    predicted_kg = surplus_model.predict(get_surplus_model(),
                                         surplus_model.DAY_NAMES.index(day_name), customers, prepared)
    st.session_state["surplus_kg"] = predicted_kg  # remember for tab 3

    st.metric("Predicted surplus", f"{predicted_kg:.1f} kg")
    if predicted_kg > 0.1 * prepared:
        st.info(f"Tip: more than 10% of the food may be wasted. Consider preparing about "
                f"{prepared - predicted_kg * 0.7:.0f} kg instead, or plan a donation in tab 3.")

    with st.expander("📊 See the SAMPLE history the model learned from"):
        st.line_chart(history.set_index("date")[["surplus_kg"]])
        st.dataframe(history.tail(14), width="stretch")


# ---------------------------------------------------------------------------
# Tab 3: Redistribution agent
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Send surplus food to people in need")

    # Use results from earlier tabs if available
    label, confidence = st.session_state.get("freshness", ("Unknown", 0.0))
    if label == "Unknown":
        st.info("Tip: run a Freshness Check in tab 1 first. The agent only donates food marked Fresh.")
    else:
        st.write(f"Freshness from tab 1: **{label}** ({confidence:.0%})")

    col_a, col_b = st.columns(2)
    surplus_kg = col_a.number_input("Surplus to donate (kg)", min_value=0.0, step=1.0,
                                    value=round(st.session_state.get("surplus_kg", 10.0), 1))
    food_description = col_b.text_input("What food is it?", "vegetable biryani and rotis")
    is_cooked = col_a.checkbox("This is cooked food", value=True)
    pickup_time = col_b.time_input("Pickup time", dt.time(21, 0)).strftime("%I:%M %p")

    if st.button("🤖 Run redistribution agent", type="primary"):
        ngos = redistribution_agent.load_ngos()
        result = redistribution_agent.run_agent(
            freshness_label=label, freshness_confidence=confidence,
            surplus_kg=surplus_kg, food_description=food_description, is_cooked=is_cooked,
            restaurant_name=restaurant_name,
            restaurant_lat=restaurant_lat, restaurant_lon=restaurant_lon,
            pickup_time=pickup_time, ngos=ngos,
        )

        st.markdown("#### Agent reasoning")
        for step in result["log"]:
            st.write(step)

        if result["success"]:
            st.markdown("#### Draft pickup message (copy and send)")
            st.text_area("Message", result["message"], height=230)

            st.markdown("#### Suitable NGOs (nearest first) - SAMPLE data")
            st.dataframe(result["candidates"][["name", "type", "distance_km", "max_pickup_kg", "phone"]],
                         width="stretch", hide_index=True)

            # Map: restaurant + suitable NGOs
            points = result["candidates"][["latitude", "longitude"]].copy()
            points.loc[len(points)] = [restaurant_lat, restaurant_lon]
            st.map(points)
