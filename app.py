"""
app.py
------
The FoodBridge AI web app, built with Streamlit.  Start it with:

    streamlit run app.py

A sidebar menu switches between 5 pages. Each page is its own small
function below. The AI logic lives in src/ (models) and agent.py (agent);
this file only draws the screens.
"""

import datetime as dt
import json
import os

import pandas as pd
import streamlit as st
from PIL import Image

import agent
from src import freshness_model, sample_data, surplus_model

st.set_page_config(page_title="FoodBridge AI", page_icon="🍲", layout="wide")

# Assumption used on every page: one meal is about 0.4 kg of food
MEAL_KG = 0.4


# Load the freshness model only once and keep it in memory (st.cache_resource),
# instead of reloading the file every time the user clicks something.
@st.cache_resource
def get_freshness_model():
    return freshness_model.load_model()


@st.cache_resource
def get_surplus_model():
    return surplus_model.load_model()


# ---------------------------------------------------------------------------
# Pages - one function per menu item
# ---------------------------------------------------------------------------
def home_page():
    st.title("🍲 FoodBridge AI")
    st.subheader("Reducing food waste for UN SDG 2: Zero Hunger")
    st.write(
        "Restaurants, hotels and caterers throw away edible food because they can't "
        "quickly tell what is still fresh, and have no easy way to get surplus food "
        "to people in need. FoodBridge AI helps in three steps:"
    )
    st.markdown(
        "1. **Check Freshness** - upload a food photo, get *Fresh* or *Stale* with a confidence score.\n"
        "2. **Predict Surplus** - forecast tomorrow's leftover food from past data.\n"
        "3. **Redistribute** - match fresh surplus to the nearest NGO and draft a pickup message."
    )
    st.info("Use the menu on the left to open each page.")


def check_freshness_page():
    st.title("🔍 Check Freshness")
    st.write("Upload a photo of food and the AI model will say **Fresh** or **Stale**.")

    # If the model hasn't been trained yet, explain what to do and stop here
    if not freshness_model.MODEL_PATH.exists():
        st.warning("The freshness model hasn't been trained yet. "
                   "Download the dataset and run `python train_freshness.py` (see README).")
        return

    # Step 1: let the user upload a photo
    uploaded = st.file_uploader("Food photo", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    # Step 2: ask the model for a prediction
    image = Image.open(uploaded)
    label, confidence = freshness_model.predict(get_freshness_model(), image)
    level, advice = freshness_model.recommendation(label, confidence)

    # Remember the result so the Redistribute page can use it later
    st.session_state["freshness"] = (label, confidence)

    # Step 3: show the photo next to the result
    col_img, col_result = st.columns([1, 2])
    col_img.image(image, caption="Uploaded photo", width=260)
    with col_result:
        st.metric("Prediction", label, f"{confidence:.0%} confidence", delta_color="off")
        st.progress(confidence)
        # Green = safe, yellow = check by hand, red = don't donate
        {"safe": st.success, "check": st.warning, "unsafe": st.error}[level](advice)
        st.caption("AI can't detect every food-safety risk (e.g. bacteria). "
                   "A person should always make the final call.")

    # Extra: how good is the model? (numbers saved by train_freshness.py)
    metrics_file = freshness_model.MODEL_PATH.parent / "freshness_metrics.json"
    chart_file = freshness_model.MODEL_PATH.parent / "freshness_confusion_matrix.png"
    if metrics_file.exists():
        with st.expander("📊 How accurate is this model?"):
            metrics = json.loads(metrics_file.read_text())
            st.write(f"Tested on **{metrics['num_test_images']}** unseen images: "
                     f"**{metrics['test_accuracy']:.1%}** correct.")
            if chart_file.exists():
                st.image(str(chart_file), width=360)


def predict_surplus_page():
    st.title("📈 Predict Surplus")
    st.write("Tell us about tomorrow and the model predicts how much cooked food will be left over.")
    st.warning("⚠️ This model is trained on **synthetic (made-up) data** for one sample restaurant, "
               "not real sales records.")

    # If the model hasn't been trained yet, explain what to do and stop here
    if not surplus_model.MODEL_PATH.exists():
        st.error("The surplus model hasn't been trained yet. Run `python train_surplus.py` first.")
        return

    sales = pd.read_csv(sample_data.SALES_CSV)

    # Step 1: the restaurant enters tomorrow's details (tomorrow's weekday is pre-selected)
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    col_day, col_customers = st.columns(2)
    day = col_day.selectbox("Tomorrow is", surplus_model.DAY_NAMES, index=tomorrow.weekday())

    # Pre-fill the usual number of customers for that weekday, from past records
    past_customers = sales.loc[sales["day_of_week"] == day, "customers"]
    customers = col_customers.number_input("Expected customers", min_value=30, max_value=500,
                                           value=int(past_customers.median()), step=10)
    col_customers.caption(f"Usual for a {day}: about {past_customers.median():.0f} customers")

    # A straight-line model is only trustworthy near the data it learned from
    if not past_customers.min() <= customers <= past_customers.max():
        st.warning(f"On past {day}s the restaurant had {past_customers.min()}–{past_customers.max()} "
                   "customers. Your number is outside that range, so this prediction is less reliable.")

    # Step 2: ask the model for a prediction
    predicted_kg = surplus_model.predict(get_surplus_model(), day, customers)
    st.session_state["surplus_kg"] = predicted_kg  # remember it for the Redistribute page

    # Step 3: show the result, in kg and in meals (MEAL_KG per meal)
    st.metric("Predicted surplus tomorrow", f"{predicted_kg:.1f} kg",
              f"≈ {predicted_kg / MEAL_KG:.0f} meals that could feed people", delta_color="off")

    # Step 4: a simple chart comparing tomorrow to the past
    chart = pd.DataFrame({
        "Surplus (kg)": [
            predicted_kg,
            sales.loc[sales["day_of_week"] == day, "surplus_kg"].mean(),
            sales["surplus_kg"].mean(),
        ]
    }, index=["Tomorrow (predicted)", f"Average past {day}", "Average all days"])
    st.bar_chart(chart, horizontal=True)

    if predicted_kg >= 5:
        st.info("💡 That's enough to donate. Plan a pickup on the **Redistribute** page, "
                "or cook a little less tomorrow.")

    # Extra: how accurate is the model, and what data did it learn from?
    with st.expander("📊 How accurate is this model? (synthetic data)"):
        if surplus_model.METRICS_PATH.exists():
            m = json.loads(surplus_model.METRICS_PATH.read_text())
            c1, c2 = st.columns(2)
            c1.metric("R² score", f"{m['r2_score']:.2f}", help="1.00 = perfect, 0 = guessing the average")
            c2.metric("Average error (MAE)", f"{m['mae_kg']:.1f} kg")
            st.caption(f"Tested on {m['num_test_days']} days the model never saw during training. "
                       f"Each extra customer changes surplus by {m['customer_weight']:+.2f} kg.")
        st.dataframe(sales.tail(14), width="stretch", hide_index=True)


def redistribute_page():
    st.title("🤝 Redistribute")
    st.write("The agent checks the food is safe, picks the best NGO, drafts a pickup request "
             "and logs the donation.")
    st.warning("⚠️ The NGOs on this page are **fictional sample data**.")

    if not agent.NGOS_CSV.exists():
        st.error("NGO list not found. Run `python train_surplus.py` to create the sample data.")
        return

    # --- Freshness result from the Check Freshness page ---
    freshness = st.session_state.get("freshness")  # (label, confidence) or None
    if freshness:
        st.write(f"Freshness check result: **{freshness[0]}** ({freshness[1]:.0%} confidence)")
    else:
        st.info("No freshness check yet. Check the food on the **Check Freshness** page first.")
    # Demo mode lets you try the agent before the freshness model is trained.
    # The agent logs it as a demo, so it never counts as a real check.
    demo_mode = st.checkbox("Demo mode: pretend the food passed the freshness check (testing only)")

    # --- The restaurant fills in the donation details ---
    ngos = pd.read_csv(agent.NGOS_CSV)
    col1, col2 = st.columns(2)
    restaurant = col1.text_input("Restaurant name", "Sample Grand Hotel")
    area = col2.selectbox("Your area", sorted(ngos["area"].unique()))
    food_type = col1.selectbox("Food type", agent.FOOD_TYPES)
    # Pre-fill the quantity with the prediction from Predict Surplus, if there is one
    quantity_kg = col2.number_input("Quantity (kg)", min_value=0.5, max_value=500.0, step=0.5,
                                    value=max(0.5, round(st.session_state.get("surplus_kg", 10.0), 1)))
    in_two_hours = dt.datetime.now() + dt.timedelta(hours=2)
    deadline_date = col1.date_input("Pickup deadline: date", in_two_hours.date())
    deadline_time = col2.time_input("Pickup deadline: time", in_two_hours.time().replace(second=0, microsecond=0))
    deadline = dt.datetime.combine(deadline_date, deadline_time)

    # --- Optional LLM key: typed here, or set as the ANTHROPIC_API_KEY environment variable ---
    with st.expander("✨ Optional: let Claude write the message (needs an Anthropic API key)"):
        typed_key = st.text_input("Anthropic API key", type="password",
                                  help="Used only for this request. Never saved.")
        st.caption("Without a key, the agent uses a ready-made message template.")
    api_key = typed_key or os.environ.get("ANTHROPIC_API_KEY")

    if not st.button("🤖 Run agent", type="primary"):
        show_donation_log()
        return

    # --- Run the agent and show each step ---
    with st.spinner("Agent working..."):
        result = agent.run_agent(restaurant, food_type, quantity_kg, area, deadline,
                                 freshness, api_key, demo_mode)

    st.subheader("Agent steps")
    for name, passed, note in result["steps"]:
        if passed and demo_mode and name.startswith("1."):
            st.warning(f"**{name}**: {note}")
        else:
            (st.success if passed else st.error)(f"**{name}**: {note}")

    if result["success"]:
        ngo = result["ngo"]
        st.subheader(f"📨 Send to {ngo['name']} ({ngo['phone']})")
        label = "written by Claude" if result["message_source"] == "llm" else "from template"
        st.text_area(f"Pickup request ({label}). Copy and send it:", result["message"], height=150)

    if result["ranked"] is not None:
        with st.expander("Why this NGO? (all NGOs, best match first)"):
            st.dataframe(result["ranked"][["name", "area", "capacity_kg", "capacity_left_kg",
                                           "accepts_cooked_food", "reason"]],
                         width="stretch", hide_index=True)
    show_donation_log()


def show_donation_log():
    """Show the most recent donations the agent has logged."""
    if agent.DONATIONS_CSV.exists():
        with st.expander("📒 Donation log (data/donations.csv)"):
            log = pd.read_csv(agent.DONATIONS_CSV)
            st.dataframe(log.drop(columns=["message"]).iloc[::-1], width="stretch", hide_index=True)


def impact_dashboard_page():
    st.title("🌍 Impact Dashboard")
    st.write("How much food FoodBridge AI has rescued, from the donations the agent logged.")

    # Step 1: load the donation log written by the agent on the Redistribute page
    if not agent.DONATIONS_CSV.exists():
        st.info("No donations logged yet. Run the agent on the **Redistribute** page first.")
        return
    donations = pd.read_csv(agent.DONATIONS_CSV, parse_dates=["logged_at"])

    # Step 2: leave out demo donations (freshness check skipped), unless the user asks for them
    is_demo = donations.get("freshness_check", pd.Series("", index=donations.index)).str.startswith("demo")
    if is_demo.any():
        include_demo = st.toggle(f"Include {is_demo.sum()} demo donation(s)", value=False,
                                 help="Demo donations skipped the freshness check, so they aren't real.")
        if not include_demo:
            donations = donations[~is_demo]
    if donations.empty:
        st.info("Only demo donations so far. Switch on the toggle above to see them.")
        return

    # Step 3: the four headline numbers
    total_kg = donations["quantity_kg"].sum()
    this_week = pd.Timestamp.now().to_period("W")  # the current Monday-Sunday week
    kg_this_week = donations.loc[donations["logged_at"].dt.to_period("W") == this_week, "quantity_kg"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Food saved", f"{total_kg:,.1f} kg", border=True)
    col2.metric("Meals served (est.)", f"{total_kg / MEAL_KG:,.0f}",
                help=f"Food saved ÷ {MEAL_KG} kg per meal", border=True)
    col3.metric("Donations", f"{len(donations):,}", border=True)
    col4.metric("Active NGO partners", donations["ngo_name"].nunique(),
                help="NGOs that received at least one donation", border=True)

    # Step 4: weekly trend. Add up kg per week (weeks start on Monday) and show
    # weeks with no donations as 0 so gaps are visible, not hidden.
    weekly = (donations.set_index("logged_at")["quantity_kg"]
              .resample("W-MON", label="left", closed="left").sum())
    # Make sure every week up to the current one has a bar (empty weeks = 0 kg)
    this_monday = pd.Timestamp.now().normalize() - pd.Timedelta(days=pd.Timestamp.now().weekday())
    weekly = weekly.reindex(pd.date_range(weekly.index.min(), this_monday, freq="W-MON"), fill_value=0)
    weekly.index = weekly.index.strftime("%d %b")  # label each bar like "22 Sep"
    st.subheader("Food saved per week (kg)")
    st.caption(f"This week so far: {kg_this_week:,.1f} kg")
    # sort=False keeps the weeks in date order (not alphabetical)
    st.bar_chart(weekly.rename("Food saved (kg)"), x_label="Week starting (Monday)", y_label="kg",
                 color="#2a78d6", sort=False)

    # Step 5: the latest donations as a table (the same data in a readable form)
    st.subheader("Recent donations")
    recent = donations.sort_values("logged_at", ascending=False).head(10)
    st.dataframe(recent[["logged_at", "food_type", "quantity_kg", "ngo_name", "ngo_area", "status"]],
                 width="stretch", hide_index=True,
                 column_config={"logged_at": st.column_config.DatetimeColumn("When", format="D MMM, HH:mm"),
                                "food_type": "Food", "quantity_kg": "kg", "ngo_name": "NGO",
                                "ngo_area": "Area", "status": "Status"})


# ---------------------------------------------------------------------------
# Sidebar menu: maps each menu label to the function that draws that page
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Home": home_page,
    "🔍 Check Freshness": check_freshness_page,
    "📈 Predict Surplus": predict_surplus_page,
    "🤝 Redistribute": redistribute_page,
    "🌍 Impact Dashboard": impact_dashboard_page,
}

with st.sidebar:
    st.header("FoodBridge AI")
    choice = st.radio("Menu", list(PAGES.keys()))
    st.divider()
    st.caption("⚠️ This demo uses SAMPLE (synthetic) data.")

# Draw the page the user picked
PAGES[choice]()
