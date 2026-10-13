"""
agent.py
--------
AI part 3: the REDISTRIBUTION AGENT.

An "agent" here is a program that looks at a situation, makes a chain of
decisions, takes actions, and explains each step. Ours has 4 steps:

  Step 1  SAFETY CHECK  Did the food pass the freshness check? Is the deadline still ahead?
  Step 2  MATCH         Which NGO is the best fit, by area and by capacity left today?
  Step 3  DRAFT         Write a short pickup request: with an LLM (Claude) if an API
                        key is given, otherwise with a fill-in-the-blanks template.
  Step 4  LOG           Save the donation to data/donations.csv (used by the Impact Dashboard).

If a step fails, the agent stops and says why. It never skips the safety check.
"""

import datetime as dt
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
NGOS_CSV = PROJECT_DIR / "data" / "ngos.csv"
DONATIONS_CSV = PROJECT_DIR / "data" / "donations.csv"

# Food must be at least this confidently "Fresh" before we donate it
MIN_FRESH_CONFIDENCE = 0.80

# The food types a restaurant can choose. Only "Cooked meals" needs an NGO
# that accepts cooked food (it spoils fast and must be served the same day).
FOOD_TYPES = ["Cooked meals", "Raw fruits & vegetables", "Bread & bakery", "Packaged food"]

LLM_MODEL = "claude-opus-5-5"


# ---------------------------------------------------------------------------
# Step 1: safety check
# ---------------------------------------------------------------------------
def step1_safety_check(freshness: tuple[str, float] | None, deadline: dt.datetime) -> tuple[bool, str]:
    """freshness is (label, confidence) from the Check Freshness page, or None."""
    if freshness is None:
        return False, "No freshness check found. Check the food on the Check Freshness page first."

    label, confidence = freshness
    if label != "Fresh":
        return False, f"Food was classified as {label} ({confidence:.0%}). It must NOT be donated."
    if confidence < MIN_FRESH_CONFIDENCE:
        return False, (f"Food looks fresh but the model is only {confidence:.0%} sure "
                       f"(minimum {MIN_FRESH_CONFIDENCE:.0%}). Ask staff to inspect it.")
    if deadline <= dt.datetime.now():
        return False, f"The pickup deadline ({deadline:%d %b %H:%M}) has already passed."

    return True, f"Food passed the freshness check: Fresh, {confidence:.0%} confidence."


# ---------------------------------------------------------------------------
# Step 2: pick the best NGO
# ---------------------------------------------------------------------------
def capacity_left_today(ngos: pd.DataFrame) -> pd.Series:
    """Each NGO's daily capacity minus what we've already sent them today."""
    used = pd.Series(0.0, index=ngos["name"])
    if DONATIONS_CSV.exists():
        log = pd.read_csv(DONATIONS_CSV)
        today = log[log["logged_at"].str.startswith(dt.date.today().isoformat())]
        used = used.add(today.groupby("ngo_name")["quantity_kg"].sum(), fill_value=0)
    return ngos["capacity_kg"] - ngos["name"].map(used).fillna(0)


def step2_find_best_ngo(ngos: pd.DataFrame, food_type: str, quantity_kg: float,
                        area: str) -> tuple[pd.Series | None, str, pd.DataFrame]:
    """
    Rules, in order:
      a) The NGO must accept this food type (cooked meals need a meal-serving NGO).
      b) The NGO must have enough capacity left today for the whole quantity.
      c) Prefer NGOs in the same area (shorter trip, food stays fresh).
      d) Among those, choose the "best fit": the smallest capacity that still fits,
         so large NGOs stay free for large donations.
    Returns (best NGO or None, explanation, table of all NGOs with the reason for each).
    """
    table = ngos.copy()
    table["capacity_left_kg"] = capacity_left_today(table)
    table["same_area"] = table["area"] == area

    # Write down why each NGO is or isn't suitable (shown in the app)
    def reason(row):
        if food_type == "Cooked meals" and not row["accepts_cooked_food"]:
            return "❌ doesn't accept cooked food"
        if row["capacity_left_kg"] < quantity_kg:
            return f"❌ only {row['capacity_left_kg']:.0f} kg capacity left today"
        return "✅ same area" if row["same_area"] else "✅ different area"
    table["reason"] = table.apply(reason, axis=1)

    # Rules a + b: keep suitable NGOs.  Rules c + d: sort them.
    suitable = table[table["reason"].str.startswith("✅")]
    suitable = suitable.sort_values(["same_area", "capacity_left_kg"], ascending=[False, True])
    ranked = pd.concat([suitable, table.drop(suitable.index)])  # suitable ones first

    if suitable.empty:
        return None, (f"No NGO can take {quantity_kg:.1f} kg of {food_type.lower()} today. "
                      "Try splitting it into smaller donations."), ranked

    best = suitable.iloc[0]
    where = "in your area" if best["same_area"] else f"in {best['area']} (none suitable in {area})"
    return best, (f"Best match: {best['name']} {where}, "
                  f"{best['capacity_left_kg']:.0f} kg capacity left today."), ranked


# ---------------------------------------------------------------------------
# Step 3: draft the pickup message
# ---------------------------------------------------------------------------
def template_message(d: dict, ngo: pd.Series) -> str:
    """Fill-in-the-blanks message. Always works, no internet needed."""
    return (
        f"Hello {ngo['name']} team, this is {d['restaurant']} ({d['area']}). "
        f"We have {d['quantity_kg']:.1f} kg of fresh {d['food_type'].lower()} ready to donate "
        f"(freshness checked: {d['freshness_confidence']:.0%}). "
        f"Could you pick it up before {d['deadline']:%I:%M %p on %d %b}? "
        f"Please reply YES to confirm. Thank you! - {d['restaurant']}, via FoodBridge AI"
    )


def llm_message(d: dict, ngo: pd.Series, api_key: str) -> str:
    """Ask Claude to write a friendlier, more natural message from the same facts."""
    import anthropic  # imported here so the app still runs without it

    client = anthropic.Anthropic(api_key=api_key)
    facts = (
        f"Restaurant: {d['restaurant']} (area: {d['area']})\n"
        f"NGO: {ngo['name']} (area: {ngo['area']})\n"
        f"Food: {d['quantity_kg']:.1f} kg of {d['food_type'].lower()}\n"
        f"Freshness check: passed, {d['freshness_confidence']:.0%} confidence\n"
        f"Pickup deadline: {d['deadline']:%I:%M %p on %d %b}"
    )
    response = client.beta.messages.create(
        model=LLM_MODEL,
        max_tokens=4000,
        output_config={"effort": "low"},  # a short message doesn't need deep thinking
        # If Claude declines for safety reasons, automatically retry on a fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=("You write short pickup request messages (SMS/WhatsApp) from a restaurant to a "
                "food-rescue NGO. Be warm and clear. Under 70 words, plain text, no subject line, "
                "no placeholders. Use only the facts given; don't invent any. "
                "End by asking them to reply to confirm."),
        messages=[{"role": "user", "content": facts}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined to write the message")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise RuntimeError("the model returned an empty message")
    return text


def step3_draft_message(d: dict, ngo: pd.Series, api_key: str | None) -> tuple[str, str, str]:
    """Returns (message, source, explanation). source is "llm" or "template"."""
    if not api_key:
        return template_message(d, ngo), "template", "No API key given, so used the message template."

    import anthropic
    try:
        return llm_message(d, ngo, api_key), "llm", f"Message written by Claude ({LLM_MODEL})."
    except anthropic.AuthenticationError:
        problem = "the API key was rejected"
    except anthropic.RateLimitError:
        problem = "the API is busy (rate limit)"
    except anthropic.APIConnectionError:
        problem = "couldn't reach the API (no internet?)"
    except (anthropic.APIStatusError, RuntimeError) as error:
        problem = str(error)
    # Any problem with the LLM: fall back to the template, so the agent never gets stuck
    return template_message(d, ngo), "template", f"LLM unavailable ({problem}), so used the template."


# ---------------------------------------------------------------------------
# Step 4: log the donation
# ---------------------------------------------------------------------------
def step4_log_donation(d: dict, ngo: pd.Series, message: str, source: str) -> str:
    """Add one row to data/donations.csv (the file is created the first time)."""
    row = pd.DataFrame([{
        "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
        "restaurant": d["restaurant"],
        "food_type": d["food_type"],
        "quantity_kg": d["quantity_kg"],
        "restaurant_area": d["area"],
        "pickup_deadline": d["deadline"].isoformat(timespec="minutes"),
        "ngo_name": ngo["name"],
        "ngo_area": ngo["area"],
        "ngo_phone": ngo["phone"],
        "freshness_check": d["freshness_check"],
        "freshness_confidence": round(d["freshness_confidence"], 3),
        "message_source": source,
        "status": "pickup requested",
        "message": message,
    }])
    # header=True only when the file is new, so the column names are written once
    row.to_csv(DONATIONS_CSV, mode="a", index=False, header=not DONATIONS_CSV.exists())
    return f"Logged {d['quantity_kg']:.1f} kg to {ngo['name']} in data/donations.csv."


# ---------------------------------------------------------------------------
# Run all 4 steps in order
# ---------------------------------------------------------------------------
def run_agent(restaurant: str, food_type: str, quantity_kg: float, area: str,
              deadline: dt.datetime, freshness: tuple[str, float] | None,
              api_key: str | None = None, demo_mode: bool = False) -> dict:
    """
    demo_mode=True skips the photo check FOR TESTING ONLY. It is recorded in
    the log so a demo donation is never mistaken for a real, checked one.

    Returns a dict:
      success  - True if a pickup request was drafted and logged
      steps    - list of (step name, passed?, explanation) to show the user
      ngo      - the chosen NGO (or None)
      ranked   - table of all NGOs and why each was / wasn't chosen
      message, message_source
    """
    result = {"success": False, "steps": [], "ngo": None, "ranked": None,
              "message": None, "message_source": None}
    steps = result["steps"]

    # Step 1
    if demo_mode:
        freshness = ("Fresh", 0.95)  # pretend result, only for testing the agent
    ok, note = step1_safety_check(freshness, deadline)
    if ok and demo_mode:
        note = "DEMO MODE: freshness photo check skipped for testing. Don't send this message."
    steps.append(("1. Safety check", ok, note))
    if not ok:
        return result

    # Step 2
    ngos = pd.read_csv(NGOS_CSV)
    ngo, note, ranked = step2_find_best_ngo(ngos, food_type, quantity_kg, area)
    result["ranked"] = ranked
    steps.append(("2. Match NGO", ngo is not None, note))
    if ngo is None:
        return result

    # Step 3
    details = {"restaurant": restaurant, "food_type": food_type, "quantity_kg": quantity_kg,
               "area": area, "deadline": deadline, "freshness_confidence": freshness[1],
               "freshness_check": "demo (skipped)" if demo_mode else "model"}
    message, source, note = step3_draft_message(details, ngo, api_key)
    steps.append(("3. Draft message", True, note))

    # Step 4
    note = step4_log_donation(details, ngo, message, source)
    steps.append(("4. Log donation", True, note))

    result.update(success=True, ngo=ngo, message=message, message_source=source)
    return result
