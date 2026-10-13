"""
sample_data.py
--------------
Creates the SAMPLE (synthetic) data used by FoodBridge AI.

We don't have a real restaurant's records, so this file invents realistic-looking
data. Everything it makes is clearly labelled SAMPLE / synthetic so nobody
mistakes it for real data.

It creates two things:
  1. data/restaurant_sales.csv  - 180 days of made-up restaurant records
  2. data/ngos.csv              - 10 fictional NGOs / shelters
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Folder paths (relative to the project folder)
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
SALES_CSV = DATA_DIR / "restaurant_sales.csv"
NGOS_CSV = DATA_DIR / "ngos.csv"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# A fixed "seed" means we get the same random numbers every run,
# so results are repeatable when you demo the project.
rng = np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# 1. Restaurant sales history (for the surplus predictor)
# ---------------------------------------------------------------------------
def make_restaurant_sales(num_days: int = 180) -> pd.DataFrame:
    """
    Invent num_days of daily records for one restaurant.

    The story behind the numbers (this is how many real kitchens work):
      - Each weekday has a USUAL crowd (weekends are busier).
      - The kitchen cooks by HABIT: a planned amount for that weekday,
        not knowing exactly how many people will come.
      - The real crowd goes up and down around the usual number.
      - Fewer customers than usual  ->  more food left over (surplus).
    """
    # Day 0 is a Monday, day 1 a Tuesday ... repeating every 7 days
    day_numbers = np.arange(num_days) % 7

    # Usual number of customers for Monday ... Sunday
    usual_customers = np.array([120, 110, 115, 125, 160, 210, 190])[day_numbers]

    # Kitchen plans 0.5 kg per usual customer, give or take 5%
    food_prepared_kg = usual_customers * 0.5 * rng.uniform(0.95, 1.05, num_days)

    # The real crowd is usually within about 15% of the usual number
    customers = (usual_customers * rng.normal(1.0, 0.15, num_days)).round().clip(min=30)

    # Each customer eats about 0.45 kg (a little more or less each day).
    # You can't sell more food than you cooked, so cap it at food_prepared_kg.
    food_sold_kg = np.minimum(customers * 0.45 * rng.uniform(0.93, 1.07, num_days),
                              food_prepared_kg)

    # Round to 1 decimal place, then surplus = what was cooked but not sold
    # (computed after rounding, so every row adds up exactly)
    food_prepared_kg = food_prepared_kg.round(1)
    food_sold_kg = food_sold_kg.round(1)
    surplus_kg = (food_prepared_kg - food_sold_kg).round(1)

    return pd.DataFrame({
        "day_of_week": [DAY_NAMES[d] for d in day_numbers],
        "customers": customers.astype(int),
        "food_prepared_kg": food_prepared_kg,
        "food_sold_kg": food_sold_kg,
        "surplus_kg": surplus_kg,
    })


# ---------------------------------------------------------------------------
# 2. NGOs and shelters (for the redistribution agent)
# ---------------------------------------------------------------------------
def make_ngos() -> pd.DataFrame:
    """
    10 FICTIONAL NGOs and shelters. Names and phone numbers are made up.

    area                - the part of the city they are in
    capacity_kg         - the most food they can collect and serve in one day
    accepts_cooked_food - cooked meals must be eaten quickly, so only places
                          that serve meals (kitchens, shelters) can take them.
                          Food banks only take raw or packaged food.
    """
    return pd.DataFrame([
        # name,                          area,      capacity_kg, accepts_cooked_food, phone
        ["Sample Hope Kitchen",          "Central", 40, True,  "+00 000 000 001"],
        ["Sample Central Food Bank",     "Central", 80, False, "+00 000 000 002"],
        ["Sample Night Shelter",         "North",   25, True,  "+00 000 000 003"],
        ["Sample Children's Home",       "North",   15, True,  "+00 000 000 004"],
        ["Sample Community Fridge",      "South",   20, False, "+00 000 000 005"],
        ["Sample Women's Shelter",       "South",   30, True,  "+00 000 000 006"],
        ["Sample Elder Care Centre",     "East",    10, True,  "+00 000 000 007"],
        ["Sample Street Meals Trust",    "East",    50, True,  "+00 000 000 008"],
        ["Sample Migrant Workers Kitchen", "West",  35, True,  "+00 000 000 009"],
        ["Sample West Food Bank",        "West",    60, False, "+00 000 000 010"],
    ], columns=["name", "area", "capacity_kg", "accepts_cooked_food", "phone"])


# ---------------------------------------------------------------------------
def create_all_sample_data() -> None:
    """Create every sample file. Called by train_surplus.py."""
    DATA_DIR.mkdir(exist_ok=True)

    make_restaurant_sales().to_csv(SALES_CSV, index=False)
    print(f"[SAMPLE DATA] Wrote {SALES_CSV.relative_to(PROJECT_DIR)}")

    make_ngos().to_csv(NGOS_CSV, index=False)
    print(f"[SAMPLE DATA] Wrote {NGOS_CSV.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    create_all_sample_data()
