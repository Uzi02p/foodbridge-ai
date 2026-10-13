"""
surplus_model.py
----------------
AI part 2: predicts tomorrow's food surplus (kg) with Linear Regression.
(The training itself is run from train_surplus.py.)

Inputs (features):  day of the week, expected number of customers
Output (target):    surplus_kg - how much cooked food will probably be left over

Linear regression learns one "weight" per input and adds them up:
    surplus = (weight x customers) + (a value for that weekday) + constant
For example, a weight of -0.4 on customers means "each extra customer
reduces the leftover food by about 0.4 kg". That makes it easy to explain.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "surplus_model.joblib"
METRICS_PATH = PROJECT_DIR / "models" / "surplus_metrics.json"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def make_features(day_of_week, customers) -> pd.DataFrame:
    """
    Turn raw inputs into the table of numbers the model reads.

    A model only understands numbers, not words like "Monday". So we make
    7 yes/no columns (is_Monday, is_Tuesday ... is_Sunday). This is called
    "one-hot encoding". Each weekday then gets its own learned value.

    Works for one day (lists of length 1) or many days (whole columns).
    """
    features = pd.DataFrame({"customers": pd.Series(customers, dtype=float)})
    days = pd.Series(day_of_week).reset_index(drop=True)
    for name in DAY_NAMES:
        features[f"is_{name}"] = (days == name).astype(int)
    return features


def train(sales: pd.DataFrame) -> dict:
    """Train on past days, save the model, and return how accurate it is."""
    # Step 1: inputs (X) and the answer we want to predict (y)
    X = make_features(sales["day_of_week"], sales["customers"])
    y = sales["surplus_kg"]

    # Step 2: keep 20% of the days aside to test on data the model hasn't seen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: fit the line (find the best weights)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: check it on the test days
    predictions = model.predict(X_test)

    # Step 5: save the trained model to a file the app can load
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {
        # R²: 1.0 = perfect, 0 = no better than always guessing the average
        "r2_score": float(r2_score(y_test, predictions)),
        # MAE: on average, how many kg the prediction is off by
        "mae_kg": float(mean_absolute_error(y_test, predictions)),
        # Weight on customers: kg of surplus change per extra customer
        "customer_weight": float(model.coef_[0]),
        "num_train_days": len(X_train),
        "num_test_days": len(X_test),
    }


def load_model() -> LinearRegression:
    """Load the trained model from disk."""
    return joblib.load(MODEL_PATH)


def predict(model: LinearRegression, day_of_week: str, customers: int) -> float:
    """Predict surplus in kg for one day, e.g. predict(model, "Friday", 150).
    Never returns a negative amount (you can't have less than zero leftovers)."""
    X = make_features([day_of_week], [customers])
    return max(0.0, float(model.predict(X)[0]))
