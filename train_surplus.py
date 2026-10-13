"""
train_surplus.py
----------------
Creates the SYNTHETIC data and trains the SURPLUS PREDICTOR (AI part 2).

Run it with:
    python train_surplus.py

Steps:
  1. Create the synthetic CSVs: data/restaurant_sales.csv and data/sample_ngos.csv
  2. Train a linear regression model on restaurant_sales.csv
  3. Print R² and MAE, and save the model to models/surplus_model.joblib
"""

import json

import pandas as pd

from src import sample_data, surplus_model

# Step 1: make the synthetic data
print("=== Step 1: Creating SYNTHETIC data ===")
sample_data.create_all_sample_data()
sales = pd.read_csv(sample_data.SALES_CSV)
print(f"{len(sales)} days of records. First 5 rows:")
print(sales.head().to_string(index=False))

# Step 2: train the model
print("\n=== Step 2: Training Linear Regression ===")
metrics = surplus_model.train(sales)
print(f"Trained on {metrics['num_train_days']} days, tested on {metrics['num_test_days']} unseen days.")

# Step 3: report how good it is, and save the numbers for the app to show
print("\n=== Step 3: Results ===")
print(f"R² score: {metrics['r2_score']:.2f}   (1.00 = perfect, 0 = no better than guessing the average)")
print(f"MAE:      {metrics['mae_kg']:.2f} kg  (average error per prediction)")
print(f"Each extra customer changes surplus by {metrics['customer_weight']:+.2f} kg")

surplus_model.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
print(f"\nSaved model to models/{surplus_model.MODEL_PATH.name}")
