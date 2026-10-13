# 🍲 FoodBridge AI

A solo internship project for **UN SDG 2: Zero Hunger**. It helps restaurants check food
freshness, predict surplus, and send leftover food to NGOs and shelters.

> ⚠️ The restaurant sales data and NGO list are **synthetic (made-up) data**, not real records.
> The freshness model is trained on a real public photo dataset (see below).

## Run it (Python 3.11)

```bash
cd foodbridge-ai
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_surplus.py        # synthetic data + surplus model (seconds)
python train_freshness.py      # freshness model (needs the dataset below)
streamlit run app.py
```

## Freshness dataset

1. Download **"Fruits fresh and rotten for classification"** from Kaggle (free account, ~3.5 GB):
   https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
2. Unzip it. Move the `train` and `test` folders from inside `dataset/` so you get:

```
foodbridge-ai/data/fresh_rotten/
├── train/
│   ├── freshapples/   freshbanana/   freshoranges/
│   └── rottenapples/  rottenbanana/  rottenoranges/
└── test/
    └── (same 6 folders)
```

Folders starting with `fresh` count as **Fresh** and folders starting with `rotten` count as **Stale**.
If you run `python train_freshness.py` without the dataset, it prints these instructions.
Training takes about 10–30 minutes on a laptop CPU. Use `--epochs 2` for a faster run.

## What each file does

| File / folder | Purpose |
|---|---|
| `app.py` | The Streamlit web app. Sidebar menu: Home, Check Freshness, Predict Surplus, Redistribute, Impact Dashboard. All pages work. |
| `requirements.txt` | The Python libraries to install. |
| `train_freshness.py` | Trains the freshness classifier: MobileNetV2 transfer learning. Prints accuracy and a confusion matrix, then saves `models/freshness.h5`. |
| `train_surplus.py` | Creates the synthetic CSVs, trains the surplus predictor (linear regression), prints R² and MAE, saves `models/surplus_model.joblib`. |
| `agent.py` | The redistribution agent, in 4 steps: safety check, match NGO, draft message (Claude or template), log donation. |
| `src/freshness_model.py` | Loads the freshness model, predicts Fresh/Stale with a confidence score, and gives a recommendation (safe / check / do not donate). |
| `src/surplus_model.py` | Turns day + customers into model inputs, trains the linear regression, predicts surplus in kg. |
| `src/sample_data.py` | Generates `data/restaurant_sales.csv` (180 synthetic days) and `data/ngos.csv` (10 fictional NGOs). |
| `data/` | Synthetic CSVs, `donations.csv` (created by the agent), and the Kaggle dataset you download into `data/fresh_rotten/`. |
| `models/` | Trained models, confusion matrix image and accuracy numbers. |
| `archive/app_v1_full.py` | An old all-in-one app version, kept for reference only. It no longer runs. |

## Surplus predictor

`data/restaurant_sales.csv` has 180 synthetic days with the columns `day_of_week, customers,
food_prepared_kg, food_sold_kg, surplus_kg`. How the data was made up: the kitchen cooks a
habitual amount for each weekday, the real crowd varies around the usual number, and
surplus = prepared − sold. So fewer customers than usual means more leftovers.

The model learns surplus from **day of week + customers**. The weekday is turned into 7 yes/no
columns (one-hot encoding). On 36 unseen test days it scores **R² 0.82** with an average error
(MAE) of **2.5 kg**. A straight line can't be trusted far outside the data it learned from, so
the app warns when the customer number is unusual for that weekday.

## Redistribution agent (`agent.py`)

The restaurant enters the food type, quantity (kg), area and pickup deadline. The agent then works
through 4 steps and shows the result of each one:

1. **Safety check**: the food must have passed the freshness check (Fresh, at least 80% confident)
   and the deadline must still be ahead. If not, the agent stops: unsafe food is never donated.
2. **Match NGO**: from `data/ngos.csv`, keep NGOs that accept the food type (cooked meals need a
   meal-serving NGO) and have enough **capacity left today** (daily capacity minus what was already
   sent today). Prefer the same area, then the smallest capacity that fits, so big NGOs stay free for big donations.
3. **Draft message**: if an Anthropic API key is given, Claude (`claude-opus-5-5`) writes the
   message from the facts only. Without a key, or if the API fails for any reason, it uses a template.
4. **Log donation**: adds a row to `data/donations.csv`.

**Using Claude (optional):** set the key before starting the app, or paste it on the Redistribute page
(it's never saved):

```bash
export ANTHROPIC_API_KEY="your-key"   # get one at https://console.anthropic.com
streamlit run app.py
```

**Demo mode:** a checkbox on the page skips the photo check so you can try the agent before training
the freshness model. Demo donations are marked `demo (skipped)` in the log.

## Impact Dashboard

Built from `data/donations.csv` (written by the agent). It shows:

- **Food saved** (total kg), **meals served** (estimate: total kg ÷ **0.4 kg per meal**),
  **number of donations**, and **active NGO partners** (NGOs that received at least one donation)
- **Food saved per week**: weeks start on Monday, and weeks with no donations show as 0
- the 10 most recent donations

Demo donations are left out unless you switch on the toggle. The 0.4 kg-per-meal assumption is
`MEAL_KG` at the top of `app.py`, and the Predict Surplus page uses the same value.

## How the recommendation works

| Model says | Recommendation |
|---|---|
| Fresh, ≥ 80% confident | ✅ Safe to donate (after a quick human check) |
| Fresh, < 80% confident | ⚠️ Not sure: staff should inspect it |
| Stale | ❌ Do not donate |

The model has only seen apples, bananas and oranges, so it may be wrong on other foods such as
cooked meals. A person should always make the final food-safety decision.
