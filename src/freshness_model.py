"""
freshness_model.py
------------------
AI part 1: USING the trained freshness classifier inside the app.
(The training itself happens in train_freshness.py.)

The model is MobileNetV2 + one small layer we trained on fresh vs rotten
food photos. Given a photo it outputs one number between 0 and 1:
the probability that the food is STALE.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "freshness.h5"

IMAGE_SIZE = (224, 224)            # MobileNetV2's standard input size
CLASS_NAMES = ["Fresh", "Stale"]   # label 0 = Fresh, label 1 = Stale
SAFE_CONFIDENCE = 0.80             # need at least 80% sure "Fresh" to donate


def load_model() -> tf.keras.Model:
    """Load the trained model file from disk."""
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model: tf.keras.Model, image: Image.Image) -> tuple[str, float]:
    """
    Classify one photo.
    Returns (label, confidence), e.g. ("Fresh", 0.93) = 93% sure it's fresh.
    """
    # Make the photo match what the model was trained on: RGB, 224x224
    image = image.convert("RGB").resize(IMAGE_SIZE)
    # The model expects a batch of images, so wrap our one image: shape (1, 224, 224, 3)
    batch = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

    prob_stale = float(model.predict(batch, verbose=0)[0][0])
    if prob_stale >= 0.5:
        return "Stale", prob_stale
    return "Fresh", 1.0 - prob_stale


def recommendation(label: str, confidence: float) -> tuple[str, str]:
    """
    Turn the prediction into advice. Returns (level, message), where level is
    "safe", "check" or "unsafe" so the app can colour it green/yellow/red.
    """
    if label == "Fresh" and confidence >= SAFE_CONFIDENCE:
        return "safe", "✅ Safe to donate. Do a quick smell and look check before packing."
    if label == "Fresh":
        return "check", ("⚠️ Not sure. The model thinks it's fresh but isn't confident. "
                         "Ask staff to inspect it before donating.")
    return "unsafe", "❌ Do not donate. Compost it or send it for animal feed."
