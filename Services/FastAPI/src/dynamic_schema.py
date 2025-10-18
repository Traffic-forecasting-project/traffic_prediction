'''
__author__ = "Georges Nassopoulos"
__contributors__ = "Mateo Villa Arias"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Dynamic Pydantic schema builder for features based on trained model or fallback JSON."
'''

## ============================
## Imports
## ============================
import os
import json
import joblib
from pydantic import create_model, BaseModel
from typing import Dict, Any

from src.constants import MODEL_PATH
from src.logging_utils import get_logger

## JSON fallback path
from pathlib import Path
FEATURE_FILE = Path("resources/feature_importances.json")

## Setup logger
logger = get_logger(__name__)

## ============================
## Helpers
## ============================
def load_feature_order() -> list[str]:
    """
        Load the list of features from model (if available) or from JSON fallback

        Returns:
            list[str]: Ordered list of features
    """
    
    ## Try model first
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            if hasattr(model, "feature_names_in_"):
                feature_order = list(model.feature_names_in_)
                logger.info(f"Loaded feature order from model: {feature_order}")

                ## Save as fallback JSON
                FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(FEATURE_FILE, "w", encoding="utf-8") as f:
                    json.dump(feature_order, f, indent=2, ensure_ascii=False)
                return feature_order
        except Exception as e:
            logger.warning(f"Could not load feature order from model: {e}")

    ## Fallback to JSON
    if FEATURE_FILE.exists():
        try:
            with open(FEATURE_FILE, "r", encoding="utf-8") as f:
                feature_order = json.load(f)
            logger.info(f"Loaded feature order from JSON fallback: {feature_order}")
            return feature_order
        except Exception as e:
            logger.error(f"Failed to load features from JSON: {e}")

    return []

def build_dynamic_pydantic_model() -> BaseModel:
    """
        Build a dynamic Pydantic BaseModel using the feature order list

            - Defaults: float for continuous, int for categorical
            - Ensures that API schema matches current trained model

        Returns:
            BaseModel: Generated Pydantic model
    """
    
    feature_order = load_feature_order()
    if not feature_order:
        logger.warning("No features available for dynamic schema, using empty fallback model.")
        return create_model("DynamicFeatures")

    ## Build fields dictionary (simplistic heuristic: int if "hour"/"weekday"/"count", else float)
    fields: Dict[str, tuple[Any, ...]] = {}
    for feat in feature_order:
        if any(key in feat.lower() for key in ["hour", "weekday", "count", "category", "number", "version"]):
            fields[feat] = (int, ...)
        else:
            fields[feat] = (float, ...)

    logger.info(f"Building dynamic schema with features: {list(fields.keys())}")
    return create_model("DynamicFeatures", **fields)


## Build dynamic model once at import
DynamicFeatures = build_dynamic_pydantic_model()
