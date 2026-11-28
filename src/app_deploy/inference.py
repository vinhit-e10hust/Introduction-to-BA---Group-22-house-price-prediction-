"""
@Thang

layer 3: Inference layer

Input:
    - load models(str) : .joblib file paths; cached in bundle
    - dataframe(df): returned by read_housing()
                    (this is the unseen data, could include multiple rows),
                    same as layer 2's input
Process:
    - load model: take the input of joblib file paths, return dict
    - predict: take the input of model and df -> output

Output: (of predict)
    - result (df): the predicted result by the predict.py
    - runtime metadata (dict): latency, ...

=======note
1. the loaded model (dict), key:val:
        - estimator : the model
        - log_radius : to calculate lower and upper (idc)
        - model: name of the model
        - parameters : dict of hyperparams: learning_rate, max_depth, ...

"""

import time
from functools import lru_cache
from pathlib import Path

import joblib

from src.models.predict import MODEL_PATH, predict_prices


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    return joblib.load(Path(path))


def predict(bundle, df):
    start = time.perf_counter()
    result = predict_prices(bundle, df)
    latency_ms = round(1000 * (time.perf_counter() - start), 2)
    price = result["PredictedSalePrice"]
    return result, {
        "model": bundle["model"],
        "log_radius": bundle["log_radius"],
        "n_rows": len(df),
        "latency_ms": latency_ms,
        "price_min": float(price.min()),
        "price_median": float(price.median()),
        "price_max": float(price.max()),
    }


if __name__ == "__main__":
    import sys

    from src.data.common import read_housing

    result, meta = predict(load_model(), read_housing(sys.argv[1]))
    for key, value in meta.items():
        print(f"{key:14} {value}")
    print()
    print(result.head(10).to_string(index=False))
