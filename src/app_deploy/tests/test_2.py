import json

import joblib

from src.data.common import read_housing

bundle = joblib.load("models/house_price.joblib")
data = read_housing("data/processed/train_cleaned.csv")
config = json.loads(bundle["estimator"]["model"].regressor_.get_booster().save_config())

print("base_score", config["learner"]["learner_model_param"]["base_score"])
print("mean on train", round(float(data["SalePrice"].mean()), 2))

"""
base_score 5E-1
mean on train 180921.2

"""



