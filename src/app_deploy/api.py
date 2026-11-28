
"""
@Thang

API :
- get model_info: get the metadata of model
- post predict

**: for docker only, streamlit sdk does not need this

"""
import json
from dataclasses import asdict

from fastapi import FastAPI, UploadFile

from src.app_deploy.contract import load_contract
from src.app_deploy.inference import load_model, predict
from src.app_deploy.validation import validate
from src.data.common import read_housing

api = FastAPI()


@api.get("/")
def model_info():
    return {
        key: value
        for key, value in load_model().items()
        if key not in ("estimator", "input_columns")
    }


@api.post("/predict")
def predict_csv(file: UploadFile):
    df = read_housing(file.file)
    report = validate(df, load_contract())
    body = {
        "is_valid": report.is_valid,
        "n_rows": report.n_rows,
        "n_missing_cells": report.n_missing_cells,
        "errors": [asdict(issue) for issue in report.errors],
        "warnings": [asdict(issue) for issue in report.warnings],
    }
    if report.errors:
        return body
    result, meta = predict(load_model(), df)
    return body | {"meta": meta, "predictions": json.loads(result.to_json(orient="records"))}
