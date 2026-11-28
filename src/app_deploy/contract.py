"""
layer 1: define the deployed app's input:
Input: Listing inputs (from models.features) + statistics of each column
Process: No actually
Output: a json file (contract.json) contains information about:
            - columns in general
            - num_cols (statistics)
            - cat_cols (statistics)

"""

import json
from pathlib import Path

CONTRACT_PATH = Path(__file__).with_name("contract.json")


def load_contract(path=CONTRACT_PATH):
    contract = json.loads(path.read_text(encoding="utf-8"))
    if len(contract["columns"]) != contract["n_columns"]:
        raise ValueError(f"{path} inconsistent; rebuild with build_contract")
    return contract


def specs(contract):
    return {**contract["num_cols"], **contract["cat_cols"]}


def required_columns(contract):
    spec = specs(contract)
    return [name for name in contract["columns"] if spec[name]["required"]]


def defaults(contract):
    spec = specs(contract)
    return {name: spec[name]["mode"] for name in contract["columns"]}
