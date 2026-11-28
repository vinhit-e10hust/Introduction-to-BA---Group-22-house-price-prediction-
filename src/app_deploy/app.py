"""
@Thang

Streamlit app:
- the app includes 3 main layers: contract -> validation -> inference. 
- the description of each layer is in the comment of each .py file

the app flow:
    1. Init app (runnable)
    2. csv_uploader (to upload input) 
    3. call validation (layer 2)
    4. call inference (layer 3)
    5. export results !

"""


import streamlit as st

from src.app_deploy.contract import load_contract
from src.app_deploy.inference import load_model, predict
from src.app_deploy.validation import validate
from src.data.common import read_housing

uploaded = st.file_uploader("csv", type="csv")
if uploaded is None:
    st.stop()

uploaded.seek(0)
df = read_housing(uploaded)
report = validate(df, load_contract())

st.text(
    "\n".join(
        [
            f"is_valid={report.is_valid}",
            f"n_rows={report.n_rows}",
            f"n_missing_cells={report.n_missing_cells}",
            *(str(issue) for issue in report.errors + report.warnings),
        ]
    )
)

if report.errors:
    st.stop()

result, meta = predict(load_model(), df)
st.text("\n".join(f"{key}={value}" for key, value in meta.items()))
st.dataframe(result)
st.download_button("csv", result.to_csv(index=False), "predictions.csv", "text/csv")
