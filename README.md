# BA-Group-22-house-price-prediction-
An end-to-end Business Analytics project for house price prediction, covering data engineering, EDA, feature engineering, machine learning, model deployment, and performance monitoring.
house-price-prediction/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── synthetic/
│   │   └── contextual_features.csv
│   └── processed/
│       ├── train_extended.csv
│       ├── test_extended.csv
│       └── train_cleaned.csv
│
├── src/
│   └── data/
│       ├── profile_data.py
│       ├── generate_synthetic.py
│       ├── integrate_data.py
│       ├── clean_data.py
│       └── validate_data.py
│
├── notebooks/
│   └── 01_data_profiling.ipynb
│
├── docs/
│   ├── data_dictionary.xlsx
│   ├── synthetic_assumptions.md
│   └── cleaning_report.md
│
├── requirements.txt
└── README.md
