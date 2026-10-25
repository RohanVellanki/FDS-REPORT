# Fundamentals of Data Science (22CSC59) — Case Studies 3 & 6

Case Studies **3** and **6**: Jupyter notebooks with executed code, outputs and explanations; equivalent Python scripts;
a combined PDF report; and a Streamlit app for Case Study 3.

**Report:** [`report/FDS_Case_Studies_Report_Rohan_Vellanki.pdf`](FDS-Case-Studies/report/FDS_Case_Studies_Report_Rohan_Vellanki.pdf)

| | Case Study 3 | Case Study 6 |
|---|---|---|
| **Topic** | Titanic survival — Logistic Regression | Bank Marketing — comparison of 4 classifiers |
| **Dataset** | `titanic.csv` (891 passengers) | UCI `bank-additional-full.csv` (41,188 calls) |
| **Notebook** | [`Titanic_Logistic_Regression.ipynb`](FDS-Case-Studies/case-study-3-titanic-logistic-regression/Titanic_Logistic_Regression.ipynb) | [`Bank_Marketing_Classification.ipynb`](FDS-Case-Studies/case-study-6-bank-marketing-classification/Bank_Marketing_Classification.ipynb) |
| **Python code** | [`titanic_logistic_regression.py`](FDS-Case-Studies/case-study-3-titanic-logistic-regression/titanic_logistic_regression.py) | [`bank_marketing_classification.py`](FDS-Case-Studies/case-study-6-bank-marketing-classification/bank_marketing_classification.py) |
| **Extra** | Streamlit app [`app.py`](FDS-Case-Studies/case-study-3-titanic-logistic-regression/app.py) | — |

## Key results

**Case Study 3 — Logistic Regression (test set, 179 passengers)**
Accuracy 0.793 · Precision 0.767 · Recall 0.667 · F1 0.713 · ROC-AUC 0.833 (5-fold CV ROC-AUC 0.864 ± 0.020).
Sex was the dominant factor (men ≈ 12× lower odds of survival), followed by class/cabin/fare, age and siblings/spouses aboard.

**Case Study 6 — Model comparison (test set, 8,236 customers; `duration` excluded)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.899 | 0.663 | 0.210 | 0.319 | 0.800 |
| Decision Tree | 0.897 | 0.590 | 0.282 | 0.382 | 0.805 |
| KNN | 0.899 | 0.668 | 0.208 | 0.317 | 0.795 |
| Random Forest | 0.900 | 0.661 | 0.233 | 0.344 | **0.811** |
| **Random Forest, threshold 0.21** | 0.878 | 0.468 | **0.601** | **0.526** | 0.811 |

Recommended: Random Forest with a 0.21 threshold — finds 60% of subscribers from 14.5% of customers.

## How to run

```bash
pip install -r requirements.txt

# Notebooks (run from inside each case-study folder; data paths are relative)
cd FDS-Case-Studies/case-study-3-titanic-logistic-regression
jupyter notebook Titanic_Logistic_Regression.ipynb

# Streamlit app (Case Study 3) — the notebook saves the model it loads
streamlit run app.py
```

To deploy the app online, create an app on [Streamlit Community Cloud](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
from this repository with main file path `FDS-Case-Studies/case-study-3-titanic-logistic-regression/app.py`.

All results are reproducible: every split and model uses `random_state = 42`.

## Repository structure

```
README.md
requirements.txt
FDS-Case-Studies/
    case-study-3-titanic-logistic-regression/
        data/titanic.csv
        Titanic_Logistic_Regression.ipynb
        titanic_logistic_regression.py
        app.py
        model/titanic_logreg_pipeline.joblib
        figures/            # every chart used in the report
    case-study-6-bank-marketing-classification/
        data/bank-additional-full.csv
        Bank_Marketing_Classification.ipynb
        bank_marketing_classification.py
        figures/
    report/FDS_Case_Studies_Report_Rohan_Vellanki.pdf
```

## Data sources
- Titanic passenger data (`titanic.csv`) supplied with the case study.
- Bank Marketing: Moro, S., Cortez, P. & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.*
  Decision Support Systems, 62, 22–31. UCI Machine Learning Repository.
