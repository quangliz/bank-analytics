# Bank Risk & Engagement Analytics

This is a small ML project about bank customer churn and segmentation.  
I tried to do end-to-end steps: data prep, modeling, and simple clustering.

## What this project does

- Predict churn (risk modeling)
- Segment customers (K-Means)
- Save models and reports

## Dataset

Kaggle dataset: https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers

## Run pipeline (python scripts)

```bash
python -m src.pipeline.run_pipeline
```

## Project structure

```
bank-analytics/
├── data/            # raw and processed data
├── notebooks/       # 01-04 notebooks
├── src/             # python modules
└── config/          # config files
```

## Further work:
- [ ] Implement explainability with SHAP