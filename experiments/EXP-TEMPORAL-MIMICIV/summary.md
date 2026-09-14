# MIMIC-IV Temporal Validation Results

## Dataset
- Source: MIMIC-IV v3.1 (core files)
- Temporal split: Train shifted years 2105-2115 (n=22552), Test 2116-2214 (n=523476)
- Features: systolic_bp, diastolic_bp, bmi, weight, height, egfr + 17 comorbidity flags
- Labels: mortality_30d (30-day all-cause mortality)
- Train prevalence: 0.0256
- Test prevalence: 0.0200

## Results

| Model | Temporal AUC | Random AUC | Δ AUC | Temporal AUPRC | Brier | ECE |
|-------|-------------|------------|-------|----------------|-------|-----|
| Logistic Regression | 0.7516 | 0.7614 | -0.0098 | 0.0586 | 0.2069 | 0.3843 |
| LightGBM | 0.7508 | 0.7839 | -0.0331 | 0.0609 | 0.1646 | 0.3513 |

## Lead Time (Top Quintile Capture Rate)
- LR: 52.78%
- LightGBM: 54.25%

## Missingness (Train)
- systolic_bp: 61.04%
- diastolic_bp: 61.04%
- bmi: 57.81%
- weight: 54.62%
- height: 61.05%
- egfr: 99.98%
- All 17 comorbidities: 0.00% (from ICD codes)
