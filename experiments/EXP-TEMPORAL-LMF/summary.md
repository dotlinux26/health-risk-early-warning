# EXP-TEMPORAL-LMF — Validation theo thời gian với outcome tử vong thật

- Nguồn: NHANES Public-Use LMF 2019 (train 2015-16 n=5048, test 2017-18 n=4773)
- Outcome: tử vong toàn bộ ≤12 tháng (train prevalence 0.97%, test 1.40%)

| Model | Split | ROC-AUC | AUPRC | Brier | ECE |
|---|---|---|---|---|---|
| LR | temporal | 0.8209 | 0.0794 | 0.01364 | 0.00391 |
| LR | random | 0.8411 | 0.0839 | 0.01221 | 0.00262 |
| LGBM | temporal | 0.7709 | 0.0459 | 0.0142 | 0.01168 |
| LGBM | random | 0.781 | 0.0662 | 0.01247 | 0.00928 |

- C-index (≤60 tháng): LR 0.8217, LGBM 0.7763
- Lead time: top-20% nguy cơ bắt 53/94 ca tử vong ≤24m, median 9.0 tháng trước sự kiện