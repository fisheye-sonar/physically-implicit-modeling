# oth-mse-smoke — is the MSE head a distribution?

native output kind `raw`; 200 held-out test games, 11,751 positions

| raw output over the 60 move outputs | mean | p05 | p50 | p95 |
|---|---|---|---|---|
| sum | 1.0181 | 0.9862 | 1.0208 | 1.0391 |
| neg_mass | 0.0064 | -0.0000 | 0.0003 | 0.0353 |
| min | -0.0013 | -0.0117 | -0.0003 | 0.0047 |
| max | 0.0503 | 0.0390 | 0.0467 | 0.0720 |
| l1_to_clipnorm | 0.0318 | 0.0059 | 0.0263 | 0.0714 |

argmax legal: 0.1289

| gates | raw | clipnorm | Bayes / CE-model reference |
|---|---|---|---|
| legal mass | 0.1698 | 0.1659 | 1.0 / 0.93 |
| top-1 legal | 0.1289 | 0.1289 | 1.0 / 0.987 |
| CE | 4.0424 | 4.0663 | 2.0061 (Bayes) |

| editability (canonical best arms) | raw | clipnorm |
|---|---|---|
| unedited | -0.026 | -0.026 |
| PI pt4·α1 | -0.026 / 1.00 | -0.026 / 1.00 |
| ND pt4·α1 | -0.010 / 4.84 | -0.026 / 1.16 |
| GS pt4·α1 | -0.027 / 1.07 | -0.026 / 1.02 |
