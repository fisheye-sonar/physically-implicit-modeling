# L-oth-20m-mse — is the MSE head a distribution?

native output kind `raw`; 10,000 held-out test games, 589,564 positions

| raw output over the 60 move outputs | mean | p05 | p50 | p95 |
|---|---|---|---|---|
| sum | 1.0001 | 0.9983 | 0.9997 | 1.0032 |
| neg_mass | 0.0400 | 0.0262 | 0.0375 | 0.0643 |
| min | -0.0046 | -0.0070 | -0.0043 | -0.0030 |
| max | 0.1762 | 0.0720 | 0.1191 | 0.5025 |
| l1_to_clipnorm | 0.0801 | 0.0518 | 0.0749 | 0.1306 |

argmax legal: 0.9984

| gates | raw | clipnorm | Bayes / CE-model reference |
|---|---|---|---|
| legal mass | 0.9888 | 0.9509 | 1.0 / 0.93 |
| top-1 legal | 0.9984 | 0.9984 | 1.0 / 0.987 |
| CE | 2.0482 | 2.0874 | 2.0107 (Bayes) |

| editability (canonical best arms) | raw | clipnorm |
|---|---|---|
| unedited | -0.817 | -0.773 |
| PI pt4·α3 | +0.684 / 0.23 | +0.633 / 0.25 |
| ND pt4·α0.1 | +0.739 / 0.18 | +0.689 / 0.19 |
| GS pt4·α0.05 | +0.730 / 0.20 | +0.663 / 0.22 |
