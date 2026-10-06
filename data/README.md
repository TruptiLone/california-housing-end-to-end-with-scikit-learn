# Dataset provenance

`housing.csv` is the unmodified `housing/housing.csv` member of the source archive:
https://github.com/ageron/data/raw/main/housing.tgz

Downloaded for this project on October 6, 2026 (UTC). It contains 20,640 rows and 10 columns, including the target `median_house_value`. File bytes were copied directly from the tar archive; no rows, values or columns were changed.

SHA-256: `2364609dc48bec7df3ba9dbb7041478e704ecddcee70ef1827ec3fc49d22c0cc`

See the [upstream dataset repository](https://github.com/ageron/data) and [chapter 2 notebook](https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb) for context. The data is attributed to its original sources; this project does not claim ownership or assign it a new license. This is historical district-level data, not current home listings.

Inspect locally with:

```python
import pandas as pd
housing = pd.read_csv("data/housing.csv")
print(housing.head())
print(housing.shape)
print(housing.isna().sum())
```
