import numpy as np
import pandas as pd
import statsmodels.api as sm

data = {
    'Jam Belajar': [2, 4, 3, 6, 5, 7, 8, 4, 9, 6],
    'Jam Tidur':   [7, 6, 8, 5, 7, 6, 5, 8, 6, 7],
    'IPK':         [2.8, 3.0, 3.1, 3.4, 3.3, 3.6, 3.7, 3.2, 3.9, 3.5]
}
df = pd.DataFrame(data)

X = df[['Jam Belajar', 'Jam Tidur']]
y = df['IPK']

X = sm.add_constant(X)   # tambahkan konstanta (intercept)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
