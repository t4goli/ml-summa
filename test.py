import pandas as pd

data = {
    "hours_studied": [1, 2, 3, 4, 5],
    "score": [50, 55, 65, 70, 80]
}

df = pd.DataFrame(data)
print(df)