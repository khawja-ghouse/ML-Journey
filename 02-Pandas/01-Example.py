import pandas as pd

data = {
    "Name": ["Asha", "Rahul", "John", "Sara"],
    "Age": [25, 30, 28, 35],
    "Salary": [30000, 50000, 45000, 70000]
}

df = pd.DataFrame(data)

print(df)

# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.head(2))
# print(df.describe())

print(df['Name'])
print(df[['Name','Age']])