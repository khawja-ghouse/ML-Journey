import numpy as np
import pandas as pd

# Employee data
names = np.array(["Asha", "Rahul", "Priya", "Kiran", "Sneha"])

salaries = np.array([50000, 65000, 80000, 45000, 90000])

experience = np.array([2, 4, 7, 1, 8])

# print("Names:", names)
# print("Salaries:", salaries)
# print("Experience:", experience)

# print(np.mean(salaries))

# print(np.max(salaries))

# print(names[0])

# print(names[:3])

df = pd.DataFrame({
    "Names" : names,
    "Salaries" : salaries,
    "Experiance" : experience
})

print(df)

# 1. Display the first 3 rows
print(df.head(3))

# 2. Get only the Salary column
print(df["Salaries"])

print(df[df["Salaries"] > 60000])

print(df.sort_values("Salaries", ascending=False))

df["Bonus"] = df["Salaries"] * 0.10

print(df["Salaries"].idxmin())

print(df.loc[df["Salaries"].idxmin()])
