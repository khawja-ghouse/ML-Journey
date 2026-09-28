import numpy as np
import pandas as pd

# Employee data
names = np.array(["Asha", "Rahul", "Priya", "Kiran", "Sneha"])

salaries = np.array([50000, 65000, 80000, 45000, 90000])

experience = np.array([2, 4, 7, 1, 8])


df = pd.DataFrame({
    "Names" : names,
    "Salaries" : salaries,
    "Experiance" : experience
})

print(df)

df["Bonus"] = np.where(
    df["Salaries"] >= 70000,
    df["Salaries"] * 0.20,
    df["Salaries"] *0.10
)

print(df)