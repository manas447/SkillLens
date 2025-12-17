import pandas as pd
import numpy as np

np.random.seed(42)
n = 800

data = {
    "cgpa": np.round(np.random.uniform(5.5, 9.5, n), 2),
    "projects": np.random.randint(0, 6, n),
    "internships": np.random.randint(0, 4, n),
    "certifications": np.random.randint(0, 6, n),
    "python": np.random.randint(0, 2, n),
    "ml": np.random.randint(0, 2, n),
    "sql": np.random.randint(0, 2, n),
    "communication": np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)

# Base score (imperfect)
score = (
    df["cgpa"] * 7
    + df["projects"] * 3
    + df["internships"] * 4
    + df["certifications"] * 2
    + df["python"] * 4
    + df["ml"] * 5
    + df["sql"] * 3
    + df["communication"] * 3
)

# ADD NOISE (THIS IS THE KEY)
noise = np.random.normal(0, 10, n)
score = score + noise

# Convert to binary target
df["employability"] = (score > 60).astype(int)

df.to_csv("student_data.csv", index=False)
print("Realistic dataset regenerated")
