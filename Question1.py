import statsmodels.api as sm
import pandas as pd
import numpy as np

# Loading and Prepare data
W = np.array([[0], [1], [1], [1], [0], [1], [1], [0], [0], [1],
              [1], [0], [0], [1], [0], [1], [0], [0], [1], [1]])
X = np.array([[19.8], [23.4], [27.7], [24.6], [21.5], [25.1], [22.4], [29.3], [20.8], [20.2],
              [27.3], [24.5], [22.9], [18.4], [24.2], [21.0], [25.9], [23.2], [21.6], [22.8]])
Y = np.array([137, 118, 124, 124, 120, 129, 122, 142, 128, 114,
              132, 130, 130, 112, 132, 117, 134, 132, 121, 128])

# Construct dataset
data = pd.DataFrame({
    "Y": Y,
    "W": W.flatten(),
    "X": X.flatten()
})

# Add constant (α)
X_reg = sm.add_constant(data[["W", "X"]])  # Includes α (intercept)
model_stats = sm.OLS(data["Y"], X_reg).fit()

# Extract coefficients
alpha = model_stats.params["const"]
tau = model_stats.params["W"]
beta = model_stats.params["X"]

print("Intercept (α):", round(alpha, 2))
print("Treatment Effect (τ̂):", round(tau, 2))
print("Spending Effect (β):", round(beta, 2))

print("\nStatistical Summary:")
print(model_stats.summary())
