from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas as pd
import numpy as np

app = Flask(__name__)

# Loading and Prepare data
W = np.array([[0], [1], [1], [1], [0], [1], [1], [0], [0], [1],
              [1], [0], [0], [1], [0], [1], [0], [0], [1], [1]])
X = np.array([[19.8], [23.4], [27.7], [24.6], [21.5], [25.1], [22.4], [29.3], [20.8], [20.2],
              [27.3], [24.5], [22.9], [18.4], [24.2], [21.0], [25.9], [23.2], [21.6], [22.8]])
Y = np.array([137, 118, 124, 124, 120, 129, 122, 142, 128, 114,
              132, 130, 130, 112, 132, 117, 134, 132, 121, 128])

features = np.hstack([W, X])
model = LinearRegression().fit(features, Y)

# Question 1 - Average Treatment Effect (ATE)
# Convert to DataFrame
data = pd.DataFrame({
    "Y": Y,
    "W": W.flatten(),
    "X": X.flatten()
})

# Adding intercept for OLS model
X_reg = sm.add_constant(data[["W", "X"]])
ols_model = sm.ols(data["Y"], X_reg).fit()

# Log model summary to file
with open("model_summary.txt", "w") as f:
  f.write(ols_model.summary().as_text())

# Print key ATE info
print("===== ATE Estimation =====")
print("Intercept (α):", round(ols_model.params["const"], 2))
print("Treatment Effect (τ̂):", round(ols_model.params["W"], 2))
print("Spending Effect (β):", round(ols_model.params["X"], 2))
print("P-value for τ̂:", round(ols_model.pvalues["W"], 4))
print("==========================")

@app.route("/predict")
def predict():
  try:
    w_val = float(request.args.get("w", 0))
    x_val = float(request.args.get("x", 0))
  except ValueError:
      return jsonify({"error": "Invalid input for w or x"}), 400
    
  y_pred = model.predict([[w_val, x_val]])[0]
  
  #  Log prediction
  with open("output.txt", "w") as f:
    f.write(f"Input w: {w_val}, Input x: {x_val} \nPrediction: {y_pred:.2f}\n")
    
  return jsonify({
    "w": w_val, 
    "x": x_val, 
    "predicted_engagement_score": round(y_pred, 2)
  })
  
if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
