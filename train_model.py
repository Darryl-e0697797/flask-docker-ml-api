import statsmodels.api as sm
import pandas as pd
import numpy as np
import pickle

# Loading and Prepare data
def run_analysis():
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
  
  X_reg = sm.add_constant(data[["W", "X"]])  # Includes α (intercept)
  model_stats = sm.OLS(data["Y"], X_reg).fit()
  
  # Save model parameters to a dictionary
    params = {
        'intercept': model_stats.params['const'],
        'tau': model_stats.params['W'],
        'beta': model_stats.params['X'],
        'model_summary': str(model_stats.summary())
        }
    
    # Save to pickle file
    with open('model_params.pkl', 'wb') as f:
        pickle.dump(params, f)
    
    return params

if __name__ == '__main__':
    params = run_analysis()
    print("Model parameters saved to model_params.pkl")
    print(f"Intercept (α): {params['intercept']:.2f}")
    print(f"ATE (τ): {params['tau']:.2f}")
    print(f"Spending coef (β): {params['beta']:.2f}")
