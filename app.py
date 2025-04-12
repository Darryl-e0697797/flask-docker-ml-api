from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model parameters
with open('model_params.pkl', 'rb') as f:
    params = pickle.load(f)

@app.route("/predict", methods =['GET'])
def predict():
    try:
        w_val = int(request.args.get("w", 0))
        x_val = float(request.args.get("x", 0))
        
        Y_pred = params['intercept'] + params['tau'] * w_val + params['beta'] * x_val

        return jsonify({
          'predicted_engagement_score': round(Y_pred, 2),
          'model_parameters': {
            'intercept': params['intercept'],
            'ATE (tau)': params['tau'],
            'spending_coef (beta)': params['beta']
          }
        })
    except ValueError:
        return jsonify({"error": "Invalid input for w or x"}), 400
      
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
