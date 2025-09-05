from flask import Flask, render_template, request
import numpy as np
import pickle

with open("brest_cancer.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get values from form
            features = [float(x) for x in request.form.values()]
            final_features = np.array(features).reshape(1, -1)

            # Predict
            prediction = model.predict(final_features)[0]
            result = "Malignant" if prediction == 0 else "Benign"

            return render_template('index.html', prediction_text=f'Tumor is likely: {result}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
