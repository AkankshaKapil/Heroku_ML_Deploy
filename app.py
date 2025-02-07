import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging: Print form data to check if values are received
        print("Received form data:", request.form)  

        # Extract form data
        int_features = [float(x) for x in request.form.values()]  # Convert to float
        final_features = [np.array(int_features)]
        
        # Predict using the model
        prediction = model.predict(final_features)

        # Round prediction result
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Employee Salary should be $ {output}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
