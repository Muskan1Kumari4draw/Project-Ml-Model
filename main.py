from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import Lasso,Ridge
from flask_cors import CORS
import numpy as np
import flask
import flask_cors
import sklearn
import pandas
import nltk

print("Flask:", flask.__version__)
print("Flask-Cors:", flask_cors.__version__)
print("scikit-learn:", sklearn.__version__)
print("Pandas:", pandas.__version__)
print("NLTK:", nltk.__version__)
app = Flask(__name__)
CORS(app)

sid = SentimentIntensityAnalyzer()


with open('irsi_data.pkl', 'rb') as file:
    model_flower = pickle.load(file)
# with open('salary_model_DTC.pkl', 'rb') as model_file:
    # model_salary = pickle.load(model_file)
with open('scaler.pkl', 'rb') as file:
    scaler_flower = pickle.load(file)
# with open('scalar_salry_DTC.pkl', 'rb') as file:
#     scaler_salary = pickle.load(file)
# Load the pre-trained models
with open('salary_model_DTC.pkl', 'rb') as file:
    dt_model = pickle.load(file)

with open('lasso_model.pkl', 'rb') as file:
    lasso_model = pickle.load(file)

with open('ridge_model.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

# Route for prediction of Iris Flower
@app.route('/flowers', methods=['POST'])
def predict():
    data = request.get_json()
    
  
    inp = pd.DataFrame(data, index=[0])
    
   
    scaled_dt = scaler_flower.transform(inp)
    
   
    prediction = model_flower.predict(scaled_dt)
    
    return jsonify({'species': prediction.tolist()})

# Route for sentiment analysis
@app.route('/sentiments', methods=['POST'])
def sentiment_analysis():
    data = request.get_json()
    
   
    review_text = data.get("review")
    if not review_text:
        return jsonify({"error": "No review text provided"}), 400
    
    
    sentiment_scores = sid.polarity_scores(review_text)
    
    return jsonify(sentiment_scores)

# Route for Salary Prediction
@app.route('/salary', methods=['POST'])
def salary_prediction():
     
    data = request.get_json()

   
    if 'YearsExperience' not in data or 'Age' not in data:
        return jsonify({"error": "Missing required fields: 'YearsExperience' and 'Age'"}), 400

   
    years_experience = data['YearsExperience']
    age = data['Age']

    
    input_data = np.array([[years_experience, age]])

   
    # dt_prediction = dt_model.predict(input_data)[0]
    lasso_prediction = lasso_model.predict(input_data)[0]
    # ridge_prediction = ridge_model.predict(input_data)[0]

    # Return the predictions as a JSON response
    # response = {
    #     "DecisionTreePrediction": dt_prediction,
    #     "LassoPrediction": lasso_prediction,
    #     "RidgePrediction": ridge_prediction
    # }

    return jsonify({'Salary': lasso_prediction.tolist()})

    # return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
