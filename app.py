from flask import Flask, render_template, request, redirect
from utils import naics_codes
import subprocess
from models import *
from utils import *
from analysis import generate_state_predictions
import pickle
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
	return render_template("home.html")

import json

# Load the model from the file
with open('models/model_combined.pkl', 'rb') as f:
    model, loss_data= pickle.load(f)

@app.route('/model_page', methods=['GET', 'POST'])
def model_page():
    predictions = None
    if request.method == 'POST':
        unemployment_rate = float(request.form['unemployment_rate'])
        industry_code = int(request.form['industry_code'])

        predictions_df = generate_state_predictions(model, unemployment_rate, industry_code)
        predictions_df['Prediction'] = predictions_df['Prediction'].apply(lambda x: np.round(x, 2))
        predictions = predictions_df.to_dict(orient='records')

    return render_template("model_page.html", predictions=predictions, industries=naics_codes)


@app.route('/about', methods=['GET'])
def about():
	return render_template("about.html")

@app.route('/figures', methods=['GET'])
def figures():
	return render_template("figures.html")


if __name__ == '__main__':
	app.run(debug=True)
