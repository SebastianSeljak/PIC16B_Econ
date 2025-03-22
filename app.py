from flask import Flask, render_template, request, redirect
import subprocess
from models import *
from utils import industry_dict_abbrev, naics_codes, industry_dict_abbrev_rev
from analysis import generate_state_predictions, plot_survival_rates
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
    print(naics_codes)
    industry_names = list(industry_dict_abbrev_rev.keys())
    print(industry_names)
    if request.method == 'POST':
        unemployment_rate = float(request.form['unemployment_rate'])
        industry_code = int(industry_dict_abbrev_rev[request.form['industry_code']])

        predictions_df = generate_state_predictions(model, unemployment_rate, industry_code)[0]
        predictions_df['Prediction'] = predictions_df['Prediction'].apply(lambda x: np.round(x, 2))
        predictions = predictions_df.to_dict(orient='records')
		
        fig = plot_survival_rates(*generate_state_predictions(model, unemployment_rate, industry_code))
        fig.write_image('static/survival_rates.png')

    return render_template("model_page.html", predictions=predictions, industries=industry_names)


@app.route('/about', methods=['GET'])
def about():
	return render_template("about.html")

@app.route('/figures', methods=['GET'])
def figures():
	return render_template("figures.html")


if __name__ == '__main__':
	app.run(debug=True)
