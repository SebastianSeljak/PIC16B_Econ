from flask import Flask, render_template, request, redirect
from utils import naics_codes
import subprocess

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
	return render_template("home.html")

import json

@app.route('/model_page', methods=['GET', 'POST'])
def model_page():
    predictions = None
    if request.method == 'POST':
        unemployment_rate = request.form['unemployment_rate']
        industry_code = request.form['industry_code']
        result = subprocess.run(
            ["python", "analysis.py", unemployment_rate, industry_code], 
            capture_output=True, text=True
        )
        predictions = json.loads(result.stdout)
	try:
            predictions = json.loads(result.stdout)
        except json.JSONDecodeError:
            predictions = None
    return render_template("model_page.html", predictions=predictions, industries=naics_codes)


@app.route('/about', methods=['GET'])
def about():
	return render_template("about.html")

@app.route('/figures', methods=['GET'])
def figures():
	return render_template("figures.html")


if __name__ == '__main__':
	app.run(debug=True)
