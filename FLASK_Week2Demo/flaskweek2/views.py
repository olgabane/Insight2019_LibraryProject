import pandas as pd
df = pd.read_csv("/Users/Olga/Documents/INSIGHT2019/FLASK_Week2Demo/flaskweek2/FlaskCSV.csv")

from flask import render_template
from flaskweek2 import app

@app.route('/')
@app.route('/index')
def index():
	user = { 'nickname': 'Olga'}
	return render_template("index.html", title = "Home", user = user)

@app.route('/')
@app.route('/try1')
def try1():
	return str(df.shape)

@app.route('/')
@app.route('/Week2Demo')
def Week2Demo():
	Libraries = list(df['LIBNAME_x'])
	return render_template('dropdown.html', colours = Libraries)
	
