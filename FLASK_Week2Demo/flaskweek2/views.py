import pandas as pd
df = pd.read_csv("/Users/Olga/Documents/INSIGHT2019/FLASK_Week2Demo/flaskweek2/FlaskCSV_2.csv")

from flask import render_template
from flaskweek2 import app
from flask import request

@app.route('/')
@app.route('/index')
def index():
	user = { 'nickname': 'Olga'}
	return render_template("index.html", title = "Home", user = user)

@app.route('/try1')
def try1():
	return str(df.shape)


@app.route('/Week2Demo')
def Week2Demo():
    Libraries = list(df['LIBNAME'])
    return render_template('dropdown.html', libs = Libraries)

@app.route('/submitted', methods=['POST'])
def submitted():
    myvariable = request.form.get("Name")
    strin = "Predicted usage:" + str(df.iloc[df[df.LIBNAME == myvariable].index[0], 2])
    return strin

