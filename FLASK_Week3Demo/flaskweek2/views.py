import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("/Users/Olga/Documents/INSIGHT2019/FLASK_Week2Demo/flaskweek2/FlaskCSV_2.csv")
df2 = pd.read_csv("/Users/Olga/Documents/INSIGHT2019/FLASK_Week3Demo/FilesSavedFromPython/FeaturesDFTrainANDTest.csv")
currentDF = pd.read_csv("/Users/Olga/Documents/INSIGHT2019/FLASK_Week3Demo/FilesSavedFromPython/FeaturesDFTrainANDTest_2016ONLY_wLIBNAME.csv")

rfmodel_pkl = open("/Users/Olga/Documents/INSIGHT2019/FLASK_Week3Demo/FilesSavedFromPython/ModelPickle1.pkl",'rb')
rfmodel = pickle.load(rfmodel_pkl)


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

@app.route('/try2')
def try2():
    string = str(rfmodel)
    return string

@app.route('/model1')
def model1():
    Libraries = list(df['LIBNAME'])
    return render_template("model_1.html", model = rfmodel, libs = Libraries)

@app.route('/tutorial1')
def tutorial1():
    return render_template('tutorial1.html')

@app.route('/tutorial2')
def tutorial2():
    return render_template('about1.html')

@app.route('/model2')
def model2():
    #Values in list
    formatted_list = list()
    list_of_feats_custom = df2.iloc[0, :].tolist()
    for item in list_of_feats_custom:
        formatted_list.append("%.2f"%item)
    
    #Dictionary
    keys = df2.columns.tolist()[1:]
    values = formatted_list[1:]
    dict_of_feats_custom = dict(zip(keys, values))
    return render_template('model_2.html', list_of_feats = formatted_list[1:], dict_of_feats = dict_of_feats_custom)

@app.route('/model3', methods=['GET'])
def model3():
    #Values in list
    formatted_list = list()
    list_of_feats_custom = df2.iloc[0, :].tolist()
    for item in list_of_feats_custom:
        formatted_list.append("%.2f"%item)
    
    #Dictionary
    keys = df2.columns.tolist()[1:]
    values = formatted_list[1:]
    dict_of_feats_custom = dict(zip(keys, values))
    return render_template('model_3.html', list_of_feats = formatted_list[1:], dict_of_feats = dict_of_feats_custom)

@app.route('/model3', methods=['POST'])
def model3_submit():
    #var = request.form["namehere"]
    #print(var)
    return request.form["namehere"]
#return render_template('model_3_submit.html', list_of_feats = list(), dict_of_feats = dict())

@app.route('/model4', methods=['GET'])
def model4():
    #Values in list
    formatted_list = list()
    list_of_feats_custom = df2.iloc[0, :].tolist()
    for item in list_of_feats_custom:
        formatted_list.append("%.2f"%item)
    
    #Dictionary
    keys = df2.columns.tolist()[1:]
    values = formatted_list[1:]
    dict_of_feats_custom = dict(zip(keys, values))
    return render_template('model_4.html', list_of_feats = formatted_list[1:], dict_of_feats = dict_of_feats_custom)

@app.route('/model4submit', methods=['POST'])
def model4_submit():
    return request.form["namehere"]

###START HERE FOR WEEK 3 DEMO
@app.route('/home1')
def home1():
    return render_template('index_bootstrap2.html')

@app.route('/library', methods=['GET'])
def library():
    Libraries = list(currentDF['LIBNAME'])
    return render_template('dropdown_lib.html', libs = Libraries)

@app.route('/model5', methods=['GET', 'POST'])
def model5():
    myvariable = request.form.get("Name")
    idx = currentDF[currentDF.LIBNAME == str(myvariable)].index.tolist()
    
    formatted_list2 = list()
    list_of_feats_trial = list(currentDF.loc[idx[0]][1:])
    for item in list_of_feats_trial:
        formatted_list2.append("%.4f"%item)
    
    return render_template('model_5.html', list_of_feats = formatted_list2)

@app.route('/model5submit', methods=['POST'])
def model5_submit():
    var1 = request.form.get("bookvol")
    var2 = request.form.get("locgovfund")
    var3 = request.form.get("othinc")
    var4 = request.form.get("kidprog")
    var5 = request.form.get("othopexp")
    var6 = request.form.get("ref")
    var7 = request.form.get("othmatexp")
    var8 = request.form.get("othpaidstaff")
    var9 = request.form.get("loanfr")
    var10 = request.form.get("libs")
    var11 = request.form.get("master")
    var12 = request.form.get("hrs")
    var13 = request.form.get("cent")
    var14 = request.form.get("benefits")
    var15 = request.form.get("loans")
    var16 = request.form.get("totlibprogs")
    var17 = request.form.get("video")
    var18 = request.form.get("stategovfund")
    
    list_of_vars = [float(var1), float(var2), float(var3), float(var4), float(var5), float(var6), float(var7), float(var8), float(var9), float(var10), float(var11), float(var12), float(var13), float(var14), float(var15), float(var16), float(var17), float(var18)]
    
    df3 = pd.DataFrame(index = [0, 1, 2], columns = df2.columns[1:].tolist())
    for i in range(18):
        df3.iloc[0, i] = list_of_vars[i]
    
    val = rfmodel.predict(df3.iloc[0:1, :])[0]
    return "Predicted Usage: " + str(val)
