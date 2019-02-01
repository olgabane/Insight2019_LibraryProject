import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from flask import render_template
from flaskweek2 import app
from flask import request

currentDF = pd.read_csv("./Model1pm/FEATURES_wLIBNAME_2016ONLY.csv", index_col=None)

rfmodel_pkl = open("./Model1pm/ModelPickle_Jan31_1pm.pkl",'rb')
rfmodel = pickle.load(rfmodel_pkl)


###START HERE FOR WEEK 3 DEMO
@app.route('/')
@app.route('/library', methods=['GET'])
def library():
    Libraries = list(currentDF['LIBNAME'])
    return render_template('dropdown_lib.html', libs = Libraries)

@app.route('/model5', methods=['GET', 'POST'])
def model5():
    myvariable = request.form.get("Name")
    idx = currentDF[currentDF.LIBNAME == str(myvariable)].index.tolist()
    
    formatted_list2 = list()
    list_of_feats_trial = list(currentDF.loc[idx[0]])
    list_of_feats_trial_xpop = [x * float(list_of_feats_trial[10]) for x in list_of_feats_trial[0:10]]
    for item in list_of_feats_trial_xpop:
        formatted_list2.append("%.0f"%item)
    
    pop = float(list_of_feats_trial[10])
    
    return render_template('userinput.html', list_of_feats = formatted_list2, pop = pop)

@app.route('/model5submit', methods=['POST'])
def model5_submit():
    var1 = request.form.get("feat1")
    var2 = request.form.get("feat2")
    var3 = request.form.get("feat3")
    var4 = request.form.get("feat4")
    var5 = request.form.get("feat5")
    var6 = request.form.get("feat6")
    var7 = request.form.get("feat7")
    var8 = request.form.get("feat8")
    var9 = request.form.get("feat9")
    var10 = request.form.get("feat10")
    var11 = request.form.get("pop")
    
    list_of_vars = [float(var1), float(var2), float(var3), float(var4), float(var5), float(var6), float(var7), float(var8),float(var9), float(var10)]
    list_of_vars_divpop = [x / float(var11) for x in list_of_vars]
        
    DF_for_RF = pd.DataFrame(index = [0, 1, 2], columns = currentDF.columns[0:10].tolist())
    for i in range(10):
        DF_for_RF.iloc[0, i] = list_of_vars_divpop[i]
    
    val = rfmodel.predict(DF_for_RF.iloc[0:1, :])[0]
    return "Predicted Usage: " + str(val)
