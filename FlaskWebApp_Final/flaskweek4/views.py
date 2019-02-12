import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from flask import render_template
from flaskweek4 import app
from flask import request

currentDF = pd.read_csv("./Model/FinalDFforFLASK.csv", index_col=None)
LibnamestateDF = pd.read_csv("./Model/Libnamestate_only.csv", index_col=None)

rfmodel_pkl = open("./Model/ModelPickle_Feb4_12pm.pkl",'rb')
rfmodel = pickle.load(rfmodel_pkl)



###START HERE FOR WEEK 3 DEMO
@app.route('/')
@app.route('/state', methods=['GET'])
def state():
    States = list(LibnamestateDF['state'])
    States = list(set(States))
    States.sort()
    return render_template('dropdown_state.html', states = States)

@app.route('/about')
def about():
    return render_template('aboutbibliopal.html')

@app.route('/aboutolga')
def aboutolga():
    return render_template('aboutolga.html')


@app.route('/library', methods=['GET', 'POST'])
def library():
    myvariable = request.form.get("Name")
    
    newlist = list()
    for i in range(LibnamestateDF.shape[0]):
        if LibnamestateDF.loc[i, 'state'] == myvariable:
            newlist.append(LibnamestateDF.loc[i, 'LIBNAMESTATE'])
    newlist.sort()

    return render_template('dropdown_lib.html', libs = newlist)

@app.route('/model', methods=['GET', 'POST'])
def model():
    myvariable = request.form.get("Name")
    idx = currentDF[currentDF.LIBNAMESTATE == str(myvariable)].index.tolist()
    
    formatted_list2 = list()
    list_of_feats_trial = list(currentDF.loc[idx[0]])
    list_of_feats_trial_xpop = [x * float(list_of_feats_trial[23]) for x in list_of_feats_trial[0:23]]
    list_of_feats_trial_xpop[8] = list_of_feats_trial_xpop[8]/float(list_of_feats_trial[23])
    for item in list_of_feats_trial_xpop:
        formatted_list2.append("%.0f"%item)

    pop = float(list_of_feats_trial[23])

    return render_template('userinput_2.html', list_of_feats = formatted_list2, pop = pop)

@app.route('/modelsubmit', methods=['POST'])
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
    var11 = request.form.get("feat11")
    var12 = request.form.get("feat12")
    var13 = request.form.get("feat13")
    var14 = request.form.get("feat14")
    var15 = request.form.get("feat15")
    var16 = request.form.get("feat16")
    var17 = request.form.get("feat17")
    var18 = request.form.get("feat18")
    var19 = request.form.get("feat19")
    var20 = request.form.get("feat20")
    var21 = request.form.get("feat21")
    var22 = request.form.get("feat22")
    var23 = request.form.get("feat23")
    var24 = request.form.get("pop")
    
    list_of_vars = [float(var1), float(var2), float(var3), float(var4), float(var5), float(var6), float(var7), float(var8),float(var9), float(var10), float(var11), float(var12), float(var13), float(var14), float(var15), float(var16), float(var17),
        float(var18), float(var19), float(var20), float(var21),float(var22), float(var23)]
    list_of_vars_divpop = [x / float(var24) for x in list_of_vars]
    list_of_vars_divpop[8] = list_of_vars_divpop[8]*float(var24)
        
    DF_for_RF = pd.DataFrame(index = [0, 1, 2], columns = currentDF.columns[0:23].tolist())
    for i in range(23):
        DF_for_RF.iloc[0, i] = list_of_vars_divpop[i]
    
    val = rfmodel.predict(DF_for_RF.iloc[0:1, :])[0]
    val = round(val, 2)
    visits = val * float(var24);
    visits = int(round(visits, 0))
    #return "Predicted Usage: " + str(val)
    return render_template('prediction.html', PredVisit = visits, rfpred = val)
