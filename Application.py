from flask import Flask,render_template,request,url_for,redirect
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import joblib 

app=Flask(__name__)

model = joblib.load("Ridge.pkl")       # Your trained model
scaler = joblib.load("scalar.pkl")     # Your fitted StandardScaler

feature_names = ["Temperature", "RH", "WS", "FFMS", "DMC", "DC", "ISI", "BUI"]


@app.route("/")
def  Landing():
    return render_template('landing.html')


@app.route("/predict_fire",methods=['GET','POST'])
def Mode_predict():

    if request.method=='POST':

        try:
            Temperature=request.form.get('Temperature')
            rs=request.form.get('RH')
            ws=request.form.get('WS')
            ffms=request.form.get('FFMS')
            dmc=request.form.get('DMC')
            dc=request.form.get('DC')
            isi=request.form.get('ISI')
            bui=request.form.get('BUI')

            

            input=scaler.transform(pd.DataFrame([[Temperature,rs,ws,ffms,dmc,dc,isi,bui]],columns=feature_names))

            output=model.predict(input)


            return render_template("Home.html",results=output)
           

        except  Exception as e:
            return render_template("Home.html",message="Unable to fetch the recode you enter pls enter again")
        
   

    

    return render_template("Home.html")







if __name__=="__main__":

    app.run(debug=True)

