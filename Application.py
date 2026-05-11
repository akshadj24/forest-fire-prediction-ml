from flask import Flask,render_template,request,url_for,redirect
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle

app=Flask(__name__)

with open("Ridge.pkl",'rb') as file:
    model = pickle.load(file)       # Your trained model

with open("scalar.pkl",'rb') as file:      
    scaler = pickle.load(file)     # Your fitted StandardScaler

feature_names = ["Temperature", "RH", "WS", "FFMS", "DMC", "DC", "ISI", "BUI"]


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/predict",methods=['GET','POST'])
def predict():

    if request.method=='POST':

        try:
            Temperature=float(request.form.get('Temperature', 0))
            rs=float(request.form.get('RH', 0))
            ws=float(request.form.get('WS', 0))
            ffmc=float(request.form.get('FFMC', 0))
            dmc=float(request.form.get('DMC', 0))
            dc=float(request.form.get('DC', 0))
            isi=float(request.form.get('ISI', 0))
            bui=float(request.form.get('BUI', 0))
            
            # Additional features required by the model
            region=float(request.form.get('Region', 0))
            rain=float(request.form.get('Rain', 0.0))
            classes=request.form.get('Classes', 'fire')
            
            classes_fire = 1 if classes == 'fire' else 0
            classes_not_fire = 1 if classes == 'not_fire' else 0

            feature_names = ['Temperature', ' RH', ' Ws', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'Region', 'Rain_new', 'Classes_fire', 'Classes_not fire']
            
            input_data=scaler.transform(pd.DataFrame([[Temperature,rs,ws,ffmc,dmc,dc,isi,bui,region,rain,classes_fire,classes_not_fire]],columns=feature_names))

            output=model.predict(input_data)
            fwi_val = output[0]
            
            if fwi_val <= 5.2:
                risk_level = "Low Risk of Fire Intensity"
            elif fwi_val <= 11.2:
                risk_level = "Moderate Risk of Fire Intensity"
            elif fwi_val <= 21.3:
                risk_level = "High Risk of Fire Intensity"
            elif fwi_val <= 38.0:
                risk_level = "Very High Risk of Fire Intensity"
            else:
                risk_level = "Extreme Risk of Fire Intensity"

            return render_template("predict.html", results=output, risk_level=risk_level)
           

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template("predict.html",message="Unable to fetch the record you entered. Please try again.")
        
   

    

    return render_template("predict.html")







if __name__=="__main__":

    app.run(debug=True)
