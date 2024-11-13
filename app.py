from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

app=Flask(__name__)

__model=None
__location=None
__data_columns=None

df=pd.read_csv('notebook/data/bengaluru_house_cleaned_csv')
__location=df.columns[4:]
__data_columns=df.columns.drop('price')

def load_save_artifacts():
   with open('notebook/banglore_home_prices_model.pickle','rb') as file:
       global __model
       if __model is None:
        __model=pickle.load(file)
        print("model",__model)

def get_estimate_price(bath,bhk,square_feet,location):
   # global __data_columns
   try:
      print("location",location)
      print("__data_columns[location]",__data_columns.get_loc(location))
      loc_index=__data_columns.get_loc(location)
      print("loc_index",loc_index)
   except:
      loc_index=-1
   x=np.zeros(len(__data_columns))
   x[0]=bath
   x[1]=bhk
   x[2]=square_feet

   if loc_index>=0:
      x[loc_index]=1

   print("x[loc_index]",loc_index,x[loc_index],x,len(x))
   return round(__model.predict([x])[0],2)




@app.route('/')
def index():
   return render_template('index.html',locations=__location)

@app.route('/predict_home_price',methods=['POST'])
def predict_home_price():
   total_sqft=request.form['square_feet']
   location=request.form['location']
   bhk=request.form['bhk']
   bath=request.form['bath']

   response=jsonify({
      'estimated_price':get_estimate_price(bath,bhk,total_sqft,location)
   })

   response.headers.add('Access-Control-Allow-Origin','*')
   print("response",response)
   return response


if __name__=="__main__":
   load_save_artifacts()
   print("loading prediction...")
   app.run(debug=True,port=5000)

