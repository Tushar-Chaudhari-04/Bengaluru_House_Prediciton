from flask import Flask,render_template,request,jsonify,json
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

app=Flask(__name__)

__model=None
__location=None
__data_columns=None

# df=pd.read_csv('notebook/data/bengaluru_house_cleaned_csv')

def load_save_artifacts():
   with open('banglore_home_prices_model.pickle','rb') as file:
       global __model
       __model=pickle.load(file)

def convertToCrore(price):
   if(price>99):
      crore_value=round(price/100,2)
      return f"{crore_value} Crore"
   else:
      return f"{price} Lakh"


def get_estimate_price(bath,bhk,square_feet,location):
   
   try:
      loc_index=__data_columns['data'].index(location)
   except:
      loc_index=-1
   x=np.zeros(len(__data_columns['data']))
   x[0]=bath
   x[1]=bhk
   x[2]=square_feet

   if loc_index>=0:
      x[loc_index]=1

   price_pred=round(__model.predict([x])[0],2)

   string_price=convertToCrore(price_pred).replace('"','').strip('"')
   print("string_price",string_price)
   return string_price




@app.route('/')
def index():
   with open('columns.json','rb') as f1:
      global __location
      global __data_columns
      __data_columns=json.load(f1)

      if __data_columns and 'data' in __data_columns:
         __location = __data_columns['data'][4:]  # Assign __location to be a slice of the 
      else:
         print("Error: __data_columns is empty or 'data' key is missing.")
      # __data_columns=__location['data']

   with open('banglore_home_prices_model.pickle','rb') as file:
       global __model
       __model=pickle.load(file)

   return render_template('index.html',locations=__location)

@app.route('/predict_home_price',methods=['POST'])
def predict_home_price():
   total_sqft=request.form['square_feet']
   location=request.form['location']
   bhk=request.form['bhk']
   bath=request.form['bath']

   response=jsonify(
      get_estimate_price(bath,bhk,total_sqft,location)
   )

   response.headers.add('Access-Control-Allow-Origin','*')
   return response

if __name__ == '__main__':
   #  load_save_artifacts()
    print("loading prediction...")
    app.run(debug=True)