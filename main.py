import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
df1 = pd.read_csv("cleaned_data.csv")
pipe = pickle.load(open('RandomForestModel.pkl','rb'))

@app.route('/')
def index():
    location = sorted(df1['Locality'].unique())
    return render_template('index.html', location = location)


@app.route('/predict', methods=['POST'])
def predict():
    Locality = request.form.get('Locality')
    BHK = request.form.get('BHK')
    Bathroom = float(request.form.get('Bathroom'))
    Per_Sqft = float(request.form.get('Per_Sqft'))
    Area = float(request.form.get('Area'))
    prediction = pipe.predict(pd.DataFrame(data=[(Area,BHK,Bathroom,Locality,Per_Sqft)],columns=['Area','BHK','Bathroom','Locality','Per_Sqft']))[0]
    return str(prediction)
if __name__ == '__main__':
    app.run(debug=True)
