import json
from flask import Flask,jsonify,request
import pandas as pd



app = Flask(__name__)

from joblib import dump, load


model = load('../model/logit_model.joblib')
columns = ['age', 'sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

@app.route('/predict', methods = ["GET","POST"])
def index():
	if(request.method == "POST"):
		data = request.json
		request_data = {
			'age': [data['age']],
			'sex': [data['sex']],
			'cp': [data['cp']],
            'trestbps': [data['trestbps']],
			'chol': [data['chol']],
			'fbs': [data['fbs']],
			'restecg': [data['restecg']],
			'thalach': [data['thalach']],
			'exang': [data['exang']],
			'oldpeak': [data['oldpeak']],
			'slope': [data['slope']],
			'ca': [data['ca']],
			'thal': [data['thal']]
        }
		df = pd.DataFrame(request_data, columns=columns)
		res = model.predict(df)
		data = {
			"target": res.tolist()[0]
        }
		return jsonify(data)
	else:
		data = {
			"method": "GET"
        }
		return jsonify(data)



if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)