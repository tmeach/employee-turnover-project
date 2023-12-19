import xgboost as xgb
import joblib
from flask import Flask 
from flask import request
from flask import jsonify


# load the model
model_file = 'xgb_model.model'
model = xgb.Booster({'nthread':4})
model.load_model(model_file)

# load the dv
dv_input_file = "dv.pkl"
dv = joblib.load(dv_input_file)

app = Flask('employee_turnover_prediciton')

@app.route('/predict', methods=['POST'])
def predict():
    employee = request.get_json()
    
    X_test = dv.transform(employee)
    features = list(dv.get_feature_names_out())
    dtest = xgb.DMatrix(X_test, feature_names=features)
   
    y_pred = model.predict(dtest)
    leave = y_pred >= 0.5
    
    result = {
        'probability_of_leaving': float(y_pred),
        'leaving_the_company': bool(leave)
    }
    
    return jsonify(result) 

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)