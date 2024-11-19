from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
   
    else:
        data=CustomData(
            ApplicantIncome=request.form.get("ApplicantIncome"),
            CoapplicantIncome=request.form.get("CoapplicantIncome"),
            LoanAmount=request.form.get("LoanAmount"),
            Loan_Amount_Term=request.form.get("Loan_Amount_Term"),
            Credit_History=request.form.get("Credit_History"),
            Gender=request.form.get("Gender"),
            Married=request.form.get("Married"),
            Dependents=request.form.get("Dependents"),
            Education=request.form.get("Education"),
            Self_Employed=request.form.get("Self_Employed"), 
            Property_Area=request.form.get("Property_Area"),
            )
        # Convert the data into a dataframe for prediction
    pred_df = data.get_data_dataframe()
    print("Received the data for prediction")

    # Create a prediction pipeline and get the prediction
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    # Ensure that 'results' is iterable (a list or array) before using the list comprehension
    prediction = [1 if pred == 'Y' else 0 for pred in results]  # Map Y to 1, N to 0
    
    print(f"Prediction: {prediction}")  # Print the final prediction for debugging

    # Return the prediction result to the home page (this assumes prediction is a list)
    return render_template("home.html", results=prediction[0])  # Only passing the first result to the page

'''
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                          'Loan_Amount_Term', 'Credit_History']
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                            'Self_Employed', 'Property_Area']
''' 
if __name__=="__main__":
    app.run("0.0.0.0",debug=True)