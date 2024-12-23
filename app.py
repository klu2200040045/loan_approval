from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open('models.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

# Create the Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get the input data from the form
            gender = int(request.form.get('Gender'))  # Gender (1 for Male, 0 for Female)
            married = int(request.form.get('Married'))  # Married (1 for Yes, 0 for No)
            dependents = int(request.form.get('Dependents'))  # Dependents (0, 1, 2, 3, or more)
            education = int(request.form.get('Education'))  # Education (1 for Graduate, 0 for Not Graduate)
            self_employed = int(request.form.get('Self_Employed'))  # Self_Employed (1 for Yes, 0 for No)
            applicant_income = float(request.form.get('ApplicantIncome'))  # Applicant Income
            coapplicant_income = float(request.form.get('CoapplicantIncome'))  # Coapplicant Income
            loan_amount = float(request.form.get('LoanAmount'))  # Loan Amount
            loan_amount_term = int(request.form.get('Loan_Amount_Term'))  # Loan Amount Term (in months)
            credit_history = float(request.form.get('Credit_History'))  # Credit History (1.0 or 0.0)
            property_area = int(request.form.get('Property_Area'))  # Property Area (0, 1, 2 for Urban, Semiurban, Rural)

            # Check for dependents value
            if dependents < 0:
                return render_template("index.html", error_message="Minimum dependents is 0")

            # Prepare the input data for the model
            input_data = [
                gender, married, dependents, education, self_employed,
                applicant_income, coapplicant_income, loan_amount,
                loan_amount_term, credit_history, property_area
            ]

            # Convert input data into a numpy array for prediction
            input_data = np.array(input_data).reshape(1, -1)

            # Make a prediction using the loaded model
            prediction = best_model.predict(input_data)[0]

            # Update the prediction result to display a readable message
            if prediction == 0:
                prediction_message = "Loan Not Approved"
            else:
                prediction_message = "Loan Approved"

            # Display the prediction result on the web page
            return render_template("index.html", prediction_message=prediction_message)

        except ValueError:
            return render_template("index.html", error_message="Please enter valid input values")

    return render_template("index.html", prediction_message=None, error_message=None)

if __name__ == "__main__":
    app.run(debug=True)
