import numpy as np
import pickle
import joblib
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import shap
shap.initjs()
from flask import Flask, request, render_template, make_response, send_file
import base64
import io
import pandas as pd
from lifelines import CoxPHFitter
import logging  # Import the logging module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
survmodel = pickle.load(open('survivemodel.pkl', 'rb'))
explainer = joblib.load(filename="explainer.bz2")  # Load explainer once

# Define the feature columns used by your model
FEATURE_COLUMNS = [
    'Gender',
    'Senior Citizen',
    'Partner',
    'Dependents',
    'Tenure Months',
    'Phone Service',
    'Multiple Lines',
    'Internet Service',
    'Online Security',
    'Online Backup',
    'Device Protection',
    'Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Contract',
    'Paperless Billing',
    'Payment Method',
    'Monthly Charges',
    'Total Charges'
]

# Define all expected feature columns after encoding
EXPECTED_FEATURES = [
    'Gender_Male',
    'Partner_Yes',
    'Dependents_Yes',
    'Phone Service_Yes',
    'Multiple Lines_No',
    'Internet Service_Fiber optic',
    'Internet Service_None',
    'Online Security_Yes',
    'Online Backup_Yes',
    'Device Protection_Yes',
    'Tech Support_Yes',
    'Streaming TV_Yes',
    'Streaming Movies_Yes',
    'Contract_Oneyear',
    'Paperless Billing_Yes',
    'Payment Method_Credit Card (automatic)',
    'Payment Method_Electronic check',
    'Monthly Charges'
    # Add the remaining required features to make a total of 23
]

# Define raw feature columns from CSV - exact column names
FEATURE_COLUMNS = [
    'Gender',
    'Senior Citizen',
    'Partner', 
    'Dependents',
    'Tenure Months',
    'Phone Service',
    'Multiple Lines',
    'Internet Service',
    'Online Security',
    'Online Backup',
    'Device Protection',
    'Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Contract',
    'Paperless Billing',
    'Payment Method',
    'Monthly Charges',
    'Total Charges'
]

# Define categorical columns for one-hot encoding
CATEGORICAL_COLUMNS = [
    'Gender',
    'Partner',
    'Dependents', 
    'Phone Service',
    'Multiple Lines',
    'Internet Service',
    'Online Security',
    'Online Backup',
    'Device Protection',
    'Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Contract',
    'Paperless Billing',
    'Payment Method'
]

# Define expected encoded features (23 total)
EXPECTED_FEATURES = [
    'Gender_Male',
    'Senior Citizen',
    'Partner_Yes', 
    'Dependents_Yes',
    'Tenure Months',
    'Phone Service_Yes',
    'Multiple Lines_Yes',
    'Internet Service_DSL',
    'Internet Service_Fiber optic',
    'Internet Service_None',
    'Online Security_Yes',
    'Online Backup_Yes',
    'Device Protection_Yes',
    'Tech Support_Yes', 
    'Streaming TV_Yes',
    'Streaming Movies_Yes',
    'Contract_Month-to-month',
    'Contract_One year',
    'Contract_Two year',
    'Paperless Billing_Yes',
    'Payment Method_Electronic check',
    'Monthly Charges'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = 0
    if request.form["gender"] == "1":
        gender = 1

    SeniorCitizen = 0
    if 'SeniorCitizen' in request.form:
        SeniorCitizen = 1

    Partner = 0
    if 'Partner' in request.form:
        Partner = 1

    Dependents = 0
    if 'Dependents' in request.form:
        Dependents = 1

    PaperlessBilling = 0
    if 'PaperlessBilling' in request.form:
        PaperlessBilling = 1

    MonthlyCharges = float(request.form["MonthlyCharges"])
    Tenure = int(request.form["Tenure"])
    TotalCharges = MonthlyCharges * Tenure

    PhoneService = 0
    if 'PhoneService' in request.form:
        PhoneService = 1

    MultipleLines = 0
    if 'MultipleLines' in request.form and PhoneService == 1:
        MultipleLines = 1

    InternetService_Fiberoptic = 0
    InternetService_No = 0
    if request.form["InternetService"] == "0":
        InternetService_No = 1
    elif request.form["InternetService"] == "2":
        InternetService_Fiberoptic = 1

    OnlineSecurity = 0
    if 'OnlineSecurity' in request.form and InternetService_No == 0:
        OnlineSecurity = 1

    OnlineBackup = 0
    if 'OnlineBackup' in request.form and InternetService_No == 0:
        OnlineBackup = 1

    DeviceProtection = 0
    if 'DeviceProtection' in request.form and InternetService_No == 0:
        DeviceProtection = 1

    TechSupport = 0
    if 'TechSupport' in request.form and InternetService_No == 0:
        TechSupport = 1

    StreamingTV = 0
    if 'StreamingTV' in request.form and InternetService_No == 0:
        StreamingTV = 1

    StreamingMovies = 0
    if 'StreamingMovies' in request.form and InternetService_No == 0:
        StreamingMovies = 1

    Contract_Oneyear = 0
    Contract_Twoyear = 0
    if request.form["Contract"] == "1":
        Contract_Oneyear = 1
    elif request.form["Contract"] == "2":
        Contract_Twoyear = 1

    PaymentMethod_CreditCard = 0
    PaymentMethod_ElectronicCheck = 0
    PaymentMethod_MailedCheck = 0
    if request.form["PaymentMethod"] == "1":
        PaymentMethod_CreditCard = 1
    elif request.form["PaymentMethod"] == "2":
        PaymentMethod_ElectronicCheck = 1
    elif request.form["PaymentMethod"] == "3":
        PaymentMethod_MailedCheck = 1

    features = [gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
                DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
                InternetService_Fiberoptic, InternetService_No, Contract_Oneyear, Contract_Twoyear,
                PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck]

    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
               'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year', 'Contract_Two year',
               'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    final_features = [np.array(features)]
    prediction = model.predict_proba(final_features)
    output = prediction[0, 1]

    # SHAP Values
    explainer = joblib.load(filename="explainer.bz2")
    shap_values = explainer.shap_values(np.array(final_features))
    shap_img = io.BytesIO()
    shap.force_plot(explainer.expected_value[1], shap_values[1], columns, matplotlib=True, show=False).savefig(
        shap_img, bbox_inches="tight", format='png')
    shap_img.seek(0)
    shap_url = base64.b64encode(shap_img.getvalue()).decode()

    # Hazard and Survival Analysis
    surv_feats = np.array([gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
                           DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
                           InternetService_Fiberoptic, InternetService_No, Contract_Oneyear, Contract_Twoyear,
                           PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck])

    surv_feats = surv_feats.reshape(1, -1)

    hazard_img = io.BytesIO()
    fig, ax = plt.subplots()
    survmodel.predict_cumulative_hazard(surv_feats).plot(ax=ax, color='red')
    plt.axvline(x=Tenure, color='blue', linestyle='--')
    plt.legend(labels=['Hazard', 'Current Position'])
    ax.set_xlabel('Tenure', size=10)
    ax.set_ylabel('Cumulative Hazard', size=10)
    ax.set_title('Cumulative Hazard Over Time')
    plt.savefig(hazard_img, format='png')
    hazard_img.seek(0)
    hazard_url = base64.b64encode(hazard_img.getvalue()).decode()

    surv_img = io.BytesIO()
    fig, ax = plt.subplots()
    survmodel.predict_survival_function(surv_feats).plot(ax=ax, color='red')
    plt.axvline(x=Tenure, color='blue', linestyle='--')
    plt.legend(labels=['Survival Function', 'Current Position'])
    ax.set_xlabel('Tenure', size=10)
    ax.set_ylabel('Survival Probability', size=10)
    ax.set_title('Survival Probability Over Time')
    plt.savefig(surv_img, format='png')
    surv_img.seek(0)
    surv_url = base64.b64encode(surv_img.getvalue()).decode()

    life = survmodel.predict_survival_function(surv_feats).reset_index()
    life.columns = ['Tenure', 'Probability']
    max_life = life['Tenure'][life['Probability'] > 0.1].max()

    CLTV = max_life * MonthlyCharges

    # Gauge Plot
    def degree_range(n):
        start = np.linspace(0, 180, n + 1, endpoint=True)[0:-1]
        end = np.linspace(0, 180, n + 1, endpoint=True)[1::]
        mid_points = start + ((end - start) / 2.)
        return np.c_[start, end], mid_points

    def rot_text(ang):
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation

    def gauge(labels=['LOW', 'MEDIUM', 'HIGH', 'EXTREME'],
              colors=['#007A00', '#0063BF', '#FFCC00', '#ED1C24'], Probability=1, fname=False):
        N = len(labels)
        colors = colors[::-1]

        gauge_img = io.BytesIO()
        fig, ax = plt.subplots()

        ang_range, mid_points = degree_range(4)
        labels = labels[::-1]

        patches = []
        for ang, c in zip(ang_range, colors):
            patches.append(Wedge((0., 0.), .4, *ang, facecolor='w', lw=2))
            patches.append(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
        [ax.add_patch(p) for p in patches]

        for mid, lab in zip(mid_points, labels):
            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab,
                    horizontalalignment='center', verticalalignment='center', fontsize=14,
                    fontweight='bold', rotation=rot_text(mid))

        r = Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2)
        ax.add_patch(r)
        ax.text(0, -0.05, 'Churn Probability ' + np.round(Probability, 2).astype(str), horizontalalignment='center',
                verticalalignment='center', fontsize=22, fontweight='bold')

        pos = (1 - Probability) * 180
        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)),
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()

        plt.savefig(gauge_img, format='png')
        gauge_img.seek(0)
        url = base64.b64encode(gauge_img.getvalue()).decode()
        return url

    gauge_url = gauge(Probability=output)

    return render_template('index.html', prediction_text='Churn probability is {} and Expected Life Time Value is ${}'.format(
        round(output, 2), CLTV), url_2=gauge_url, url_4=shap_url, url_1=hazard_url, url_3=surv_url)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file part in the request.")
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="No file selected")
        if file and file.filename.endswith('.csv'):
            try:
                # Read CSV
                df = pd.read_csv(file)
                
                # Drop 'customerID' as it's not a feature
                if 'customerID' in df.columns:
                    df = df.drop(columns=['customerID'])
                
                # Convert 'tenure' and 'MonthlyCharges' to numeric, handle missing values
                df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
                df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
                
                # Calculate 'TotalCharges'
                df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
                
                # Define feature columns exactly as in CSV excluding 'customerID'
                feature_columns = [
                    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod',
                    'MonthlyCharges',
                    'TotalCharges'  # Include 'TotalCharges' in features
                ]

                # Create working copy
                X = df[feature_columns].copy()

                # One-hot encode categorical columns with drop_first=False
                categorical_columns = [
                    'gender', 'Partner', 'Dependents', 'PhoneService',
                    'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod'
                ]
                
                X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=False)

                # Define expected encoded features based on your model training
                expected_features = [
                    'gender_Female',
                    'gender_Male',
                    'SeniorCitizen',
                    'Partner_No',
                    'Partner_Yes',
                    'Dependents_No',
                    'Dependents_Yes',
                    'tenure',
                    'PhoneService_No',
                    'PhoneService_Yes',
                    'MultipleLines_No phone service',
                    'MultipleLines_No',
                    'MultipleLines_Yes',
                    'InternetService_DSL',
                    'InternetService_Fiber optic',
                    'InternetService_No',
                    'OnlineSecurity_No',
                    'OnlineSecurity_Yes',
                    'OnlineBackup_No',
                    'OnlineBackup_Yes',
                    'DeviceProtection_No',
                    'DeviceProtection_Yes',
                    'TechSupport_No',
                    'TechSupport_Yes',
                    'StreamingTV_No',
                    'StreamingTV_Yes',
                    'StreamingMovies_No',
                    'StreamingMovies_Yes',
                    'Contract_Month-to-month',
                    'Contract_One year',
                    'Contract_Two year',
                    'PaperlessBilling_No',
                    'PaperlessBilling_Yes',
                    'PaymentMethod_Bank transfer (automatic)',
                    'PaymentMethod_Credit card (automatic)',
                    'PaymentMethod_Electronic check',
                    'PaymentMethod_Mailed check',
                    'MonthlyCharges',
                    'TotalCharges'
                ]

                # Handle missing encoded columns by adding them with 0s
                for col in expected_features:
                    if col not in X_encoded.columns:
                        X_encoded[col] = 0

                # Reorder columns to match model expectations
                X_final = X_encoded[expected_features]

                # Select only the features expected by the model (23 in total)
                # Adjust the list below based on your actual model's required features
                # For demonstration, selecting the first 23 features
                X_final = X_final[expected_features[:23]]

                # Verify feature count
                if X_final.shape[1] != 23:
                    raise ValueError(f"Expected 23 features, got {X_final.shape[1]}")

                # Make predictions
                predictions = model.predict_proba(X_final)[:,1]

                # Add predictions to DataFrame
                df['Churn Probability'] = predictions

                # Create churn probability distribution plot
                plt.figure(figsize=(10, 6))
                plt.hist(predictions, bins=30, color='skyblue', edgecolor='black')
                plt.title('Distribution of Churn Probabilities')
                plt.xlabel('Churn Probability')
                plt.ylabel('Number of Customers')

                # Save plot to bytes
                churn_img = io.BytesIO()
                plt.savefig(churn_img, format='png', bbox_inches='tight', dpi=150)
                plt.close()
                churn_img.seek(0)
                churn_graph = base64.b64encode(churn_img.getvalue()).decode()

                # Split based on probability into five ranges
                df_0_20 = df[(df['Churn Probability'] > 0) & (df['Churn Probability'] <= 0.2)].copy()
                df_20_40 = df[(df['Churn Probability'] > 0.2) & (df['Churn Probability'] <= 0.4)].copy()
                df_40_60 = df[(df['Churn Probability'] > 0.4) & (df['Churn Probability'] <= 0.6)].copy()
                df_60_80 = df[(df['Churn Probability'] > 0.6) & (df['Churn Probability'] <= 0.8)].copy()
                df_80_100 = df[(df['Churn Probability'] > 0.8) & (df['Churn Probability'] <= 1.0)].copy()

                # Convert each subset to CSV
                csv_0_20 = df_0_20.to_csv(index=False)
                csv_20_40 = df_20_40.to_csv(index=False)
                csv_40_60 = df_40_60.to_csv(index=False)
                csv_60_80 = df_60_80.to_csv(index=False)
                csv_80_100 = df_80_100.to_csv(index=False)

                # Encode CSVs for download
                encoded_0_20 = base64.b64encode(csv_0_20.encode()).decode()
                encoded_20_40 = base64.b64encode(csv_20_40.encode()).decode()
                encoded_40_60 = base64.b64encode(csv_40_60.encode()).decode()
                encoded_60_80 = base64.b64encode(csv_60_80.encode()).decode()
                encoded_80_100 = base64.b64encode(csv_80_100.encode()).decode()

                return render_template('upload.html',
                                    download_0_20=True,
                                    download_20_40=True,
                                    download_40_60=True,
                                    download_60_80=True,
                                    download_80_100=True,
                                    encoded_0_20=encoded_0_20,
                                    encoded_20_40=encoded_20_40,
                                    encoded_40_60=encoded_40_60,
                                    encoded_60_80=encoded_60_80,
                                    encoded_80_100=encoded_80_100,
                                    filename_0_20='churn_1_20.csv',
                                    filename_20_40='churn_21_40.csv',
                                    filename_40_60='churn_41_60.csv',
                                    filename_60_80='churn_61_80.csv',
                                    filename_80_100='churn_81_100.csv',
                                    churn_graph=churn_graph)
                
            except Exception as e:
                logging.exception("Error processing file")
                return render_template('upload.html', error=str(e))
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)