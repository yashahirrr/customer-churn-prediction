import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import roc_curve
from pathlib import Path
import shap

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    confusion_matrix
)



BASE_DIR = Path(__file__).resolve().parent.parent

logistic_model = load(BASE_DIR / "models" / "logistic_model.pkl")
rf_model = load(BASE_DIR / "models" / "rf_model.pkl")
xgb_model = load(BASE_DIR / "models" / "xgb_model.pkl")

DATA_PATH = BASE_DIR / "data" / "Telco_Customer_Churn.csv"
churn_data = pd.read_csv(DATA_PATH)
X_test = pd.read_csv(
    BASE_DIR / "data" / "X_test.csv"
)

y_test = pd.read_csv(
    BASE_DIR / "data" / "y_test.csv"
).squeeze()

tab1, tab2 = st.tabs(["Prediction", "Model Insights"])
preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

with tab1:

    st.title("Customer Churn Prediction App")

    st.write("Enter customer details to predict churn probability")


    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )


    gender = st.selectbox("Gender", ["Male","Female"])
    senior = st.selectbox("Senior Citizen", [0,1])
    partner = st.selectbox("Partner", ["Yes","No"])
    dependents = st.selectbox("Dependents", ["Yes","No"])

    tenure = st.slider("Tenure (months)",0,72)

    phoneservice = st.selectbox("Phone Service", ["Yes","No"])
    multiplelines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])

    internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

    onlinesecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
    onlinebackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
    deviceprotection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
    techsupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])

    streamingtv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
    streamingmovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes","No"])

    payment = st.selectbox(
        "Payment Method",
        ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
    )

    monthly = st.number_input("Monthly Charges",0.0,200.0)
    total = st.number_input("Total Charges",0.0,10000.0)


    data = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phoneservice],
    "MultipleLines":[multiplelines],
    "InternetService":[internet],
    "OnlineSecurity":[onlinesecurity],
    "OnlineBackup":[onlinebackup],
    "DeviceProtection":[deviceprotection],
    "TechSupport":[techsupport],
    "StreamingTV":[streamingtv],
    "StreamingMovies":[streamingmovies],
    "Contract":[contract],
    "PaperlessBilling":[paperless],
    "PaymentMethod":[payment],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total]
    })

    # X_transformed = preprocessor.transform(data)
    # feature_names = preprocessor.get_feature_names_out()
    # X_transformed_df = pd.DataFrame(
    #     X_transformed,
    #     columns=feature_names
    # )



    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    if st.button("Predict Churn"):

        prob = model.predict_proba(data)[0][1]

        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.progress(float(prob))

        if prob > 0.6:
            st.error("High Risk Customer")
        elif prob > 0.3:
            st.warning("Medium Risk Customer")
        else:
            st.success("Low Risk Customer")
        st.write(f"Model Used: **{model_choice}**")
        st.subheader("Prediction Explanation (SHAP)")
        selected_preprocessor = model.named_steps["preprocessor"]

        if model_choice == "Logistic Regression":
            selected_classifier = model.named_steps["classifier"]
        else:
            selected_classifier = model.named_steps["model"]

        X_transformed = selected_preprocessor.transform(data)

        feature_names = selected_preprocessor.get_feature_names_out()

        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

# Create model-specific SHAP explainer
        if model_choice == "Logistic Regression":

            # Use real customers from the Telco dataset as the SHAP background
            background_data = churn_data.drop(
                columns=["Churn", "customerID"],
                errors="ignore"
            ).sample(
                n=min(300, len(churn_data)),
                random_state=42
            )

            # Apply the same preprocessing used during model training
            X_background = selected_preprocessor.transform(background_data)

            X_background_df = pd.DataFrame(
                X_background,
                columns=feature_names
            )

            # Create SHAP explainer using real customer data as background
            explainer = shap.LinearExplainer(
                selected_classifier,
                X_background_df
            )

            # Explain the currently entered customer
            shap_values = explainer(X_transformed_df)

            explanation = shap.Explanation(
                values=shap_values.values[0],
                base_values=shap_values.base_values[0],
                data=X_transformed_df.iloc[0].values,
                feature_names=feature_names
            )

        elif model_choice in ["Random Forest", "XGBoost"]:

            explainer = shap.TreeExplainer(
                selected_classifier
            )

            shap_values = explainer(X_transformed_df)

            # SHAP 0.52 returns either:
            # (samples, features)
            # or (samples, features, classes)
            if len(shap_values.values.shape) == 3:

                explanation = shap.Explanation(
                    values=shap_values.values[0, :, 1],
                    base_values=shap_values.base_values[0, 1],
                    data=X_transformed_df.iloc[0].values,
                    feature_names=feature_names
                )

            else:

                # XGBoost in your environment returns this format:
                # (1, 46)
                explanation = shap_values[0]

        fig, ax = plt.subplots(figsize=(10, 6))

        shap.plots.waterfall(
            explanation,
            max_display=12,
            show=False
        )

        st.pyplot(fig)

        plt.close(fig)

        


# st.subheader("Prediction Explanation (SHAP)")

# X_transformed = preprocessor.transform(data)

# explainer = shap.Explainer(rf_classifier)
# shap_values = explainer(X_transformed_df)

# fig = plt.figure()

# shap.plots.waterfall(
#     shap_values[0, :, 1],
#     show=False
# )
# st.pyplot(fig)

with tab2:
    preprocessor = rf_model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    rf_classifier = rf_model.named_steps["model"]
    feature_importance = rf_classifier.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)

    top_features = feat_imp.head(15)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.invert_yaxis()
    ax.set_title("Top Drivers of Customer Churn")
    st.pyplot(fig)

    st.subheader("SHAP Summary Plot")

   # Use real customer records for SHAP analysis
    sample_data = churn_data.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    ).sample(
        n=min(300, len(churn_data)),
        random_state=42
    )

    X_sample_transformed = preprocessor.transform(sample_data)
    feature_names = preprocessor.get_feature_names_out()

    X_sample_df = pd.DataFrame(
        X_sample_transformed,
        columns=feature_names
    )
    explainer = shap.Explainer(rf_classifier)
    shap_values = explainer(X_sample_df)

    fig = plt.figure()
    shap.plots.beeswarm(
        shap_values[:, :, 1],
        max_display=15,
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("SHAP Feature Importance")

    fig = plt.figure()
    shap.plots.bar(
        shap_values[:, :, 1],
        max_display=15,
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)

    models = {
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
    }

    performance_results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        performance_results.append({
            "Model": name,
            "ROC AUC": round(roc_auc_score(y_test, y_prob), 2),
            "F1 Score": round(f1_score(y_test, y_pred), 2),
            "Precision": round(precision_score(y_test, y_pred), 2),
            "Recall": round(recall_score(y_test, y_pred), 2)
        })

    performance_df = pd.DataFrame(performance_results)
    




    

    st.subheader("Model Performance")
    st.dataframe(performance_df)
    st.subheader("ROC Curve")

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():

        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        auc_score = roc_auc_score(y_test, y_prob)

        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_score:.2f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Model Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


    st.subheader("Confusion Matrix")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, model) in zip(axes, models.items()):

        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        ax.imshow(cm)

        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_yticklabels(["No Churn", "Churn"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    st.subheader("Model Comparison")

    comparison_df = performance_df.set_index("Model")[
        ["ROC AUC", "F1 Score", "Precision", "Recall"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    comparison_df.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)

    ax.legend(title="Metric")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)















    st.subheader("Business Insights")
    st.markdown("""
### Key Drivers of Customer Churn

Based on the model analysis and feature importance results, several factors significantly influence customer churn:

**1️⃣ Customer Tenure**
- Customers with shorter tenure are much more likely to churn.
- New customers have a higher probability of leaving compared to long-term subscribers.

**2️⃣ Contract Type**
- Customers on **month-to-month contracts** show the highest churn risk.
- Long-term contracts such as **one-year or two-year agreements significantly reduce churn**.

**3️⃣ Monthly Charges**
- Higher monthly charges correlate with increased churn probability.
- Customers paying more are more likely to switch providers if they perceive better value elsewhere.

**4️⃣ Internet Service Type**
- Customers using **fiber optic internet services** show relatively higher churn rates compared to DSL users.

**5️⃣ Lack of Value-Added Services**
- Customers without services like **online security, tech support, or device protection** are more likely to churn.

---

### Business Recommendations

• Encourage **long-term contracts** through discounts or loyalty rewards.  
• Offer **bundled services (security, tech support)** to increase customer retention.  
• Provide **special retention offers for high-charge customers** to reduce churn risk.  
• Focus retention campaigns on **new customers with low tenure**.
""")





# this is the nw file 
