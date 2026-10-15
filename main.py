# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import base64

# Show a matplotlib figure by embedding it directly in the page
# (avoids Streamlit's media file server, which was returning 500 errors)
def show_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" style="width:100%">',
        unsafe_allow_html=True,
    )

# Configure the Streamlit page
st.set_page_config(
    page_title="DrugVision",
    page_icon="💊",
    layout="wide"
)

# Load the dataset only once
@st.cache_data
def load_data():
    return pd.read_csv("drug200.csv")

# Load the label encoder and models only once
@st.cache_resource
def load_models():
    with open("labelencoder.pkl", "rb") as f:
        cholesterol_encoder = pickle.load(f)
    with open("decision_tree.pkl", "rb") as f:
        dt_model = pickle.load(f)
    with open("random_forest.pkl", "rb") as f:
        rf_model = pickle.load(f)
    return (cholesterol_encoder, dt_model, rf_model)

# Prepare the processed feature set matching how the models were trained
@st.cache_data
def prepare_features(raw_data):
    # Copy the raw data so the original stays untouched
    processed = raw_data.copy()
    # Encode Sex the same way as training (F -> 1, M -> 0)
    processed["Sex"] = processed["Sex"].replace({"F": 1, "M": 0}).astype(int)
    # One-hot encode BP into HIGH / LOW / NORMAL columns
    bp_dummies = pd.get_dummies(processed["BP"])
    # Drop the original BP column
    processed = processed.drop("BP", axis=1)
    # Attach the one-hot BP columns
    processed = pd.concat([processed, bp_dummies], axis=1)
    # Return the processed dataframe
    return processed

# Prepare model evaluation data and predictions only once
@st.cache_data
def prepare_evaluation_data(_dt_model, _rf_model, processed_data):
    # Separate features and target
    X = processed_data.drop("Drug", axis=1)
    Y = processed_data["Drug"]
    # Split using the same random_state as the notebook
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=48)
    # Generate predictions
    DT_predict = _dt_model.predict(X_test)
    RF_predict = _rf_model.predict(X_test)
    return (Y_test, DT_predict, RF_predict)

# Load the dataset
data = load_data()

# Load the encoder and models
cholesterol_encoder, dt_model, rf_model = load_models()

# Prepare the processed feature set (Sex mapped, BP one-hot, Cholesterol label-encoded)
processed_data = prepare_features(data)
# Encode Cholesterol using the saved label encoder
processed_data["Cholesterol"] = cholesterol_encoder.transform(processed_data["Cholesterol"])

# Reorder columns to match the training feature order
feature_order = ["Age", "Sex", "Cholesterol", "Na_to_K", "HIGH", "LOW", "NORMAL", "Drug"]
processed_data = processed_data[feature_order]

# Prepare model evaluation data
Y_test, DT_predict, RF_predict = prepare_evaluation_data(dt_model, rf_model, processed_data)

# Display the app title
st.title("💊 DrugVision")
# Display the app description
st.write("Drug Type Classification using Machine Learning")

# Add a divider
st.divider()

# Create top navigation bar using tabs
tab_home, tab_predict = st.tabs(["🏠 Home", "🔮 Predictor"])

# Small helper to keep chart sizes consistent and compact for card layout
CARD_FIGSIZE = (4.5, 3)

# Create the user input function
def get_user_input():
    # Display the input section
    st.subheader("Enter Patient Details")

    # Let user choose input mode
    input_mode = st.radio(
        "Input Method",
        ["✍️ Manual Input", "🎲 Load Sample"],
        horizontal=True,
        key="input_mode"
    )

    # Set default values from the raw dataset
    default_age = int(data["Age"].mean())
    default_na_to_k = float(data["Na_to_K"].mean())
    default_sex = "F"
    default_bp = "HIGH"
    default_cholesterol = "HIGH"

    # Handle sample loading mode
    if input_mode == "🎲 Load Sample":
        # Explain the sample option
        st.info("Pick a row index from the dataset to auto-fill the values below.")

        # Initialize session state index if not present
        if "sample_idx" not in st.session_state:
            st.session_state["sample_idx"] = 0

        # Sample index picker
        sample_idx = st.number_input(
            "Sample row index",
            min_value=0,
            max_value=len(data) - 1,
            value=st.session_state["sample_idx"],
            step=1,
            key="idx_input"
        )
        # Update session state with chosen index
        st.session_state["sample_idx"] = sample_idx

        # Show the valid row range for reference
        st.caption(f"Valid row range: 0 to {len(data) - 1} ({len(data)} total rows)")

        # Pull the sample row from the raw (unprocessed) dataset
        sample_row = data.iloc[st.session_state["sample_idx"]]
        default_age = int(sample_row["Age"])
        default_sex = sample_row["Sex"]
        default_bp = sample_row["BP"]
        default_cholesterol = sample_row["Cholesterol"]
        default_na_to_k = float(sample_row["Na_to_K"])

        # Show the actual drug for that sample
        st.caption(f"Actual Drug for row {st.session_state['sample_idx']}: {sample_row['Drug']}")

        # Build a live preview of the selected sample row
        preview_df = pd.DataFrame([sample_row])
        # Display the live preview table
        st.write("**Selected Row Preview:**")
        st.dataframe(preview_df, width="stretch", hide_index=True)

    else:
        # Explain manual input
        st.info("Enter the patient details manually.")

    # Use the sample index (if any) as part of widget keys so fields refresh on change
    refresh_key = st.session_state.get("sample_idx", "manual")

    # Create input columns
    col1, col2, col3 = st.columns(3)

    # First column
    with col1:
        Age = st.number_input(
            "Age",
            min_value=int(data["Age"].min()),
            max_value=int(data["Age"].max()),
            value=default_age,
            key=f"Age_{input_mode}_{refresh_key}"
        )
        Sex = st.selectbox(
            "Sex",
            ["F", "M"],
            index=["F", "M"].index(default_sex),
            key=f"Sex_{input_mode}_{refresh_key}"
        )

    # Second column
    with col2:
        BP = st.selectbox(
            "Blood Pressure (BP)",
            ["HIGH", "LOW", "NORMAL"],
            index=["HIGH", "LOW", "NORMAL"].index(default_bp),
            key=f"BP_{input_mode}_{refresh_key}"
        )
        Cholesterol = st.selectbox(
            "Cholesterol",
            ["HIGH", "NORMAL"],
            index=["HIGH", "NORMAL"].index(default_cholesterol),
            key=f"Cholesterol_{input_mode}_{refresh_key}"
        )

    # Third column
    with col3:
        Na_to_K = st.number_input(
            "Na to K Ratio",
            min_value=float(data["Na_to_K"].min()),
            max_value=float(data["Na_to_K"].max()),
            value=default_na_to_k,
            key=f"NaK_{input_mode}_{refresh_key}"
        )

    # Encode Sex the same way as training
    sex_encoded = 1 if Sex == "F" else 0
    # Encode Cholesterol using the saved label encoder
    cholesterol_encoded = int(cholesterol_encoder.transform([Cholesterol])[0])
    # One-hot encode BP manually
    high_flag = 1 if BP == "HIGH" else 0
    low_flag = 1 if BP == "LOW" else 0
    normal_flag = 1 if BP == "NORMAL" else 0

    # Combine input values in the same order as training
    input_data = pd.DataFrame(
        [[Age, sex_encoded, cholesterol_encoded, Na_to_K, high_flag, low_flag, normal_flag]],
        columns=["Age", "Sex", "Cholesterol", "Na_to_K", "HIGH", "LOW", "NORMAL"]
    )

    # Return the prepared input
    return input_data

# Function to show predictions from both models combined into one result
def show_prediction(input_data):
    # Display section title
    st.subheader("Prediction Result")

    # Generate predictions from both models
    dt_pred = dt_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]

    # Build result table combining both models
    result = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest"],
        "Predicted Drug": [dt_pred, rf_pred]
    })

    # Display result table
    st.dataframe(result, width="stretch", hide_index=True)

    # Check if both models agree
    if dt_pred == rf_pred:
        st.success(f"✅ Predicted Drug: {dt_pred}")
    else:
        st.warning(f"⚠️ Models disagree — Decision Tree: {dt_pred}, Random Forest: {rf_pred}")

# Function to handle batch prediction from an uploaded CSV using both models
def handle_batch_prediction():
    # Display section title
    st.subheader("📂 Batch Prediction from CSV")
    # Explain expected format
    st.write("Upload a CSV file with the following columns: Age, Sex, BP, Cholesterol, Na_to_K")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")

    # Process the uploaded file
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            batch_data = pd.read_csv(uploaded_file)
            # Define required columns
            required_cols = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
            # Check for missing columns
            missing_cols = [c for c in required_cols if c not in batch_data.columns]

            # Show error if columns are missing
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Copy the batch data for processing
                X_batch = batch_data.copy()
                # Encode Sex the same way as training
                X_batch["Sex"] = X_batch["Sex"].replace({"F": 1, "M": 0}).astype(int)
                # Encode Cholesterol using the saved label encoder
                X_batch["Cholesterol"] = cholesterol_encoder.transform(X_batch["Cholesterol"])
                # One-hot encode BP
                bp_dummies = pd.get_dummies(X_batch["BP"])
                # Ensure all three BP columns exist even if not all present in the upload
                for col in ["HIGH", "LOW", "NORMAL"]:
                    if col not in bp_dummies.columns:
                        bp_dummies[col] = 0
                # Drop the original BP column
                X_batch = X_batch.drop("BP", axis=1)
                # Attach the one-hot BP columns
                X_batch = pd.concat([X_batch, bp_dummies], axis=1)
                # Reorder columns to match training feature order
                X_batch_final = X_batch[["Age", "Sex", "Cholesterol", "Na_to_K", "HIGH", "LOW", "NORMAL"]]

                # Generate predictions from both models
                result = batch_data.copy()
                result["Decision Tree Prediction"] = dt_model.predict(X_batch_final)
                result["Random Forest Prediction"] = rf_model.predict(X_batch_final)

                # Display the results
                st.success(f"Predictions generated for {len(result)} rows.")
                st.dataframe(result, width="stretch")

                # Prepare CSV for download
                csv_bytes = result.to_csv(index=False).encode("utf-8")
                # Display download button
                st.download_button(
                    "⬇️ Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="drug_predictions.csv",
                    mime="text/csv",
                    key="batch_download"
                )
        except Exception as e:
            # Display any processing errors
            st.error(f"Error processing file: {e}")

# Display Home tab content (combined EDA + Model Comparison, card-style layout)
with tab_home:
    # Display overview metrics as cards
    st.subheader("Dataset Overview")
    # Create metric columns
    m1, m2, m3, m4 = st.columns(4)
    # Display total samples
    with m1:
        with st.container(border=True):
            st.metric("Total Samples", data.shape[0])
    # Display number of features
    with m2:
        with st.container(border=True):
            st.metric("Input Features", data.shape[1] - 1)
    # Display number of drug types
    with m3:
        with st.container(border=True):
            st.metric("Drug Types", data["Drug"].nunique())
    # Display missing values
    with m4:
        with st.container(border=True):
            st.metric("Missing Values", data.isnull().sum().sum())

    # Add spacing
    st.write("")

    # Display model accuracy as cards
    st.subheader("Model Accuracy")
    # Calculate accuracies
    dt_accuracy = accuracy_score(Y_test, DT_predict)
    rf_accuracy = accuracy_score(Y_test, RF_predict)
    # Create metric columns for accuracy
    a1, a2 = st.columns(2)
    # Display Decision Tree accuracy
    with a1:
        with st.container(border=True):
            st.metric("Decision Tree", f"{dt_accuracy * 100:.2f}%")
    # Display Random Forest accuracy
    with a2:
        with st.container(border=True):
            st.metric("Random Forest", f"{rf_accuracy * 100:.2f}%")

    # Display best model callout
    best_model = "Decision Tree" if dt_accuracy >= rf_accuracy else "Random Forest"
    best_accuracy = max(dt_accuracy, rf_accuracy)
    st.success(f"Best performing model: {best_model} with an accuracy of {best_accuracy * 100:.2f}%.")

    # Add spacing
    st.write("")
    st.subheader("Charts & Insights")

    # Row 1: Drug distribution card + Age distribution card
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        with st.container(border=True):
            st.write("**Drug Type Distribution**")
            drug_counts = data["Drug"].value_counts().reset_index()
            drug_counts.columns = ["Drug", "Count"]
            fig, ax = plt.subplots(figsize=CARD_FIGSIZE)
            sns.barplot(data=drug_counts, x="Drug", y="Count", ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            show_fig(fig)
            plt.close(fig)
    with r1c2:
        with st.container(border=True):
            st.write("**Age Distribution**")
            fig, ax = plt.subplots(figsize=CARD_FIGSIZE)
            sns.histplot(data=data, x="Age", kde=True, ax=ax)
            ax.set_ylabel("Count")
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            show_fig(fig)
            plt.close(fig)

    # Row 2: Na_to_K distribution card + Categorical breakdown card
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        with st.container(border=True):
            st.write("**Na to K Ratio Distribution**")
            fig, ax = plt.subplots(figsize=CARD_FIGSIZE)
            sns.histplot(data=data, x="Na_to_K", kde=True, ax=ax)
            ax.set_ylabel("Count")
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            show_fig(fig)
            plt.close(fig)
    with r2c2:
        with st.container(border=True):
            st.write("**Feature Breakdown by Drug**")
            selected_cat_feature = st.selectbox(
                "Select a feature", ["Sex", "BP", "Cholesterol"], key="cat_feature_select"
            )
            fig, ax = plt.subplots(figsize=CARD_FIGSIZE)
            sns.countplot(data=data, x=selected_cat_feature, hue="Drug", ax=ax)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=6)
            plt.tight_layout()
            show_fig(fig)
            plt.close(fig)

    # Row 3: Correlation heatmap card + Accuracy comparison chart card
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        with st.container(border=True):
            st.write("**Feature Correlation**")
            fig, ax = plt.subplots(figsize=CARD_FIGSIZE)
            sns.heatmap(
                processed_data.drop("Drug", axis=1).corr(),
                annot=True,
                annot_kws={"size": 6},
                ax=ax,
                cbar=False
            )
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            show_fig(fig)
            plt.close(fig)
    with r3c2:
        with st.container(border=True):
            st.write("**Accuracy Comparison**")
            comparison_data = pd.DataFrame({
                "Model": ["Decision Tree", "Random Forest"],
                "Accuracy (%)": [dt_accuracy * 100, rf_accuracy * 100]
            })
            fig, ax = plt.subplots(figsize=CARD_FIGSIZE)
            sns.barplot(data=comparison_data, x="Model", y="Accuracy (%)", ax=ax)
            ax.set_ylim(0, 100)
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            show_fig(fig)
            plt.close(fig)

    # Row 4: Confusion matrices, one card per model
    st.write("**Confusion Matrices**")
    r4c1, r4c2 = st.columns(2)
    labels = sorted(data["Drug"].unique())
    with r4c1:
        with st.container(border=True):
            st.write("Decision Tree")
            fig, ax = plt.subplots(figsize=CARD_FIGSIZE)
            sns.heatmap(
                confusion_matrix(Y_test, DT_predict, labels=labels),
                annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 7}, cbar=False, ax=ax
            )
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            show_fig(fig)
            plt.close(fig)
    with r4c2:
        with st.container(border=True):
            st.write("Random Forest")
            fig, ax = plt.subplots(figsize=CARD_FIGSIZE)
            sns.heatmap(
                confusion_matrix(Y_test, RF_predict, labels=labels),
                annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 7}, cbar=False, ax=ax
            )
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            show_fig(fig)
            plt.close(fig)

    # Detailed classification reports kept compact inside an expander
    with st.expander("📋 Detailed Classification Reports"):
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.write("**Decision Tree**")
            dt_report = classification_report(Y_test, DT_predict, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(dt_report).transpose(), width="stretch")
        with rep_col2:
            st.write("**Random Forest**")
            rf_report = classification_report(Y_test, RF_predict, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rf_report).transpose(), width="stretch")

# Display Predictor tab content
with tab_predict:
    # Display page title
    st.header("🔮 Predictor")
    # Display page description
    st.write("Enter patient details manually or load a sample row, then predict the recommended drug.")

    # Get user input
    input_data = get_user_input()

    # Predict drug type
    if st.button("🔮 Predict Drug", type="primary"):
        # Show the combined prediction result
        show_prediction(input_data)

    # Display divider before batch section
    st.divider()
    # Show batch prediction section
    handle_batch_prediction()