
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv("Iris.csv")
df.drop("Id", axis=1, inplace=True)

# Encode the target variable
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Split the data
x = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200, solver='lbfgs')
model.fit(x_train, y_train)

# Streamlit app
st.title("Iris Species Prediction")
st.write("This app predicts the species of the iris flower using Logistic Regression")

# Input features
sepal_length = st.slider("Sepal Length (cm)", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()))
sepal_width = st.slider("Sepal Width (cm)", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()))
petal_length = st.slider("Petal Length (cm)", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()))
petal_width = st.slider("Petal Width (cm)", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()))

# Prediction
if st.button("Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    predicted_species = le.inverse_transform([prediction])[0]
    st.success(f"Predicted Species: **{predicted_species}**")

# Model performance
with st.expander("Show Model Performance"):
    y_pred = model.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: **{acc:.2f}**")
    
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax)
    st.pyplot(fig)
