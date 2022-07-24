import streamlit as st
import pandas as pd
import requests
import os
import json

# GLOBALS
FLASK_ADDRESS = os.environ.get("FLASK_ADDRESS", "http://0.0.0.0")
FLASK_PORT = os.environ.get("FLASK_PORT", "5000")
MODEL_DICT = {"Logistic Regression": "log_reg", "SVM": "svm", "KNN": "knn"}

# Functions
def get_all_data():
    resp = requests.get(f"{FLASK_ADDRESS}:{FLASK_PORT}/get/all").json()
    df = pd.DataFrame.from_dict(resp)
    df['age'] = df['age'].astype("int32", errors="ignore")
    # convert fare to 2 decimal place float
    df['fare'] = df['fare'].round(2)
    # encoding male and female to binary values for classification purposes
    df.loc[df['sex'] == 0, "sex"] = "M"
    df.loc[df['sex'] == 1, "sex"] = "F"
    # map status
    status = {1:"upper", 2:"middle", 3:"lower"}
    df['pclass'] = df['pclass_id'].replace(status)
    df = df.drop(columns=['pclass_id'])
    return df

def filter_data(sex, status, survived):
    df = get_all_data()
    # parameter maps
    sex_dict = {"Male": "M", "Female": "F"}
    class_dict = {"Upper":"upper", "Middle":"middle", "Lower":"lower"}
    survived_dict = {"Yes": 1, "No": 0}
    # filter for sex
    if sex != "All":
        df = df[df["sex"] == sex_dict[sex]]
    # filter for status
    if status != "All":
        df = df[df["pclass"] == class_dict[status]]
    # filter for survival
    if survived != "All":
        df = df[df["survived"] == survived_dict[survived]]

    return df

def main():
    app_mode = st.sidebar.selectbox('Select Page', ["Home", "Train Model"])

    if app_mode == "Home":
        st.title("Titanic for Machine Learning")
        st.image("images/titanic-new.png")
        st.markdown("Dataset: ")
        # Filter data
        st.sidebar.header("Filter Data:")
        sex = st.sidebar.radio('Sex:', ("All", "Male", "Female"))
        status = st.sidebar.radio('Class:', ("All", "Upper", "Middle", "Lower"))
        survived = st.sidebar.radio('Survived:', ("All", "Yes", "No"))
        filter = st.sidebar.button("Filter")

        if filter:
            st.write(filter_data(sex, status, survived))
        else:
            st.write(get_all_data())


    if app_mode == "Train Model":
        st.title("Train Model")
        st.subheader("Use the radio buttons in the sidebar to select a model.")
        model = st.sidebar.radio('Select Model', ("Logistic Regression", "SVM", "KNN"))
        train = st.sidebar.button("Train")

        # Act on button submit
        if train:
            resp = requests.post(f"{FLASK_ADDRESS}:{FLASK_PORT}/train", json={"model": MODEL_DICT[model]}).json()
            precision = resp["precision"]
            recall = resp["recall"]
            f1 = resp["f1"]
            st.success(f"Successfully trained a {model}!")
            st.markdown("Results: ")
            st.info(f'''
                Precision: {precision} \n
                Recall: {recall}\n
                F1-Score: {f1}
            ''')



if __name__ == '__main__':
    main()
