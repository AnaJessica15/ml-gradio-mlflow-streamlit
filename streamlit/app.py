import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():

    st.title('Introduction to Migration Data')
    st.sidebar.title('This is the sidebar')    
    @st.cache(persist=True)

    def load_data():
        data=pd.read_csv("Migration Data - Sheet1.csv")
        data.dropna(inplace=True) 
        return data
  
    def plot_metrics(matrics_list):
        if 'Scatter Plot' in matrics_list:
            st.subheader("Scatter Plot")
            plt.scatter(df["DATE"],df["DATA_MIGRATED"])
            st.pyplot()
        if 'Histogram' in matrics_list:
            st.subheader("Histogram")
            fig, ax =plt.subplots(figsize=(70,50))
            sns.barplot(df['DATE'], df['DATA_MIGRATED'])
            plt.xticks(rotation=45)
            st.pyplot()
        if 'Line Plot' in matrics_list:
            st.subheader("Line Plot")
            fig, ax =plt.subplots(figsize=(70,50))
            ax.plot(df['DATE'], df['DATA_MIGRATED'])
            plt.xticks(rotation=45)
            st.pyplot()
    df = load_data()
    
  
    metrics = st.sidebar.multiselect("What to plot?", ('Scatter Plot','Histogram','Line Plot'))
    if st.sidebar.button("Plot", key='Find'):
            plot_metrics(metrics)

if __name__ == '__main__':
    main()