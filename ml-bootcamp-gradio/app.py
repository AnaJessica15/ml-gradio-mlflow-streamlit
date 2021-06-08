import gradio as gr
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression




class TactModel(object):
    
    def __init__(self, csv_path, pickle_path = None):
        
        self.csv_filepath = csv_path
        self.pickle_filepath = pickle_path
        self.model = None
        
    def make_model(self):
        
        df = pd.read_csv(self.csv_filepath)
        df.columns = df.columns.str.strip()
            
        df.dropna(how='any', inplace=True)

        # Train and Build
        X= df[["Experience", "Learning mindset", "AWS", "Looking for product company"]]
        y = df[["Joined Bootcamp"]]
        
        dtrmodel = LogisticRegression()
        
        dtrmodel.fit(X, y)
        
        self.model = dtrmodel
    
    def save_model_to_pickle(self):
        pass
    
    def load_model(self):
        
        return self.model
        
    def predict(self, data):
        
        prediction = self.model.predict(data)
        
        return prediction[0]


def joined_bootcamp( experience, learning, aws,product_company):

    tact_model = TactModel('BootCamp Prediction - India.csv')
    tact_model.make_model()

    df = [[experience, learning, aws,product_company]]
    pred = tact_model.model.predict_proba(df)[0]
    return {'Will Join': pred[0], 'Will not Join': pred[1]}

exp = gr.inputs.Slider(1,10, label="Experience")
learn = gr.inputs.Radio(["No", "Yes"], type="index")
aws = gr.inputs.Radio(["No", "Yes"], type="index")
product = gr.inputs.Radio(["No", "Yes"], type="index")

gr.Interface(joined_bootcamp, [exp, learn, aws, product], "label", live=True, title="What's the probability they will join the Bootcamp").launch(share=True)


