import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class predictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path =  "artifacts\preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor =  load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    

class CustomData:
    def __init__(self,price:float,p1h:float,p24h:float,p7d:float,p24h_volume:float):
        self.price = price
        self.p1h = p1h
        self.p7d = p7d
        self.p24h = p24h
        self.p24h_volume = p24h_volume

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "price" : [self.price],
                "1h": [self.p1h],
                "24h":[self.p24h],
                "7d":[self.p7d],
                "24h_volume" : [self.p24h_volume]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)        