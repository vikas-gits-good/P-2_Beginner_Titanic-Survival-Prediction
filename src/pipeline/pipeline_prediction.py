import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Attempting to predict score based on user inputs")

            model_path = "artifacts/04_Models/best_model.pkl"
            preprc_path = "artifacts/03_Pipeline/ppln_prpc.pkl"

            model = load_object(file_path=model_path)
            preprc = load_object(file_path=preprc_path)

            x_pred_tf = preprc.fit_transform(features)
            y_pred = model.predict(x_pred_tf)

            return y_pred

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)


class CustomData:
    def __init__(
        self,
        Pclass: int,
        Sex: str,
        Age: int,
        SibSp: int,
        Parch: int,
        Fare: float,
        Cabin: str,
        Embarked: str,
    ):
        self.Pclass = int(Pclass)
        self.Sex = str(Sex)
        self.Age = int(Age)
        self.SibSp = int(SibSp)
        self.Parch = int(Parch)
        self.Fare = float(Fare)
        self.Cabin = str(Cabin)
        self.Embarked = str(Embarked)

    def get_DataFrame(self):
        try:
            logging.info("Converting user inputs to DataFrame")
            cust_data = {
                "Pclass": [self.Pclass],
                "Sex": [self.Sex],
                "Age": [self.Age],
                "SibSp": [self.SibSp],
                "Parch": [self.Parch],
                "Fare": [self.Fare],
                "Cabin": [self.Cabin],
                "Embarked": [self.Embarked],
            }
            df_cust = pd.DataFrame(cust_data)
            logging.info("Successfully converted user inputs to DataFrame")
            return df_cust

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
