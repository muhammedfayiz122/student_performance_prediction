import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    This class is used to store the configuration for data ingestion
    """
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')

class DataIngestion:
    def __init__(self):
        #storing path for raw, test, train
        self.ingestion_config = DataIngestionConfig()

    def intiate_data_ingestion(self):
        """
        This function is used to intiate the data ingestion process
        """
        try:
            self.df = pd.read_csv('data/stud.csv')
            logging.info("Read the dataset as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            logging.info("artifact folder created")

            self.df.to_csv(self.ingestion_config.raw_data_path)
            logging.info("raw data stored")

            logging.info("Train test split started....")
            train_set, test_set = train_test_split(self.df, test_size=0.2, random_state=48)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train test split completed")

        except Exception as e:
            logging.info(f"error : {e}")
            raise CustomException(e, sys)
        
        return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.intiate_data_ingestion()

    data_transformation = DataTransformation() 
    train_data, test_data, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    r_square = model_trainer.Initiate_model_trainer(train_data, test_data)
    # print(r_square)
    

"""
summary:
Just taking data source and save as raw_data, train_data and test_data
"""