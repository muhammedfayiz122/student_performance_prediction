import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    """
    define the path for preprocessor object
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        """
        Initializing data transformation configurations
        """
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        """
        It creates pipeline for categorical and numerical columns
        """
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(steps = [
                    #Step1 : Impute missing value with the median
                    ('imputer', SimpleImputer(strategy = 'median')),
                    #Step2 : Standardize the data having mean=0 and variance=1
                    ('std_scalar', StandardScaler())
                ])

            cat_pipeline = Pipeline(steps = [
                    #Step1 : Impute missing value with the median
                    ('imputer', SimpleImputer(strategy = 'most_frequent')),
                    #Step2 : Encode categorical values
                    ('encoder', OneHotEncoder()),
                    #Step3 : Standardize the data having mean=0 and variance=1
                    ('std_scalar', StandardScaler(with_mean=False))
                ])
            
            logging.info(f"Pipeline created for numerical features : {numerical_columns}")
            logging.info(f"Pipeline created for categorical features : {categorical_columns}")

            #Combine numerical pipeline and categorical pipeline using ColumnTransformer
            preprocesser = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            logging.info("Combined numerical pipeline and categorical pipeline")

            return preprocesser
        except Exception as e:
            logging.info("Error in data preprocessing")
            raise CustomException(e,sys) 
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        It intializes the pipelines and save the preprocesses object
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train data and test data completed")

            #get the preprocessing object
            preprocessing_obj = self.get_data_transformer()

            #Define target column name
            target_col_name = 'math_score'

            #Define numerical columns naem
            numerical_columns = ['writing_score','reading_score']

            #Seperate input eatures and output features on train data
            input_feature_train_df = train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df = train_df[target_col_name]

            #Seperate input eatures and output features on test data
            input_feature_test_df = test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info(f"Applying preprocessing on training data and test data")

            #Apply the preprocessing pipeline to the training data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            #Apply the preprocessing pipeline to the test data
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #Combine the processed input feature with the target feature fro training data 
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
            #Combine the processed input feature with the target feature for testing data
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
                ]
            
            save_object(
                obj = preprocessing_obj,
                file_path = self.data_tranformation_config.preprocessor_obj_file_path
            )

            logging.info(f"Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
            



            

