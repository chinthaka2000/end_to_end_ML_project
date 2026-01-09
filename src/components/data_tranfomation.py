import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Create a pipeline for numerical columns (imputation and scaling)
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ])

            # Create a pipeline for categorical columns (imputation, one-hot encoding, scaling)
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),

            ])

            logging.info("Numerical and categorical columns: {}".format(numerical_columns + categorical_columns))

            # Combine the numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical", num_pipeline, numerical_columns),
                    ("categorical", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train data shape: {}".format(train_df.shape))
            logging.info("Test data shape: {}".format(test_df.shape))
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Dropping target column from train and test data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            # Fit the preprocessor on the training data and transform the train data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # Transform the test data using the already fitted preprocessor
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target for training data
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            # Combine features and target for testing data (test_arr is missing in the original code)
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object: {preprocessing_obj.__class__.__name__}")

            # Save the preprocessor object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


# Function to save the preprocessor object (you may already have this function)
def save_object(file_path, obj):
    try:
        import pickle
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Preprocessor object saved at {file_path}")
    except Exception as e:
        raise CustomException(f"Error in saving object: {e}", sys)
