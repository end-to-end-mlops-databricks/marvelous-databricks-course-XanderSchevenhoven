from .config import ProjectConfig
from .utils import to_snake

from databricks.sdk.runtime import spark
import pyspark.sql.functions as f

import pandas as pd
from re import sub
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, config: ProjectConfig) -> None:
        """
        Initiate DataProcessor class and its attributes. The provided csv file_path is
        automatically loaded and stored in this class.

        Parameters
        ----------
        config : ProjectConfig
            ProjectConfig instance with with project configuration variables
        """
        self.config = config
        self.pdf = self.load_data()
        self.pdf_features = None
        self.pdf_target = None
        self.preprocessor = None
        self.full_schema_name = f"{self.config.catalog_name}.{self.config.schema_name}"
        

    def load_data(self):
        """
        Method to get the power consumption dataset as pandas dataframe.

        Returns
        -------
        Pandas DataFrame

        """
        # fetch dataset
        power_consumption_of_tetouan_city = fetch_ucirepo(id=849)

        # data (as pandas dataframes)
        return power_consumption_of_tetouan_city.data.original

    def prepare_data(self):
        """
        Method to create preprocessing pipeline steps.

        This method modifies the following instance attributes:
        - self.pdf: Removes rows with missing target values
        - self.pdf_features: Sets features DataFrame
        - self.pdf_target: Sets target Series
        - self.preprocessor: Sets the sklearn preprocessing pipeline

        Raises
        ------
        KeyError
            If required configuration keys are missing
        ValueError
            If configuration values are invalid
        """

        # get target_column_name from config
        target_column_name = self.config.target
        numeric_features = self.config.numeric_features

        if not numeric_features:
            raise ValueError("numeric_features cannot be empty")

        if not all(col in self.pdf.columns for col in numeric_features + [target_column_name]):
            raise ValueError("Some specified columns not found in dataset")

        # remove rows with missing values in target
        self.pdf = self.pdf.dropna(subset=[target_column_name])

        # convert column names to snake strings
        self.pdf.columns = [
            to_snake(column) for column in self.pdf.columns
        ]
        self.target = to_snake(target_column_name)
        self.numeric_features = [
            to_snake(column) for column in numeric_features
        ]
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Method to split the dataset into training and test datasets.

        Parameters
        ----------
        test_size : float, optional (default=0.2)
            Proportion of the dataset to include in the test split
        random_state : int, optional (default=42)
            Random state for reproducibility

        Returns
        -------
        tuple
            train_set, test_set split of the dataset

        Raises
        ------
        ValueError
            If preprocessing hasn't been performed or parameters are invalid
        """
        if self.pdf is None:
            raise ValueError("Must call preprocess_data before splitting")

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        return train_test_split(self.pdf, test_size=test_size, random_state=random_state)
    
    def _add_current_timestamp_column(self, sdf):
        return (
            sdf
            .withColumn("update_timestamp", f.current_timestamp())
        )

    def _convert_pandas_to_spark(self, pdf):
        return spark.createDataFrame(pdf)
    
    def _append_to_table(self, sdf, table_name):
        
        full_table_name = f"{self.full_schema_name}.{table_name}"
        
        # append to table
        (
            sdf
            .write
            .mode('append')
            .saveAsTable(full_table_name)
        )

        # set properties
        spark.sql(
            f"""
            ALTER TABLE {full_table_name}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
            """
        )

    
    def save_to_catalog(self, pdf_train_set: pd.DataFrame, pdf_test_set: pd.DataFrame):
        """
        Save the train and test sets into Databricks tables.
        """

        # convert pandas to spark
        sdf_train_set = self._convert_pandas_to_spark(pdf_train_set)
        sdf_test_set = self._convert_pandas_to_spark(pdf_test_set)

        # add update timestamp column
        sdf_train_set = self._add_current_timestamp_column(sdf_train_set)
        sdf_test_set = self._add_current_timestamp_column(sdf_test_set)

        # write
        self._append_to_table(sdf_train_set, "train_set")
        self._append_to_table(sdf_test_set, "test_set")
