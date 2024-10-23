import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataProcessor:
    def __init__(self, file_path, config):
        """
        Initiate DataProcessor class and its attributes. The provided csv file_path is
        automatically loaded and stored in this class.

        Parameters
        ----------
        file_path : str
            File path to the location of a CSV file.
        config : dict
            Dictionary with configuration options for preprocessing.

        """
        self.pdf = self.load_data(file_path)
        self.config = config
        self.pdf_features = None
        self.pdf_target = None
        self.preprocessor = None

    def load_data(self, file_path):
        """
        Method to read csv file as pandas dataframe.

        Parameters
        ----------
        file_path : str
            File path to the location of a CSV file.

        Returns
        -------
        Pandas DataFrame

        """
        return pd.read_csv(file_path)

    def preprocess_data(self):
        """
        Method to create preprocessing pipeline steps.
        """

        # get target_column_name from config
        target_column_name = self.config["target_column_name"]
        numeric_features = self.config["numeric_features"]
        category_features = self.config["category_features"]
        all_features = numeric_features + category_features

        # remove rows with missing values in target
        self.pdf = self.pdf.dropna(subset=[target_column_name])

        # separate features and target into their own dataframes
        self.pdf_features = self.pdf[all_features]
        self.pdf_target = self.pdf[target_column_name]

        # create sklearn preprocessing pipeline steps
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        category_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # combine preprocessing steps into single transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categories", category_transformer, category_features),
            ]
        )

    def split_data(self, test_size=0.2, random_state=42):
        """
        Method to split the dataset into training and test datasets.
        """
        return train_test_split(self.pdf_features, self.pdf_target, test_size=test_size, random_state=random_state)
