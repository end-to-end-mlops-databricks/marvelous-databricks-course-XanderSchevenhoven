from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


class DataProcessor:
    def __init__(self, config):
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
        self.pdf = self.load_data()
        self.config = config
        self.pdf_features = None
        self.pdf_target = None
        self.preprocessor = None

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

    def preprocess_data(self):
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
        required_keys = ["target_column_name", "numeric_features"]
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise KeyError(f"Missing required config keys: {missing_keys}")

        # get target_column_name from config
        target_column_name = self.config["target_column_name"]
        numeric_features = self.config["numeric_features"]

        if not numeric_features:
            raise ValueError("numeric_features cannot be empty")

        if not all(col in self.pdf.columns for col in numeric_features + [target_column_name]):
            raise ValueError("Some specified columns not found in dataset")

        # remove rows with missing values in target
        self.pdf = self.pdf.dropna(subset=[target_column_name])

        # separate features and target into their own dataframes
        self.pdf_features = self.pdf[numeric_features]
        self.pdf_target = self.pdf[target_column_name]

        # create sklearn preprocessing pipeline steps
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        # combine preprocessing steps into single transformer
        self.preprocessor = ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features)])

    def split_data(self, test_size=0.2, random_state=42):
        """
        Method to split the dataset into training and test datasets.
        """
        return train_test_split(self.pdf_features, self.pdf_target, test_size=test_size, random_state=random_state)
