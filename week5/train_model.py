"""
This script trains a LightGBM model for house price prediction with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM regressor
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups and custom feature functions
- Outputs model URI for downstream tasks

The model uses both numerical and categorical features, including a custom calculated house age feature.
"""

from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
import argparse
from pyspark.sql import functions as f
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from power import DataProcessor, ProjectConfig, to_snake



parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
git_sha = args.git_sha
job_run_id = args.job_run_id

config_path = (f"{root_path}/project_config.yml")
config = ProjectConfig.from_yaml(config_path)

print("Configuration loaded:")
print(config)

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# define extra configs
full_schema_name = f"{config.catalog_name}.{config.schema_name}"
numeric_features_clean = [to_snake(column) for column in config.numeric_features]
target_clean = to_snake(config.target)

# Define table names and function name
feature_table_name = f"{full_schema_name}.power_features"
function_name = f"{full_schema_name}.convert_celsius_to_fahrenheit"

# Load training and test sets
train_set = (
    spark.table(f"{full_schema_name}.train_set")
    .withColumn("temperature_fahrenheit", f.col("temperature").cast("int"))
    .drop("temperature", "humidity")
)
test_set = spark.table(f"{full_schema_name}.test_set").toPandas()

# setup feature engineering
training_set = fe.create_training_set(
    df=train_set,
    label=target_clean,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["humidity"],
            lookup_key="date_time",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="temperature",
            input_bindings={"temperature": "temperature_fahrenheit"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate temp fahrenheit for test set
test_set["temperature"] = test_set["temperature"] * (9 / 5) + 32

# Split features and target
X_train = training_df[numeric_features_clean]
y_train = training_df[target_clean]
X_test = test_set[numeric_features_clean]
y_test = test_set[target_clean]

# create numeric features transformer
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

# combine preprocessing steps into single transformer
preprocessor = ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features_clean)])

# create regressor
regressor = RandomForestRegressor(
    n_estimators=config.parameters["n_estimators"],
    max_depth=config.parameters["max_depth"],
    random_state=42,
)

# create sklearn pipeline with preprocssor and regressor
model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name=config.experiment_name)

with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate and print metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForest with preprocessing")
    mlflow.log_params(config.parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=model,
        flavor=mlflow.sklearn,
        artifact_path=config.model_artifact_name,
        training_set=training_set,
        signature=signature,
    )

model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe'
mlflow.register_model(
    model_uri=model_uri, 
    name=f"{full_schema_name}.{config.model_name}_fe"
)
dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)