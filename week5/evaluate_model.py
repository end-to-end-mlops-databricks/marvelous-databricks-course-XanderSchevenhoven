"""
This script evaluates and compares a new house price prediction model against the currently deployed model.
Key functionality:
- Loads test data and performs feature engineering
- Generates predictions using both new and existing models
- Calculates and compares performance metrics (MAE and RMSE)
- Registers the new model if it performs better
- Sets task values for downstream pipeline steps

The evaluation process:
1. Loads models from the serving endpoint
2. Prepares test data with feature engineering
3. Generates predictions from both models
4. Calculates error metrics
5. Makes registration decision based on MAE comparison
6. Updates pipeline task values with results
"""

from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime
import mlflow
import argparse
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator

from power import ProjectConfig, to_snake



parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--new_model_uri",
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

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
root_path = args.root_path
new_model_uri = args.new_model_uri
job_run_id = args.job_run_id
git_sha = args.git_sha

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

# Define the serving endpoint
serving_endpoint_name = "power-consumption-model-serving-fe"
serving_endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
model_name = serving_endpoint.config.served_models[0].model_name
model_version = serving_endpoint.config.served_models[0].model_version
previous_model_uri = f"models:/{model_name}/{model_version}"

# Load test set and create additional features in Spark DataFrame
current_year = datetime.now().year
test_set = spark.table(f"{full_schema_name}.test_set")
test_set = test_set.withColumn("temperature", col("temperature").cast("int"))
test_set = test_set.withColumn("house_age", F.lit(current_year) - col("YearBuilt"))

# Select the necessary columns for prediction and target
X_test_spark = test_set.select(numeric_features_clean + ["temperature", "date_time"])
y_test_spark = test_set.select("date_time", target_clean)

# Generate predictions from both models
predictions_previous = fe.score_batch(model_uri=previous_model_uri, df=X_test_spark)
predictions_new = fe.score_batch(model_uri=new_model_uri, df=X_test_spark)

predictions_new = predictions_new.withColumnRenamed("prediction", "prediction_new")
predictions_old = predictions_previous.withColumnRenamed("prediction", "prediction_old")
test_set = test_set.select("date_time", target_clean)

# Join the DataFrames on the 'date_time' column
df = (
    test_set
    .join(predictions_new, on="date_time")
    .join(predictions_old, on="date_time")
)

# Calculate the absolute error for each model
df = df.withColumn("error_new", F.abs(df[target_clean] - df["prediction_new"]))
df = df.withColumn("error_old", F.abs(df[target_clean] - df["prediction_old"]))

# Calculate the absolute error for each model
df = df.withColumn("error_new", F.abs(df[target_clean] - df["prediction_new"]))
df = df.withColumn("error_old", F.abs(df[target_clean] - df["prediction_old"]))

# Calculate the Mean Absolute Error (MAE) for each model
mae_new = df.agg(F.mean("error_new")).collect()[0][0]
mae_old = df.agg(F.mean("error_old")).collect()[0][0]

# Calculate the Root Mean Squared Error (RMSE) for each model
evaluator = RegressionEvaluator(labelCol=target_clean, predictionCol="prediction_new", metricName="rmse")
rmse_new = evaluator.evaluate(df)

evaluator.setPredictionCol("prediction_old")
rmse_old = evaluator.evaluate(df)

# Compare models based on MAE and RMSE
print(f"MAE for New Model: {mae_new}")
print(f"MAE for Old Model: {mae_old}")

if mae_new < mae_old:
    print("New model is better based on MAE.")
    model_version = mlflow.register_model(
      model_uri=new_model_uri,
      name=f"{full_schema_name}.house_prices_model_fe",
      tags={"git_sha": f"{git_sha}",
            "job_run_id": job_run_id})

    print("New model registered with version:", model_version.version)
    dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
    dbutils.jobs.taskValues.set(key="model_update", value=1)
else:
    print("Old model is better based on MAE.")
    dbutils.jobs.taskValues.set(key="model_update", value=0)