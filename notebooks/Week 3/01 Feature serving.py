# Databricks notebook source
# MAGIC %md
# MAGIC # Excercise week 2 - preprocessing using python package
# MAGIC Steps:
# MAGIC 1. Load / open file
# MAGIC 2. Preprocess data
# MAGIC 3. Save to unity catalog

# COMMAND ----------
# MAGIC %md 
# MAGIC ## I. Setup notebook

# COMMAND ----------
# MAGIC %md 
# MAGIC ##### 1. Install / import libraries

# COMMAND ----------
# install package required to run if running in Databricks
get_ipython().run_line_magic("pip", 'install -e "../.."')
get_ipython().run_line_magic("pip", "install --upgrade databricks-sdk")

# COMMAND ----------
import time
import random
import requests
import pandas as pd
import mlflow
from concurrent.futures import ThreadPoolExecutor, as_completed

from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
# from databricks.sdk.runtime import spark, dbutils
from databricks.sdk.service.catalog import OnlineTableSpec
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, 
    ServedEntityInput
)

from power import ProjectConfig, to_snake

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# MAGIC %md 
# MAGIC ##### 2. Get configs

# COMMAND ----------

config = ProjectConfig.from_yaml("../project_config.yml")

print("Configuration loaded:")
print(config)

catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = config.model_name

# define extra configs
full_schema_name = f"{config.catalog_name}.{config.schema_name}"
numeric_features_clean = [to_snake(column) for column in config.numeric_features]
target_clean = to_snake(config.target)

# COMMAND ----------

# Define table names
feature_table_name = f"{full_schema_name}.power_consumption_preds"
online_table_name = f"{full_schema_name}.power_consumption_preds_online"

# Load training and test sets from Catalog
train_set = spark.table(f"{full_schema_name}.train_set").toPandas()
test_set = spark.table(f"{full_schema_name}.test_set").toPandas()

# combine train/test into single dataframe
df = pd.concat([train_set, test_set])

# COMMAND ----------
# Load the MLflow model for predictions
pipeline = mlflow.sklearn.load_model(f"models:/{full_schema_name}.{model_name}/1")

# COMMAND ----------

# Prepare the DataFrame for predictions and feature table creation - these features are the ones we want to serve.
serving_features = ["temperature", "humidity"]
serving_features_lookup_key = ["date_time"]
preds_df = df[serving_features_lookup_key + serving_features]
preds_df["predicted_power_consumption"] = pipeline.predict(df[numeric_features_clean])

preds_df = spark.createDataFrame(preds_df)

# COMMAND ----------

# 1. Create the feature table in Databricks

fe.create_table(
    name=feature_table_name, 
    primary_keys=["date_time"], 
    df=preds_df, 
    description="House Prices predictions feature table"
)

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# 2. Create the online table using feature table
# -> using OnlineTableSpec directly as in example didn't work for me due 
#    to ImportError for OnlineTableSpecTriggeredSchedulingPolicy
spec = OnlineTableSpec.from_dict(
    {
        "primary_key_columns": ['date_time'],
        'source_table_full_name': feature_table_name,
        'run_triggered': {'triggered': 'true'},
        'perform_full_copy': False
    }
)

# Create the online table in Databricks
online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------
# 3. Create feture look up and feature spec table feature table

# Define features to look up from the feature table
features = [
    FeatureLookup(
        table_name=feature_table_name, 
        lookup_key=serving_features_lookup_key, 
        feature_names=serving_features
    )
]

# Create the feature spec for serving
feature_spec_name = f"{full_schema_name}.return_predictions"

fe.create_feature_spec(
    name=feature_spec_name, 
    features=features, 
    exclude_columns=None
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Feature Serving Endpoint

# COMMAND ----------
# 4. Create endpoing using feature spec

# Create a serving endpoint for the house prices predictions
serving_endpoint_name = "power-consumption-feature-serving"
workspace.serving_endpoints.create(
    name=serving_endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=feature_spec_name,  # feature spec name defined in the previous step
                scale_to_zero_enabled=True,
                workload_size="Small",  # Define the workload size (Small, Medium, Large)
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call The Endpoint

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

dates_list = preds_df["date_time"]

# COMMAND ----------

datetime_to_look_up = "1/1/2017 21:40"

start_time = time.time()
serving_endpoint = f"https://{host}/serving-endpoints/{serving_endpoint_name}/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{serving_features_lookup_key[0]: datetime_to_look_up}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")


# COMMAND ----------
# another way to call the endpoint

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_split": {"columns": [serving_features_lookup_key[0]], "data": [[datetime_to_look_up]]}},
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Test

# COMMAND ----------
# Initialize variables
serving_endpoint = f"https://{host}/serving-endpoints/house-prices-feature-serving/invocations"
dates_list = preds_df.select(serving_features_lookup_key[0]).rdd.flatMap(lambda x: x).collect()
headers = {"Authorization": f"Bearer {token}"}
num_requests = 10


# Function to make a request and record latency
def send_request():
    random_id = random.choice(dates_list)
    start_time = time.time()
    response = requests.post(
        serving_endpoint,
        headers=headers,
        json={"dataframe_records": [{serving_features_lookup_key[0]: random_id}]},
    )
    end_time = time.time()
    latency = end_time - start_time  # Calculate latency for this request
    return response.status_code, latency


# Measure total execution time
total_start_time = time.time()
latencies = []

# Send requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")