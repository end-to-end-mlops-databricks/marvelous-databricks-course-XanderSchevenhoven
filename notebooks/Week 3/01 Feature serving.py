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

import pandas as pd
import mlflow

from power import DataProcessor, ProjectConfig

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

# COMMAND ----------

# Define table names
feature_table_name = f"{catalog_name}.{schema_name}.house_prices_preds"
online_table_name = f"{catalog_name}.{schema_name}.house_prices_preds_online"

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# combine train/test into single dataframe
df = pd.concat([train_set, test_set])

# COMMAND ----------
# Load the MLflow model for predictions
pipeline = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.{model_name}/1")