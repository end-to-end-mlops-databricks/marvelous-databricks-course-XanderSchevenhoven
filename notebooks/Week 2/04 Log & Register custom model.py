# Databricks notebook source
# MAGIC %md
# MAGIC # Excercise week 2 - preprocessing using python package
# MAGIC Steps:
# MAGIC 1. Create model
# MAGIC 2. Log training run to MLFlow Experiment
# MAGIC 3. Register model in Unity Catalog

# COMMAND ----------
# install package required to run if running in Databricks
get_ipython().run_line_magic("pip", 'install -e "../.."')

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from databricks.sdk.runtime import spark

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from power import ProjectConfig, to_snake

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri('databricks-uc') # It must be -uc for registering models to Unity Catalog
