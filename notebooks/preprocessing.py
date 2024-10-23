# Databricks notebook source
# MAGIC %md
# MAGIC # Excercise week 1 - preprocessing using python package
# MAGIC Steps:
# MAGIC 1. Load / open file
# MAGIC 2. Preprocess data
# MAGIC 3. Train simple model
# MAGIC 4. Evaluate model

# COMMAND ----------
# MAGIC %md ## I. Setup notebook

# COMMAND ----------
# MAGIC %md ##### 1. Install / import libraries

# COMMAND ----------
%pip install ".."

# COMMAND ----------

from power import DataProcessor, PowerModel
import yaml

# COMMAND ----------
# MAGIC %md ##### 2. Get configs

# COMMAND ----------

with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# MAGIC %md ## II. Apply preprocessor and model

