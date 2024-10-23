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

# COMMAND ----------
# MAGIC %md ##### 2. Get configs

# COMMAND ----------
# MAGIC %md ## II. Apply preprocessor and model

