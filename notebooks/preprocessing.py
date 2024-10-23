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
# MAGIC %md ## 1. Install / import libraries

# COMMAND ----------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# COMMAND ----------