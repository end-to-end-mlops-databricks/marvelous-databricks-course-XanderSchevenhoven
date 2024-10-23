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
# install package required to run if running in Databricks
# get_ipython().run_line_magic("pip", 'install ".."')

# COMMAND ----------

import yaml
from power import (
    DataProcessor, 
    PowerModel, 
    visualize_results, 
    plot_feature_importance
)

# COMMAND ----------
# MAGIC %md 
# MAGIC ##### 2. Get configs

# COMMAND ----------

with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# MAGIC %md 
# MAGIC ## II. Apply preprocessor and model

# COMMAND ----------
 
processor = DataProcessor(config = config)
processor.preprocess_data()

# COMMAND ----------

features_train, features_test, target_train, target_test = processor.split_data()

print("Training set shape:", features_train.shape)
print("Training set shape:", features_test.shape)

# COMMAND ----------
# Initialize and train the model
model = PowerModel(processor.preprocessor, config)
model.train(features_train, target_train)

# COMMAND ----------
# Evaluate the model
mse, r2 = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# COMMAND ----------
## Visualizing Results
y_pred = model.predict(X_test)
visualize_results(y_test, y_pred)

# COMMAND ----------
## Feature Importance
feature_importance, feature_names = model.get_feature_importance()
plot_feature_importance(feature_importance, feature_names)
