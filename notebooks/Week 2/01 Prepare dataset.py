# Databricks notebook source
# MAGIC %md
# MAGIC # Excercise week 2 - preprocessing using python package
# MAGIC Steps:
# MAGIC 1. Load / open file
# MAGIC 2. Preprocess data
# MAGIC 3. Save to unity catalog

# COMMAND ----------
# MAGIC %md ## I. Setup notebook

# COMMAND ----------
# MAGIC %md ##### 1. Install / import libraries

# COMMAND ----------
# install package required to run if running in Databricks
get_ipython().run_line_magic("pip", 'install -e "../.."')

# COMMAND ----------

# %reload_ext autoreload
# %autoreload 2

# COMMAND ----------


from power import DataProcessor, ProjectConfig

# COMMAND ----------
# MAGIC %md
# MAGIC ##### 2. Get configs

# COMMAND ----------

config = ProjectConfig.from_yaml("../project_config.yml")

print("Configuration loaded:")
print(config)

# COMMAND ----------
# MAGIC %md
# MAGIC ## II. Apply preprocessor and model

# COMMAND ----------

processor = DataProcessor(config=config)
processor.prepare_data()

# COMMAND ----------

pdf_train_set, pdf_test_set = processor.split_data()
processor.save_to_catalog(pdf_train_set, pdf_test_set)
