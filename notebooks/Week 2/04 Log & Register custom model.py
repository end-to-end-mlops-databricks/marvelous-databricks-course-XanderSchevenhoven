# Databricks notebook source
# MAGIC %md
# MAGIC # Excercise week 2 - preprocessing using python package
# MAGIC Steps:
# MAGIC 1. Load previously trained model
# MAGIC 2. Wrap into custom pyfunc model & log into MLFlow
# MAGIC 3. Register custom pyfunc model

# COMMAND ----------
# install package required to run if running in Databricks
get_ipython().run_line_magic("pip", 'install -e "../.."')

# COMMAND ----------
import json

import mlflow
import pandas as pd
from databricks.sdk.runtime import spark
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

from power import ProjectConfig, adjust_predictions, to_snake

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # It must be -uc for registering models to Unity Catalog
client = MlflowClient()

# COMMAND ----------
# load configs
config = ProjectConfig.from_yaml("../project_config.yml")
print(config)

# define extra configs
full_schema_name = f"{config.catalog_name}.{config.schema_name}"
numeric_features_clean = [to_snake(column) for column in config.numeric_features]
target_clean = to_snake(config.target)

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=[config.experiment_name],
    filter_string="tags.branch='week2'",
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/{config.model_artifact_name}")

# COMMAND ----------
# create custom model wrapper


class PowerConsumptionModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": adjust_predictions(predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------
# load training and testing sets
train_set_spark = spark.table(f"{full_schema_name}.train_set")
train_set = spark.table(f"{full_schema_name}.train_set").toPandas()
test_set = spark.table(f"{full_schema_name}.test_set").toPandas()

X_train = train_set[numeric_features_clean]
y_train = train_set[target_clean]

X_test = test_set[numeric_features_clean]
y_test = test_set[target_clean]

# COMMAND ----------
wrapped_model = PowerConsumptionModelWrapper(model)  # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------
# this is a trick with custom packages
# https://docs.databricks.com/en/machine-learning/model-serving/private-libraries-model-serving.html
# but does not work with pyspark, so we have a better option :-)

mlflow.set_experiment(experiment_name=config.experiment_name)
git_sha = "ffa63b430205ff7"

with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    # get run id of experiment run
    run_id = run.info.run_id

    # create input model signature & log input dataset
    signature = infer_signature(model_input=X_train, model_output={"Prediction": example_prediction})
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{full_schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    # add .whl dependency
    custom_package_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/package/.internal/mlops_with_databricks-0.0.1-py3-none-any.whl"
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[custom_package_path],
        additional_conda_channels=None,
    )

    # log model with .whl dependency
    pyfunc_model_artifact_name = f"pyfunc-{config.model_artifact_name}"
    mlflow.pyfunc.log_model(
        python_model=wrapped_model, artifact_path=pyfunc_model_artifact_name, conda_env=conda_env, signature=signature
    )

# COMMAND ----------
# load pyfunc version of model and unwrap into original state
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{pyfunc_model_artifact_name}")
loaded_model.unwrap_python_model()

# COMMAND ----------
# register pyfunc model
pyfunc_model_name = f"{config.model_name}_pyfunc"
full_pyfunc_model_name = f"{full_schema_name}.{pyfunc_model_name}"

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/{pyfunc_model_artifact_name}",
    name=full_pyfunc_model_name,
    tags={"git_sha": f"{git_sha}"},
)
# COMMAND ----------
# show model version?
with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------
# set model alias to already registered model
model_version_alias = "latest_model"
client.set_registered_model_alias(full_pyfunc_model_name, model_version_alias, "3")

# load registered model under alias
model_uri = f"models:/{full_pyfunc_model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# get model version from alias
client.get_model_version_by_alias(full_pyfunc_model_name, model_version_alias)

# COMMAND ----------

# display model
model  # noqa
