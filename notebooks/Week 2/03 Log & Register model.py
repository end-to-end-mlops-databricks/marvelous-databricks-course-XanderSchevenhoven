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

# COMMAND ----------

# load configs
config = ProjectConfig.from_yaml("../project_config.yml")
print(config)

# define extra configs
full_schema_name = f"{config.catalog_name}.{config.schema_name}"
numeric_features_clean = [to_snake(column) for column in config.numeric_features]
target_clean = to_snake(config.target)

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
# create numeric features transformer
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler())
    ]
)

# combine preprocessing steps into single transformer
preprocessor = ColumnTransformer(
    transformers=[("numeric", numeric_transformer, numeric_features_clean)]
)

# create regressor
regressor = RandomForestRegressor(
    n_estimators=config.parameters['n_estimators'],
    max_depth=config.parameters['max_depth'],
    random_state=42,
)

# create sklearn pipeline with preprocssor and regressor
model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
model

# COMMAND ----------
# create experiment
mlflow.set_experiment(experiment_name=config.experiment_name)
git_sha = "ffa63b430205ff7"

# COMMAND ----------
# start training run with mlflow
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}",
          "branch": "week2"},
) as run:
    run_id = run.info.run_id

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "Random Forest with preprocessing")
    mlflow.log_params(config.parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # log input dataset
    dataset = mlflow.data.from_spark(
        train_set_spark, 
        table_name=f"{full_schema_name}.train_set",
        version="0"
    )
    mlflow.log_input(dataset, context="training")
    
    # log model
    model_name = config.model_artifact_name
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_name,
        signature=signature
    )

# COMMAND ----------
# register model
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/{model_name}',
    name=f"{full_schema_name}.{config.model_name}",
    tags={"git_sha": f"{git_sha}"})

# COMMAND ----------
# get dataset info and load source from mlflow run
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()
