# Databricks notebook source
# MAGIC %md
# MAGIC # Excercise week 2 - preprocessing using python package
# MAGIC Steps:
# MAGIC 1. Create feature table & retrieval function
# MAGIC 2. Train model using feature table & function
# MAGIC 3. Register custom model

# COMMAND ----------
# install package required to run if running in Databricks
get_ipython().run_line_magic("pip", 'install -e "../.."')

# COMMAND ----------

import mlflow
import pyspark.sql.functions as f
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk.runtime import spark
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from power import ProjectConfig, to_snake

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # It must be -uc for registering models to Unity Catalog
client = MlflowClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------
# load configs
config = ProjectConfig.from_yaml("../project_config.yml")
print(config)

# define extra configs
full_schema_name = f"{config.catalog_name}.{config.schema_name}"
numeric_features_clean = [to_snake(column) for column in config.numeric_features]
target_clean = to_snake(config.target)

# feature table & function name
feature_table_name = f"{full_schema_name}.power_features"
function_name = f"{full_schema_name}.convert_celsius_to_fahrenheit"

# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{full_schema_name}.train_set")
test_set = spark.table(f"{full_schema_name}.test_set")

# COMMAND ----------
# Create or replace the house_features table
spark.sql(f"""
    CREATE OR REPLACE TABLE {feature_table_name}
    (
        date_time string NOT NULL,
        temperature double,
        humidity double
    );
""")

# add primary key constraints
spark.sql(f"ALTER TABLE {feature_table_name} " "ADD CONSTRAINT house_pk PRIMARY KEY(date_time);")

# enable change data feed
spark.sql(f"ALTER TABLE {feature_table_name} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# Insert data into the feature table from both train and test sets
spark.sql(
    f"INSERT INTO {feature_table_name} " f"SELECT date_time, temperature, humidity FROM {full_schema_name}.train_set"
)
spark.sql(
    f"INSERT INTO {feature_table_name} " f"SELECT date_time, temperature, humidity FROM {full_schema_name}.test_set"
)

# COMMAND ----------
# Define a function to calculate the house's age using the current year and YearBuilt
spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(temperature INT)
    RETURNS INT
    LANGUAGE PYTHON AS
    $$
    return temperature * (9/5) + 32
    $$
""")

# COMMAND ----------
# Load training and test sets
train_set = (
    spark.table(f"{full_schema_name}.train_set")
    .withColumn("temperature_fahrenheit", f.col("temperature").cast("int"))
    .drop("temperature", "humidity")
)
test_set = spark.table(f"{full_schema_name}.test_set").toPandas()

# setup feature engineering
training_set = fe.create_training_set(
    df=train_set,
    label=target_clean,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["humidity"],
            lookup_key="date_time",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="temperature",
            input_bindings={"temperature": "temperature_fahrenheit"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate temp fahrenheit for test set
test_set["temperature"] = test_set["temperature"] * (9 / 5) + 32

# Split features and target
X_train = training_df[numeric_features_clean]
y_train = training_df[target_clean]
X_test = test_set[numeric_features_clean]
y_test = test_set[target_clean]

# create numeric features transformer
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

# combine preprocessing steps into single transformer
preprocessor = ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features_clean)])

# create regressor
regressor = RandomForestRegressor(
    n_estimators=config.parameters["n_estimators"],
    max_depth=config.parameters["max_depth"],
    random_state=42,
)

# create sklearn pipeline with preprocssor and regressor
model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
model

# COMMAND ----------
# Set and start MLflow experiment
mlflow.set_experiment(experiment_name=config.experiment_name)
git_sha = "ffa63b430205ff7"

with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate and print metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForest with preprocessing")
    mlflow.log_params(config.parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=model,
        flavor=mlflow.sklearn,
        artifact_path=config.model_artifact_name,
        training_set=training_set,
        signature=signature,
    )


mlflow.register_model(
    model_uri=f"runs:/{run_id}/{config.model_artifact_name}", name=f"{full_schema_name}.{config.model_name}_fe"
)
