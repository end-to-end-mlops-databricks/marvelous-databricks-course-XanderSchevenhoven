from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


class PowerModel:

    def __init__(self, preprocessor, config):
        
        # initiate config as part of class
        self.confg = config
        self.preprocessor = preprocessor

        # create regressor
        self.regressor = RandomForestRegressor(
            n_estimators=config['parameters']['n_estimators'],
            max_depth=config['parameters']['max_depth'],
            random_state=42
        )

        # create sklearn pipeline with preprocssor and regressor
        self.model = Pipeline(
            steps = [
                ('preprocessor', self.preprocessor),
                ('regressor', self.regressor)
            ]
        )

    
    def train(self, features_train, target_train):
        self.model.fit(features_train, target_train)


    def predict(self, feautures):
        return self.model.predict(X)


    def evaluate(self, features_test, target_test):
        target_predictions = self.model.predict(X_test)
        mse = mean_squared_error(target_test, target_predictions)
        r2 = r2_score(target_test, target_predictions)
        return mse, r2


    def get_feature_importance(self, features_train, target_train):
        feature_importance = self.model.named_steps['regressor'].feature_importances_
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        return feature_importance, feature_names

        