from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        self.dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())])
        self.time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        self.preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])], remainder="drop")
        return self.preproc_pipe

    def run(self, X_train, y_train):
        """train the pipeline"""
        self.pipe = Pipeline([
        ('preproc', preproc_pipe),
        ('linear_model', LinearRegression())])
        return self.pipe.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        self.y_pred = pipe.predict(X_test)
        self.RMSE = np.sqrt(((y_pred - y_test)**2).mean())
        return self.RSME

if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
