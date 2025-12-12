from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np


def get_supervised_models(task_type):
    """Return available models based on task type"""
    if task_type == "Regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "XGBoost": XGBRegressor(n_estimator = 300, learning_rate = 0.1, max_depth = 5)
            }
    else:
     return {
         "Logistic Regression": LogisticRegression(),
         "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
         "XGBoost": XGBClassifier(n_estimator = 300, learning_rate = 0.1, max_depth = 5)
         }



def train_supervised(model, X_train, y_train, X_test):
    """Train supervised model and return predictions"""
    model.fit(X_train, y_train)
    return model.predict(X_test)

def train_multiple_models(models, X_train, y_train, X_test, y_test, task_type):
    """
    Train all models in the dictionary and return predictions + performance scores.
    - For regression → return MSE and RMSE
    - For classification → return Accuracy_score
    """
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            results[name] = {
                "model": model,
                "mse": mse,
                "rmse": rmse,
                "y_pred": y_pred,
            }
        else:
            acc = accuracy_score(y_test, y_pred)
            results[name] = {
                "model": model,
                "accuracy": acc,
                "y_pred": y_pred,
            }

    return results


def get_unsupervised_models():
    """Return unsupervised learning models (by name)"""
    return {
        "KMeans": KMeans,
        "DBSCAN": DBSCAN,
    }

def train_unsupervised(model_name, X_scaled, params):
    """
    Train unsupervised clustering model and return fitted model + labels.
    Parameters depend on model type:
        {"n_clusters": 3}  or  {"eps": 0.5, "min_samples": 5}
    """
    if model_name == "KMeans":
        cluster_model = KMeans(
            n_clusters=params.get("n_clusters", 3),
            random_state=42,
            n_init="auto"
        )
    elif model_name == "DBSCAN":
        cluster_model = DBSCAN(
            eps=params.get("eps", 0.5),
            min_samples=params.get("min_samples", 5)
        )
    else:
        raise ValueError(f"Unknown unsupervised model: {model_name}")

    labels = cluster_model.fit_predict(X_scaled)
    return cluster_model, labels




