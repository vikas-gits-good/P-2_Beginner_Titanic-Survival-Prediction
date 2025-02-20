from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier

models_dict = {
    # Ensemble Models
    "RandomForestClassifier": RandomForestClassifier(n_jobs=-1, random_state=44),
    "GradientBoostingClassifier": GradientBoostingClassifier(
        loss="log_loss", criterion="friedman_mse", random_state=44
    ),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=44),
    "XGBClassifier": XGBClassifier(),
    "XGBRFClassifier": XGBRFClassifier(),
    # # Tree Models
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=44),
    # Neightbour Models
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1),
    # Linear Models
    "LogisticRegression": LogisticRegression(random_state=44, n_jobs=-1),
    "MLPClassifier": MLPClassifier(random_state=44, activation="relu"),
}

params_dict = {
    "RandomForestClassifier": {
        "n_estimators": [700, 800],
        "max_depth": [8, 10],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [2, 3],
        "max_leaf_nodes": [30, 35],
        "min_impurity_decrease": [0.001],
    },
    "GradientBoostingClassifier": {
        "learning_rate": [0.4, 0.5],
        "n_estimators": [250, 300],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [3, 4],
        "max_depth": [5, 6],
        "max_leaf_nodes": [5, 6],
    },
    "AdaBoostClassifier": {
        "n_estimators": [100, 150],
        "learning_rate": [2, 3],
    },
    "XGBClassifier": {
        # "": [],
    },
    "XGBRFClassifier": {
        #     "": [],
    },
    "DecisionTreeClassifier": {
        "max_depth": [8, 9],
        "min_samples_leaf": [13, 14],
        "max_leaf_nodes": [5, 6],
    },
    "KNeighborsClassifier": {"p": [1, 2], "n_neighbors": [4, 5]},
    "LogisticRegression": {
        "C": [2, 3],
    },
    "MLPClassifier": {
        "learning_rate_init": [0.004, 0.005],
    },
}
