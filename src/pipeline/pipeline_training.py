import re
import pandas as pd
import numpy as np
from typing import Literal

from src.logger import logging
from src.exception import CustomException

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PowerTransformer,
    PolynomialFeatures,
)
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# from sklearn.decomposition import PCA
# from optuna import


class Save_DataFrame(BaseEstimator, TransformerMixin):
    """Save Dataframe generated during preprocessing and feature engineeering phase to .pkl for analysis

    Args:
        BaseEstimator (object): For pipeline compatibility
        TransformerMixin (object): For pipeline compatibility
    """

    def fit(self, X, y=None):
        try:
            self.column_names = list(X.columns)
            return self

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def transform(self, X, y=None):
        try:
            if X.shape[0] == 1:
                if X.shape[1] == 17:
                    self.save_path = "artifacts/02_DataFrames/Prediction/df_pred_pp.pkl"
                else:
                    self.save_path = "artifacts/02_DataFrames/Prediction/df_pred_fs.pkl"

            elif X.shape[0] > 180:
                if X.shape[1] == 17:
                    self.save_path = "artifacts/02_DataFrames/Training/df_train_pp.pkl"
                else:
                    self.save_path = "artifacts/02_DataFrames/Training/df_train_fs.pkl"

            elif X.shape[0] < 180:
                if X.shape[1] == 17:
                    self.save_path = "artifacts/02_DataFrames/Training/df_test_pp.pkl"
                else:
                    self.save_path = "artifacts/02_DataFrames/Training/df_test_fs.pkl"

            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.column_names)
            if self.save_path:
                X.to_pickle(self.save_path)
                logging.info(f"DataFrame saved to {self.save_path}")
            return X

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.column_names


class MultiModelEstimator(BaseEstimator, TransformerMixin):
    """Pipeline compatible method to train multiple models

    Args:
        BaseEstimator (object): For pipeline compatibility
        TransformerMixin (object): For pipeline compatibility
    """

    def __init__(
        self,
        models: dict[dict],
        param_grids: dict[dict],
        cv=3,
        scoring: str = None,
        Method: Literal[
            "GridSearchCV", "RandomizedSearchCV", "Optuna"
        ] = "GridSearchCV",
    ):
        self.models = models
        self.param_grids = param_grids
        self.cv = cv
        self.scoring = scoring
        self.method = Method
        self.grid_searches = {}

        if not set(models.keys()).issubset(set(param_grids.keys())):
            missing_params = list(set(models.keys()) - set(param_grids.keys()))
            logging.info("They keys in model dict isnt matching that in params dict")
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params
            )

    def fit(self, X, y=None):
        """Iterates over each model looking for the best hyperparametes

        Args:
            X (array/DataFrame): independant feature
            y (array, optional): dependant feature. Kept for consistency. Defaults to None.

        Raises:
            CustomException: Error during hyperparameter tuning

        Returns:
            self: calculates best paramters and stores it for use with predict method
        """
        try:
            for name, model in self.models.items():
                logging.info(f"Fitting {self.method} for {name}")

                if self.method == "GridSearchCV":
                    gs = GridSearchCV(
                        model,
                        self.param_grids[name],
                        cv=self.cv,
                        scoring=self.scoring,
                        refit=True,
                        n_jobs=-1,
                    )
                    gs.fit(X, y)

                elif self.method == "RandomizedSearchCV":
                    gs = RandomizedSearchCV(
                        model,
                        self.param_grids[name],
                        cv=self.cv,
                        scoring=self.scoring,
                        refit=True,
                        n_jobs=-1,
                    )
                    gs.fit(X, y)

                elif self.method == "Optuna":
                    pass

                self.grid_searches[name] = gs
                logging.info(f"Best parameters for {name}: {gs.best_params_}")

            return self

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def predict(self, X):
        """Iterates over each model and predicts the dependant variable

        Args:
            X (array/DataFrame): independant features from the test/valdiation set

        Raises:
            CustomException: Error during prediction

        Returns:
            Tuple[DataFrame,dict(Models)]: DataFrame with y_pred from each model and a dictionary of models
        """
        try:
            self.predictions_ = {}
            self.models_ = {}

            for name, grid_search in self.grid_searches.items():
                logging.info(f"Predicting {self.method} for {name}")
                best_model = grid_search.best_estimator_
                # if hasattr(best_model, "predict_proba"):
                #     self.predictions_[name] = best_model.predict_proba(X)
                #     self.models_[name] = best_model
                # else:
                #     self.predictions_[name] = best_model.predict(X)
                #     self.models_[name] = best_model
                self.predictions_[name] = best_model.predict(X)
                self.models_[name] = best_model
            df_y_pred = pd.DataFrame(self.predictions_)
            return df_y_pred, self.models_

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def get_feature_names_out(self, input_features=None):
        return list(self.predictions_.keys())


class PipelineConstructor:
    """Class that creates a pipeline that will do preprocessing and feature engineering and cant be fit with the model training pipeline"""

    def __init__(
        self,
        cols_drop: list[str] = None,
    ):
        self.cols_drop = cols_drop

    def create_new_feats(
        self, data: pd.DataFrame = None, drop: list[str] = None
    ) -> pd.DataFrame:
        try:
            df = data.copy()
            # Create new columns LastName & Title
            for name in df["Name"]:
                df.loc[df["Name"] == name, "LastName"] = (
                    re.search(r"^([^,]+),", name).group(1).strip()
                )
                df.loc[df["Name"] == name, "Title"] = (
                    re.search(r",\s*([^\.]+)\.", name).group(1).strip()
                )

            # Create new column FamilySize based on LastName
            df["FamilySize"] = df.groupby("LastName")["LastName"].transform("count")
            # df["FamilySize"] = df["SibSp"] + df["Parch"]

            # Create new column AgeGroup by binning
            df["AgeGroup"] = pd.cut(
                x=df["Age"],
                bins=list(range(0, 100, 10)),
                labels=["{0}-{1}".format(i, i + 9) for i in range(0, 90, 10)],
                right=False,
                include_lowest=True,
            ).astype(str)

            # Ordinal Encode 'AgeGroup' column
            df["AgeGroup"] = (
                df["AgeGroup"]
                .replace(
                    to_replace={
                        "20-29": 1,
                        "30-39": 2,
                        "10-19": 3,
                        "0-9": 4,
                        "40-49": 5,
                        "50-59": 6,
                        "60-69": 7,
                        "80-89": 8,
                        "70-79": 9,
                    }
                )
                .astype(float)
            )

            # Change Cabin Column data and fill NaN with 'Missing'
            df["Cabin"] = df["Cabin"].str.replace(r"[^A-Za-z]", "", regex=True).str[0]
            df.loc[df["Cabin"] == "T", "Cabin"] = "A"  # Only one instance of T
            df.loc[df["Cabin"] == "G", "Cabin"] = "F"  # Only four instance of G

            # Replace Title Ms. to Miss.
            df.loc[df["Title"] == "Ms", "Title"] = "Miss"

            # Create column ImpPpl based on Title
            imp_ppl = [
                "Dr",
                "Major",
                "Col",
                "Sir",
                "Mlle",
                "Mme",
                "Lady",
                "the Countess",
            ]
            df["ImpPpl"] = np.where(df["Title"].isin(imp_ppl), 50, 1)

            # Drop unnecessary colummns
            if drop:
                df.drop(columns=drop, inplace=True)

            return df

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def CappingOutlier(
        self,
        data,
        columns: list[str] = None,
        threshold: float = 3,
        method: Literal["z_score", "iqr"] = "z_score",
    ):
        try:
            # Temporary workaraound. The dropped columns sent here through self.vols_numr is causing KeyError
            columns = [col for col in data.columns if data[col].dtypes != "O"]
            df = data.copy()
            for col in columns:
                if method == "z_score":
                    mean = df[col].mean()
                    std = df[col].std()
                    upper_bound = mean + threshold * std
                    lower_bound = mean - threshold * std
                elif method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    upper_bound = Q3 + threshold * IQR
                    lower_bound = Q1 - threshold * IQR

                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            return df

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def create_pipeline(self):
        """Creates a preprocessing + feature engineering pipeline.
        Method is already defined.

        Raises:
            CustomException: Error in pipeline creation

        Returns:
            object: pipeline object to be used with model training
        """
        try:
            ppln = Pipeline(
                steps=[
                    # Impute
                    (
                        "Imputer",
                        ColumnTransformer(
                            transformers=[
                                ("knn", KNNImputer(n_neighbors=5), ["Age"]),
                                (
                                    "sml-1",
                                    SimpleImputer(strategy="most_frequent"),
                                    ["Embarked"],
                                ),
                                (
                                    "sml-2",
                                    SimpleImputer(
                                        strategy="constant", fill_value="Missing"
                                    ),
                                    ["Cabin"],
                                ),
                            ],
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                        ),
                    ),
                    # New Features
                    (
                        "New_Feats",
                        FunctionTransformer(
                            func=self.create_new_feats, kw_args={"drop": self.cols_drop}
                        ),
                    ),
                    # Encode and Standardise
                    (
                        "ColumnOperations",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "Encode",
                                    OneHotEncoder(sparse_output=False, drop="first"),
                                    ["Sex", "Cabin", "Embarked"],
                                ),
                                ("Standardise", PowerTransformer(), ["Age", "Fare"]),
                            ],
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                        ),
                    ),
                    # Save DataFrame
                    ("Save_DF_preprocessing", Save_DataFrame()),
                    # Feature Engineering
                    ("PolyFeat", PolynomialFeatures(degree=2, include_bias=False)),
                    ("Selection", SelectKBest(score_func=f_regression, k=20)),
                    # ('PCA',PCA(n_components=20)),
                    ("Save_DF_feat_slcn", Save_DataFrame()),
                ]
            ).set_output(transform="pandas")

            return ppln

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
