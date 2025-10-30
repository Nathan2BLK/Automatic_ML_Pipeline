# lead_time_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator, clone, ClassifierMixin
from custom_transformers import (
    JointFeatureSelector, FeatureAligner, ArrayToDataFrame, NamedVarianceThreshold,
    build_preprocessor, NaNChecker  # utility function to build column transformer
)
from collections import defaultdict
import warnings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

def classify_categorical_columns(df, cat_cols, threshold_low=50, threshold_high=200,
                                multilabel_sep="#", topk_threshold=100, rare_cutoff=10):
    
    """
    Classifies categorical columns into transformation groups for ML pipelines.

    This function analyzes each categorical column and assigns it to the most appropriate 
    encoding strategy based on:

    - Cardinality (number of unique values)
    - Whether the column is a multi-label field (e.g., "A#B#C")
    - Average length of strings
    - Frequency distribution of multi-label tags

    Parameters:
    - df (pd.DataFrame): The input data
    - cat_cols (List[str]): List of categorical column names
    - threshold_low (int): Max number of unique values for OneHotEncoding
    - threshold_high (int): Max number of unique values for TopK or frequency encoding
    - multilabel_sep (str): Separator for multi-label fields
    - topk_threshold (int): Max number of unique tags to allow full binarization for multilabel
    - rare_cutoff (int): Minimum frequency to avoid tagging as "rare"

    Returns:
    - col_groups (dict): Groups of single-label categorical columns per encoding strategy
    - ml_col_groups (dict): Groups of multi-label categorical columns per encoding strategy

    -------------------------------
    Example Transformations:

    1. One-hot encoding (few unique values):
        status = ["Open", "Closed", "Pending"]
        ➡️ onehot_cols → status_Open, status_Closed, status_Pending

    2. Top-K encoding (medium unique values):
        country = ["FR", "US", "CN", ...] (100+ values)
        ➡️ topk_cols → keep most frequent K (e.g. 50), others → "Other"

    3. Frequency encoding (high unique short):
        project_id = ["PRJ001", "PRJ002", ..., "PRJ999"]
        ➡️ high_card_cols → values replaced by frequency of occurrence

    4. Tfidf encoding (high unique long strings):
        description = ["This project is about...", ...]
        ➡️ text_cols → converted to vector representation using TF-IDF

    5. Multi-label full binarize:
        tags = ["ML#NLP", "CV#DL"]
        ➡️ ml_full_binarize_cols → tags_ML, tags_NLP, tags_CV, tags_DL

    6. Multi-label top-K flag:
        skills = ["Python#Pandas#Java"]
        ➡️ ml_topk_flag_cols → only top-K most frequent tags become binary flags

    7. Multi-label count:
        tools = ["Docker#K8s#Jenkins"]
        ➡️ ml_count_cols → replaced with integer count of tags per row
    -------------------------------
    """

    col_groups = defaultdict(list)
    ml_col_groups = defaultdict(list)

    for col in cat_cols:
        values = df[col].dropna().astype(str)
        is_multilabel = values.str.contains(multilabel_sep).mean() > 0.5
        n_unique = values.nunique()

        if is_multilabel:
            parts = values.str.split(multilabel_sep)
            flat_parts = [tag.strip() for sublist in parts for tag in sublist if tag.strip()]
            unique_counts = pd.Series(flat_parts).value_counts()
            vocab_size = unique_counts.shape[0]
            n_rare = (unique_counts < rare_cutoff).sum()
            rare_ratio = n_rare / vocab_size if vocab_size > 0 else 0

            if vocab_size <= topk_threshold:
                ml_col_groups["ml_full_binarize_cols"].append(col)
            elif rare_ratio < 0.3:
                ml_col_groups["ml_topk_flag_cols"].append(col)
            
            ml_col_groups["ml_count_cols"].append(col)

            col_groups["multilabel_cols"].append(col)

        elif n_unique <= threshold_low:
            col_groups["onehot_cols"].append(col)
        elif threshold_low < n_unique <= threshold_high:
            col_groups["topk_cols"].append(col)
        elif n_unique > threshold_high:
            avg_len = values.str.len().mean()
            if avg_len > 30:
                col_groups["text_cols"].append(col)
            else:
                col_groups["high_card_cols"].append(col)

    return dict(col_groups), dict(ml_col_groups)

def detect_date_columns(df, sample_size=10, threshold=0.8):
    date_cols = []
    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).sample(min(sample_size, len(df)), random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= threshold:
                date_cols.append(col)
    return date_cols

def detect_column_types(df):
    date_cols = detect_date_columns(df)
    
    num_cols = df.select_dtypes(include=["number"]).columns.difference(date_cols).tolist()
    
    # Only keep non-date object-like columns
    potential_cat = df.select_dtypes(include=["object", "category", "bool"]).columns
    cat_cols = [col for col in potential_cat if col not in date_cols]
    
    return cat_cols, date_cols, num_cols

class LeadTimePipeline(BaseEstimator):
    def __init__(self, k_features:int, rf_params=None, model_type="classification", X_ref=None):
        self.k_features = k_features
        self.rf_params = rf_params or {
            "n_estimators": 300,
            "max_depth": 30,
            "min_samples_split": 8,
            "min_samples_leaf": 2,
            "random_state": 42,
            "class_weight": "balanced"
        }
        self.pipeline_ = None
        self.expected_features_ = None
        self.preprocessor_ = None
        self.model_type = model_type
        self.X_ref = X_ref

    def fit(self, X, y):
        if self.X_ref is None:
            self.X_ref = X
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")

        # 🔍 Detect column types and group them
        cat_cols, date_cols, num_cols = detect_column_types(X)
        col_groups, ml_col_groups = classify_categorical_columns(X, cat_cols)

        # 🏗 Build the preprocessor using grouped columns
        self.preprocessor_ = build_preprocessor(
            col_groups=col_groups,
            ml_col_groups=ml_col_groups,
            date_cols=date_cols,
            num_cols=num_cols
        )

        # name provider evaluated after preprocessor.fit()
        to_df = ArrayToDataFrame(source=self.preprocessor_)
        varth = NamedVarianceThreshold(threshold=0.0)


        selector = JointFeatureSelector(
            X_ref=self.X_ref,
            k=self.k_features,
            preprocessor=self.preprocessor_  # still fine to pass; not used for names anymore
        )

        if self.model_type == "regression":
            self.model = RandomForestRegressor(**self.rf_params)
        elif self.model_type == "classification":
            rf = RandomForestClassifier(**self.rf_params)
            self.model = CalibratedClassifierCV(rf, method='sigmoid', cv=5)

        self.pipeline_ = Pipeline([
            ("preprocessor", self.preprocessor_),
            ("toDF", to_df),                # -> DataFrame with names
            ("nanChecker", NaNChecker()),
            ("varth", varth),               # -> drop zero-variance, keep names
            ("selector", selector),         # -> reads X.columns (post-drop)
            ("classifier", self.model),
        ])

        self.pipeline_.fit(X, y)

        # Save expected features for prod usage
        self.expected_features_ = self.pipeline_.named_steps["selector"].get_feature_names_out()
        return self
    
    # in LeadTimePipeline
    def transform_for_classifier(self, X, *, as_dataframe: bool = True):
        if self.pipeline_ is None:
            raise ValueError("Model not trained. Call fit first.")

        pre   = self.pipeline_.named_steps["preprocessor"]
        toDF  = self.pipeline_.named_steps["toDF"]
        varth = self.pipeline_.named_steps["varth"]
        sel   = self.pipeline_.named_steps["selector"]

        Xt    = pre.transform(X)
        Xt_df = toDF.transform(Xt)             # DataFrame with names
        Xt_v  = varth.transform(Xt_df)         # DataFrame after variance filter

        cols  = list(sel.get_feature_names_out())
        # Reindex to selected columns (fill missing with 0 for safety)
        if not isinstance(Xt_v, pd.DataFrame):
            # fallback if your varth returned ndarray (shouldn’t with NamedVarianceThreshold)
            vt_names = varth.get_feature_names_out(toDF.get_feature_names_out())
            Xt_v = pd.DataFrame(Xt_v, columns=vt_names)

        out = Xt_v.reindex(columns=cols, fill_value=0.0)
        return out if as_dataframe else out.to_numpy()

    def predict(self, X):
        return self.pipeline_.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline_.predict_proba(X)

    def cross_validate(self, X, y, cv=5):
        return cross_validate(self.pipeline_, X, y, cv=cv, scoring=["accuracy", "f1_weighted"], return_train_score=True)

    def save(self, path):
        joblib.dump({
            "pipeline": self.pipeline_,
            "expected_features": self.expected_features_,
            "preprocessor": self.preprocessor_
        }, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls.__new__(cls)            # bypass __init__
        # restore attrs
        obj.pipeline_ = data["pipeline"]
        obj.preprocessor_ = data.get("preprocessor")
        obj.expected_features_ = data.get("expected_features")
        obj.model_type = data.get("model_type", "classification")
        obj.k_features = data.get("k_features", 140)
        obj.rf_params = data.get("rf_params", None)
        obj.selector_ = None              # selector lives inside pipeline
        return obj
