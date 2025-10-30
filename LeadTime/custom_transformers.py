import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import f_classif,mutual_info_classif,VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings

class DatePartExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, features=["year", "month"]):
        self.features = features
        self.output_columns_ = []

    def fit(self, X, y=None):
        self.output_columns_ = []
        for col in X.columns:
            if "year" in self.features:
                self.output_columns_.append(col + "_year")
            if "month" in self.features:
                self.output_columns_.append(col + "_month")
        return self

    def transform(self, X):
        X_out = pd.DataFrame()
        for col in X.columns:
            dates = pd.to_datetime(X[col], errors="coerce")
            if "year" in self.features:
                X_out[col + "_year"] = dates.dt.year
            if "month" in self.features:
                X_out[col + "_month"] = dates.dt.month
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns_)

class MultiLabelCount(BaseEstimator, TransformerMixin):
    def __init__(self, sep="#"): self.sep = sep
    def fit(self, X, y=None):
        self.output_columns_ = X.columns.tolist()
        return self
    def transform(self, X):
        return X.fillna("").apply(lambda col: col.str.split(self.sep).apply(lambda x: len([i for i in x if i.strip()]))).astype(int)
    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns_)

class TopKEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k
        self.top_k_ = {}
        self.output_columns_ = []

    def fit(self, X, y=None):
        self.top_k_.clear()
        self.output_columns_.clear()
        for col in X.columns:
            top_k = X[col].value_counts().nlargest(self.k).index.tolist()
            self.top_k_[col] = top_k
            # expected columns = all top-K + "Other"
            self.output_columns_.extend([f"{col}_{val}" for val in top_k] + [f"{col}_Other"])
        return self

    def transform(self, X):
        blocks = []
        for col in X.columns:
            top_k = self.top_k_[col]
            ser = X[col].astype(object).fillna("Other")
            ser = ser.where(ser.isin(top_k), other="Other")

            # build dummies, then reindex to expected columns in the same order
            dummies = pd.get_dummies(ser, prefix=col, dtype=int)  # only present cats
            expected = [f"{col}_{val}" for val in top_k] + [f"{col}_Other"]
            dummies = dummies.reindex(columns=expected, fill_value=0)

            blocks.append(dummies)

        X_out = pd.concat(blocks, axis=1)
        return X_out.values  # ColumnTransformer will stack with others

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns_)

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.freq_maps_ = {}
        self.output_columns_ = X.columns.tolist()
        for col in X.columns:
            self.freq_maps_[col] = X[col].value_counts(normalize=True)
        return self
    def transform(self, X):
        return X.apply(lambda col: col.map(self.freq_maps_[col.name]).fillna(0))
    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns_)

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col): self.col = col
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.col].fillna("")
    def get_feature_names_out(self, input_features=None):
        return np.array([self.col])
    
class MultiLabelBinarizerPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, sep="#"):
        self.sep = sep
        self.binarizers = {}
        self.output_columns_ = []

    def fit(self, X, y=None):
        self.output_columns_ = []
        sep = self.sep

        if isinstance(X, pd.Series):
            X = X.to_frame()

        for col in X.columns:
            parts = X[col].fillna("").astype(str).apply(lambda x: [i.strip() for i in x.split(sep) if i.strip()])
            mlb = MultiLabelBinarizer()
            mlb.fit(parts)
            self.binarizers[col] = mlb
            self.output_columns_.extend([f"{col}_{cls}" for cls in mlb.classes_])
        return self

    def transform(self, X):
        import warnings
        if isinstance(X, pd.Series):
            X = X.to_frame()

        binarized = []
        for col in X.columns:
            # normalize tokens exactly like in fit
            parts = (
                X[col]
                .fillna("")
                .astype(str)
                .apply(lambda s: [t.strip() for t in s.split(self.sep) if t.strip()])
            )

            mlb   = self.binarizers[col]
            known = set(mlb.classes_)

            # drop unknown tokens -> no “unknown class(es)” warning
            parts_filtered = parts.apply(lambda lst: [t for t in lst if t in known])

            # (A) safest: silence only this sklearn warning just in case
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"unknown class\(es\).*\bwill be ignored\b",
                    category=UserWarning,
                    module=r"sklearn\.preprocessing\._label",
                )
                arr = mlb.transform(parts_filtered)

            binarized.append(arr)

        return np.hstack(binarized)

    def get_feature_names_out(self, input_features=None):
            return np.array(self.output_columns_)
    
class TopKFlagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, sep="#"): self.k = k; self.sep = sep; self.top_tags_ = {}; self.output_columns_ = []
    def fit(self, X, y=None):
        for col in X.columns:
            parts = X[col].fillna("").astype(str).str.split(self.sep)
            all_tags = [tag.strip() for lst in parts for tag in lst if tag.strip()]
            top_k_tags = pd.Series(all_tags).value_counts().nlargest(self.k).index.tolist()
            self.top_tags_[col] = top_k_tags
            self.output_columns_.extend([f"{col}_{tag}" for tag in top_k_tags])
        return self
    def transform(self, X):
        binarized = []
        for col in X.columns:
            top_tags = self.top_tags_[col]
            flags = pd.DataFrame(0, index=X.index, columns=[f"{col}_{tag}" for tag in top_tags])
            for i, row in X[col].fillna("").astype(str).str.split(self.sep).items():
                tags = [t.strip() for t in row if t.strip()]
                for tag in tags:
                    if tag in top_tags:
                        flags.at[i, f"{col}_{tag}"] = 1
            binarized.append(flags)
        return pd.concat(binarized, axis=1).values
    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns_)
    
class CounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sep="#"):
        self.sep = sep
        self.output_columns_ = []

    def fit(self, X, y=None):
        self.output_columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        X_filled = X.fillna("")
        return X_filled.apply(
            lambda col: col.str.split(self.sep).apply(
                lambda items: len([i for i in items if i.strip()])
            )
        ).astype(int)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns_)
    
class NaNChecker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        nan_cols = df.columns[df.isnull().any()].tolist()
        if nan_cols:
            print(f"NaN detected before preprocessing: {nan_cols}")
        return X
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else None

class DualFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=97, X_new_sample=None, score_func=f_classif):
        self.k = k
        self.X_new_sample = X_new_sample
        self.score_func = score_func

    def fit(self, X, y):
        def combine_feature_scores(X, y, discrete_features='auto', weight_f=0.5, weight_mi=0.5):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            f_scores, _ = f_classif(X, y) #linear eval
            mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features) #non-linear eval

            scaler = MinMaxScaler()
            f_norm = scaler.fit_transform(f_scores.reshape(-1, 1)).flatten()
            mi_norm = scaler.fit_transform(mi_scores.reshape(-1, 1)).flatten()

            return pd.Series(weight_f * f_norm + weight_mi * mi_norm, index=X.columns).sort_values(ascending=False)

        # Step 1: Ensure X is a DataFrame
        X_train_proc = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Step 2: Handle new sample
        X_new_proc = None
        if self.X_new_sample is not None:
            X_new_proc = pd.DataFrame(self.X_new_sample) if not isinstance(self.X_new_sample, pd.DataFrame) else self.X_new_sample

        # Step 3: Align on common columns if possible
        if X_new_proc is not None:
            common_cols = list(set(X_train_proc.columns) & set(X_new_proc.columns))
            X_train_aligned = X_train_proc[common_cols]
            X_new_aligned = X_new_proc[common_cols]
        else:
            common_cols = list(X_train_proc.columns)
            X_train_aligned = X_train_proc[common_cols]
            X_new_aligned = None

        # Step 4: Compute training feature scores
        mi_train = combine_feature_scores(X_train_aligned, y)

        # Step 5: New data filters (variance + non-null presence)
        if X_new_aligned is not None:
            variance_mask = X_new_aligned.var() > 0.001
            min_presence_ratio = 0.10
            min_count = int(len(X_new_aligned) * min_presence_ratio)
            presence_mask = ((X_new_aligned.notna()) & (X_new_aligned != 0)).sum(axis=0) > min_count
            combined_mask = variance_mask & presence_mask
            filtered_scores = mi_train * combined_mask
        else:
            filtered_scores = mi_train

        # Step 6: Store selected top-k features
        self.selected_columns_ = filtered_scores.sort_values(ascending=False).head(self.k).index.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X[self.selected_columns_]  # Only keep selected aligned columns
        return X

    def get_support(self):
        return self.selected_columns_


class FeatureAligner(BaseEstimator, TransformerMixin):
    def __init__(self, expected_features=None):
        if expected_features is None:
            raise ValueError("You must provide expected_features.")
        self.expected_features = expected_features

    def fit(self, X, y=None):
        # No fitting needed; just return self
        return self

    def transform(self, X):
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Add missing columns
        for col in self.expected_features:
            if col not in X.columns:
                X[col] = 0

        # Drop extra columns and reorder
        return X[self.expected_features]

    def get_feature_names_out(self):
        return self.expected_features

class ArrayToDataFrame(BaseEstimator, TransformerMixin):
    """
    Wrap previous step’s array into a DataFrame with proper column names.
    Use either:
      - source=<transformer with get_feature_names_out()>, or
      - names=<list/array of column names>
    """
    def __init__(self, source=None, names=None):
        self.source = source
        self.names = None if names is None else np.asarray(names, dtype=object)
        self.columns_ = None

    def fit(self, X, y=None):
        if self.names is not None:
            self.columns_ = self.names
        elif self.source is not None:
            # source must be already-fitted when Pipeline calls our fit() at runtime
            cols = self.source.get_feature_names_out()
            self.columns_ = np.asarray(cols, dtype=object)
        else:
            raise ValueError("ArrayToDataFrame requires 'source' or explicit 'names'.")
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):  # sparse -> dense
            X = X.toarray()
        return pd.DataFrame(X, columns=self.columns_)

    def get_feature_names_out(self, input_features=None):
        return self.columns_

class NamedVarianceThreshold(VarianceThreshold):
    """VarianceThreshold that preserves and returns column names."""
    def fit(self, X, y=None):
        self._was_df_ = isinstance(X, pd.DataFrame)
        self._in_cols_ = X.columns.to_numpy() if self._was_df_ else None
        X_arr = X.values if self._was_df_ else X
        super().fit(X_arr, y)
        return self

    def transform(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_out = super().transform(X_arr)
        cols = self.get_feature_names_out(
            X.columns.to_numpy() if isinstance(X, pd.DataFrame) else self._in_cols_
        )
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_out, index=X.index, columns=cols)
        return X_out

    def get_feature_names_out(self, input_features=None):
        mask = self.get_support()
        feats = np.asarray(input_features if input_features is not None else self._in_cols_)
        return feats[mask]
    
class JointFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select top-k features using a blend of ANOVA F-score and Mutual Information
    computed on TRAIN (post-preprocessing & post-variance-filter) and lightly
    regularized by a reference sample (X_new_sample) compatibility signal.

    Expectation: X entering fit/transform is a DataFrame with *final* column
    names after steps like toDF + NamedVarianceThreshold.
    """
    def __init__(self, X_ref, k=100, preprocessor=None,
                weight_f=0.5, weight_mi=0.5, ref_weight=0.15,
                min_presence=0.05):
        self.X_ref = X_ref
        self.k = k
        self.preprocessor = preprocessor
        self.weight_f = float(weight_f)
        self.weight_mi = float(weight_mi)
        self.ref_weight = float(ref_weight)
        self.min_presence = float(min_presence)

        self.selected_features_ = None
        self._train_cols_ = None  # columns seen at fit (post-drop)

    def _ensure_df(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        if isinstance(X, np.ndarray):
            raise ValueError(
                "JointFeatureSelector expects a DataFrame input. "
                "Add ArrayToDataFrame (and NamedVarianceThreshold if used) before this step."
            )
        return X

    def fit(self, X, y):
        X_df = self._ensure_df(X).copy()
        self._train_cols_ = list(X_df.columns)  # train (post-drop) order

        # --- Encode target (safe for string labels)
        y_arr = pd.Series(y).to_numpy()
        if y_arr.dtype.kind not in "iu":  # not integer-like
            le = LabelEncoder()
            y_arr = le.fit_transform(y_arr)

        # --- Build reference in the *preprocessor* space, then align by name to TRAIN
        X_ref_df = None
        if self.X_ref is not None and self.preprocessor is not None:
            pre_new = clone(self.preprocessor).fit(self.X_ref)
            Xr = pre_new.transform(self.X_ref)
            if hasattr(Xr, "toarray"): Xr = Xr.toarray()
            ref_names = pre_new.get_feature_names_out()
            X_ref_df = pd.DataFrame(Xr, columns=ref_names)
            # align to TRAIN order & intersection
            common = [c for c in self._train_cols_ if c in X_ref_df.columns]
        else:
            # no reference available -> use train columns as common set
            common = list(self._train_cols_)

        X_train_aligned = X_df[common]
        X_ref_aligned   = X_ref_df[common] if X_ref_df is not None else None

        # --- Guard against constants on TRAIN
        X_arr = X_train_aligned.to_numpy()
        var = np.nan_to_num(np.var(X_arr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        nonconst = var > 0.0

        n = X_arr.shape[1]
        f_scores = np.zeros(n, dtype=float)
        mi_scores = np.zeros(n, dtype=float)

        if nonconst.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # silence "constant features" from sklearn
                f_nc, _ = f_classif(X_arr[:, nonconst], y_arr)
                mi_nc   = mutual_info_classif(X_arr[:, nonconst], y_arr, discrete_features=False)
            f_scores[nonconst]  = np.nan_to_num(f_nc, nan=0.0, posinf=0.0, neginf=0.0)
            mi_scores[nonconst] = np.nan_to_num(mi_nc, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Normalize each score into [0,1] robustly
        def _minmax_safe(a):
            a = a.astype(float)
            lo, hi = np.min(a), np.max(a)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return np.zeros_like(a, dtype=float)
            return (a - lo) / (hi - lo)

        f_norm  = _minmax_safe(f_scores)
        mi_norm = _minmax_safe(mi_scores)
        train_sig = self.weight_f * f_norm + self.weight_mi * mi_norm  # [0,1]

        # --- Reference compatibility (optional): coverage & non-zero variance
        if X_ref_aligned is not None and len(X_ref_aligned):
            Xr_arr = X_ref_aligned.to_numpy()
            # coverage: fraction of non-nan AND non-zero entries
            present = (~np.isnan(Xr_arr)) & (Xr_arr != 0)
            cov = present.mean(axis=0)  # [0..1]
            cov = np.nan_to_num(cov, nan=0.0)

            # ref variance > 0 mask
            rv = np.nan_to_num(np.var(Xr_arr, axis=0), nan=0.0) > 0.0

            cov_norm = _minmax_safe(cov)
            ref_sig = cov_norm * rv.astype(float)  # [0..1]
        else:
            ref_sig = np.zeros_like(train_sig, dtype=float)

        # --- Blend
        combined = (1.0 - self.ref_weight) * train_sig + self.ref_weight * ref_sig

        # --- Optional minimum presence filter (on reference)
        if X_ref_aligned is not None and len(X_ref_aligned):
            present_ratio = (~np.isnan(Xr_arr) & (Xr_arr != 0)).mean(axis=0)
            presence_mask = present_ratio >= float(self.min_presence)
            combined = combined * presence_mask.astype(float)

        # --- Select top-k (keep TRAIN order for stability)
        k = int(min(self.k, len(combined)))
        if k <= 0:
            # fallback: keep at least one column (e.g., highest variance)
            keep1 = [X_df.var(numeric_only=True).idxmax()]
            self.selected_features_ = keep1
            return self

        top_idx = np.argsort(combined)[-k:]
        top_idx.sort()
        common = np.asarray(common, dtype=object)
        self.selected_features_ = common[top_idx].tolist()
        return self

    def transform(self, X):
        # Expect a DataFrame thanks to toDF + NamedVarianceThreshold
        if hasattr(X, "toarray"):
            X = X.toarray()
        if isinstance(X, np.ndarray):
            raise ValueError("Selector expects a DataFrame after toDF/varth. Got ndarray.")

        if self.selected_features_ is None:
            raise RuntimeError("JointFeatureSelector not fitted yet (selected_features_ is None).")

        # Fill any missing selected columns with zeros (robust to drift)
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            for m in missing:
                X[m] = 0.0

        # Reindex in the selected order; never returns None
        out_df = X.reindex(columns=self.selected_features_, fill_value=0.0)

        # Final sanity checks
        if out_df is None:
            raise RuntimeError("Internal error: reindex returned None.")
        if out_df.shape[1] == 0:
            # Optional: fallback to at least 1 feature to avoid downstream errors
            # return np.zeros((len(out_df), 1), dtype=float)
            pass

        return out_df.to_numpy()

    # sklearn API helpers
    def get_support(self):
        return np.array([c in set(self.selected_features_) for c in self._train_cols_], dtype=bool)

    def get_feature_names_out(self):
        return np.array(self.selected_features_, dtype=object)
    
def build_preprocessor(col_groups, ml_col_groups, date_cols, num_cols):
    """
    Build and return a complete ColumnTransformer for preprocessing.
    """
    from custom_transformers import (
        DatePartExtractor,
        MultiLabelBinarizerPipeline,
        TopKFlagTransformer,
        FrequencyEncoder,
        TopKEncoder,
        CounterTransformer
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    date_pipeline = Pipeline([
        ("extract_parts", DatePartExtractor(features=["year", "month"])),
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1))
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("onehot", onehot_pipeline, col_groups.get("onehot_cols", [])),
        ("date", date_pipeline, date_cols),
        ("topk", TopKEncoder(k=10), col_groups.get("topk_cols", [])),
        ("high_card", FrequencyEncoder(), col_groups.get("high_card_cols", [])),
        ("ml_count", CounterTransformer(), ml_col_groups.get("ml_count_cols", [])),
        ("ml_binarize", MultiLabelBinarizerPipeline(), ml_col_groups.get("ml_full_binarize_cols", [])),
        ("ml_topk", TopKFlagTransformer(k=10), ml_col_groups.get("ml_topk_flag_cols", [])),
    ], remainder="drop")