#!/usr/bin/env python
# run_leadtime.py
#
# One entry point to TRAIN or PREDICT with your LeadTime pipeline.
# - Centralizes important variables (paths, target/drop columns, status filters)
# - Clean argparse for Jenkins (env overrides supported)
# - Properly uses X_new_sample during CV & final fit
# - Writes metrics and model artifacts

import os, sys, json, argparse, warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import subprocess
import pathlib

import numpy as np
import pandas as pd

# ---- import your project modules (assumes this file is run from repo root OR src on sys.path)
from lead_time_model import LeadTimePipeline
# If you kept the pretty Excel exporter in predict_new_data.py we’ll use it when available:
try:
    from exports import export_predictions_to_excel_pretty as export_xlsx
except Exception:
    export_xlsx = None  # we'll fall back to CSV if not importable

SRC = pathlib.Path(__file__).resolve().parents[2]  # -> .../src
sys.path.insert(0, str(SRC))

from common import utils

# --------------------------------------------------------------------------------------
# Optional: silence noisy, known-benign warnings (can be toggled with --quiet-warnings)
WARN_FILTERS = [
    dict(message=r"unknown class\(es\).*\bwill be ignored\b",
        category=UserWarning, module=r"sklearn\.preprocessing\._label"),
    dict(message=r"X has feature names, but .* was fitted without feature names",
        category=UserWarning, module=r"sklearn\.base"),
]
# --------------------------------------------------------------------------------------

def _apply_warning_filters(enable: bool):
    if not enable:
        return
    for wf in WARN_FILTERS:
        warnings.filterwarnings("ignore", **wf)

@dataclass
class Config:
    # --- Data paths (UNC defaults can be overridden via CLI or env) ---
    train_csv: str = r"\\ncehospms01\AHP_Factory\prediction_model\LeadTime\train_data\filtered_dataset_final.csv"
    ref_csv:   str = r"\\ncehospms01\AHP_Factory\DATA\Epics.csv"

    # --- Output locations ---
    model_dir: str = r"\\ncehospms01\AHP_Factory\prediction_model\LeadTime\ML_pipeline\models"
    results_dir: str = r"\\ncehospms01\AHP_Factory\prediction_model\LeadTime\Result"
    metrics_dir: str = r"\\ncehospms01\AHP_Factory\prediction_model\LeadTime\Result\metrics"

    # --- ML params ---
    target_col: str = "needed_lead_time"
    drop_cols: List[str] = field(default_factory=lambda: [
        "key", "needed_lead_time", "existing_lead_time", "issuekey", "epic_feature_signoff", "epic_fix_version",
        "epic_owner_email", "epic_initiative_key", "fix_version_date", "dev_roadmap_date", "epic_dev_roadmap",
        "epic_jira_spent_dev_etc_md", "extract_date", "epic_level", "epic_jira_forecast_dev_etc_md",
        "epic_jira_remaining_dev_etc_md", "epic_summary", "epic_sizing_task_records",
        "epic_target_end", "epic_target_start"
    ])
    cv_folds: int = 10
    test_size: float = 0.2
    random_state: int = 42
    # selector ref-weight (only used if your selector supports it; safe to keep here for future)
    ref_weight: float = 0.10

    # Segmentation
    status_col: str = "epic_status"

    ALL_INCLUDE: List[str] = field(default_factory=list)   # optional allow-list (overrides exclude)
    ALL_EXCLUDE: List[str] = field(default_factory=lambda: ["Archived", "Backlog", "Blocked", "Abandoned", "Analyzing", "Done", "Funnel", "Implementing"])
    # 🔒 Hardcoded default groups (one model per group). Edit to your taste.
    GROUPS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Finished": [
            "Accepted", "Development Completed",
            "E2E Validated for Community", "E2E Test Completed"
        ],
        "inflight": [
            "Development Started", "Planned"
        ],
        "pre-CP": [
            "High Level Plan Done", "High Level Sizing Done"
        ],
        "Youngster": [
            "Open", "T-Shirt Sizing Done"
        ]
    })

    # What to do with statuses not listed in any group:
    #  - "skip": ignore them
    #  - "perstatus": train/predict a separate model for each leftover status
    UNGROUPED_POLICY: str = "skip"                         # "skip" or "perstatus"

    # Where the study is (use the same URL you pass to --storage when tuning)
    optuna_storage: Optional[str] = None

    # Default naming of studies:
    #  - for pre/post:  "leadtime_{segment}"   -> leadtime_pre, leadtime_post
    #  - for all-units: "leadtime_{unit_type}_{unit_slug}" -> e.g. leadtime_group_inflight
    optuna_study_tpl_segment: str = "leadtime_{segment}"
    optuna_study_tpl_unit:     str = "leadtime_{unit_type}_{unit_slug}"

    # new-data id col used in outputs
    id_col: str = "epic_id"

    # --- toggles ---
    quiet_warnings: bool = True

    refresh: bool = False

    def ensure_dirs(self):
        for d in (self.model_dir, self.results_dir, self.metrics_dir):
            os.makedirs(d, exist_ok=True)

def _study_name_for_unit(u: dict, cfg: Config, override: Optional[str] = None) -> str:
    if override: return override
    return cfg.optuna_study_tpl_unit.format(unit_type=u["type"], unit_slug=u["slug"])

def _json_best_params_path(tag: str, cfg: Config) -> str:
    # matches what tune_* writes
    return os.path.join(cfg.metrics_dir, f"optuna_{tag}", "best_params.json")

def _load_best_from_json(tag: str, cfg: Config) -> Optional[dict]:
    path = _json_best_params_path(tag, cfg)
    if not os.path.exists(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("best_params") or data  # support both shapes
    except Exception:
        return None

def _load_best_from_optuna(storage: str, study_name: str) -> Optional[dict]:
    try:
        import optuna
        study = optuna.load_study(study_name=study_name, storage=storage)
        if study.best_trial is None:
            return None
        p = study.best_trial.params
        # normalize shape we return
        return {
            "k_features": p.get("k_features"),
            "rf_params": {
                # map with defaults if user didn't tune something
                "n_estimators":     p.get("n_estimators", 300),
                "max_depth":        p.get("max_depth", 30),
                "min_samples_split":p.get("min_samples_split", 8),
                "min_samples_leaf": p.get("min_samples_leaf", 2),
                "max_features":     p.get("max_features", "sqrt"),
            }
        }
    except Exception as e:
        print(f"[optuna] Could not load study '{study_name}' from '{storage}': {e}")
        return None

def _best_hparams_for_unit(u: dict, cfg: Config, *, study_name_override: Optional[str] = None) -> Optional[dict]:
    tag = f"{u['type']}_{u['slug']}"
    # 1) from DB
    if cfg.optuna_storage:
        name = _study_name_for_unit(u, cfg, override=study_name_override)
        bp = _load_best_from_optuna(cfg.optuna_storage, name)
        if bp: return bp
    # 2) fallback JSON
    bp = _load_best_from_json(tag, cfg)
    return bp

def _apply_defaults_to_rf(bp_rf: dict, random_state: int) -> dict:
    rf = dict(bp_rf or {})
    rf.setdefault("random_state", random_state)
    rf.setdefault("class_weight", "balanced")
    rf.setdefault("n_jobs", 5)
    return rf

def _csv_list(s: str) -> List[str]:
    if not s: return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _parse_groups_arg(spec: str) -> dict:
    """
    Parse group spec like:
    "dev=Implementing,Development Started;plan=Planned,High Level Plan Done"
    Returns: {"dev": [...], "plan":[...]}
    """
    groups = {}
    if not spec: return groups
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk: continue
        if "=" not in chunk: 
            # allow "name: a,b"
            if ":" in chunk:
                name, vals = chunk.split(":", 1)
            else:
                raise ValueError(f"Bad group spec chunk: {chunk}")
        else:
            name, vals = chunk.split("=", 1)
        name = name.strip()
        statuses = _csv_list(vals)
        if not name or not statuses:
            raise ValueError(f"Bad group spec chunk: {chunk}")
        groups[name] = statuses
    return groups


def slugify(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s)).strip("_").lower()

def load_and_segment(df: pd.DataFrame, col: str, segment: str, cfg: Config) -> pd.DataFrame:
    if segment == "pre":
        return df[df[col].isin(cfg.PRE_EXCLUDE)].copy()
    elif segment == "post":
        return df[df[col].isin(cfg.POST_INCLUDE)].copy()
    elif segment == "all":
        return df.copy()
    else:
        raise ValueError(f"Unknown segment '{segment}'. Use one of: pre, post, all.")

def build_Xy(df: pd.DataFrame, cfg: Config):
    y = df[cfg.target_col]
    X = df.drop(columns=[c for c in cfg.drop_cols if c in df.columns])
    return X, y

def _safe_cv(y: pd.Series, desired_folds: int) -> int:
    min_class = int(y.value_counts().min()) if len(y) else 0
    return max(2, min(desired_folds, min_class)) if min_class > 0 else 2

def _plan_all_units(df: pd.DataFrame, cfg: Config) -> List[dict]:
    """
    Build work units for segment=all.
    If GROUPS provided -> one unit per group (type='group') and, depending on policy,
    optionally per-status for leftovers.
    Else -> per-status for included/excluded sets.
    Returns list of dicts: {"type": "group"|"status", "name": str, "slug": str, "statuses": [..]}
    """
    all_statuses = sorted(set(df[cfg.status_col].dropna().astype(str)))

    def _slug(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in s).strip("_").lower()

    units = []
    used = set()

    if cfg.GROUPS:
        # explicit groups
        for gname, glist in cfg.GROUPS.items():
            # keep only statuses present in data
            present = [s for s in glist if s in all_statuses]
            if not present:
                print(f"[all] group '{gname}' has no matching rows; skipping")
                continue
            units.append({"type":"group","name":gname,"slug":_slug(gname),"statuses":present})
            used.update(present)

        # leftovers (not used by any group)
        leftovers = [s for s in all_statuses if s not in used]
        if cfg.UNGROUPED_POLICY == "perstatus":
            for st in leftovers:
                units.append({"type":"status","name":st,"slug":_slug(st),"statuses":[st]})
        # if "skip": ignore leftovers entirely
        return units

    # No groups -> per-status with include/exclude
    if cfg.ALL_INCLUDE:
        candidates = [s for s in all_statuses if s in cfg.ALL_INCLUDE]
    else:
        candidates = [s for s in all_statuses if s not in set(cfg.ALL_EXCLUDE)]

    for st in candidates:
        units.append({"type":"status","name":st,"slug":_slug(st),"statuses":[st]})
    return units

def _run_refresh():
    """
    Runs the DB refresh script.
    Command resolution priority:
    1) Env var LT_REFRESH_CMD (full command line)
    2) Heuristics for typical paths (runs with `python`).
    Fails if --refresh was requested but no command is found or it returns non-zero.
    """
    print(utils.create_ascii_message("Refresh training dataset"))
    
    try:
        from data_retriever import refresh_initiatives
        print(f"[refresh] running in-process refresh_initiatives()")
        df_raw, raw_path = refresh_initiatives()
    except Exception as e:
        raise RuntimeError(f"[refresh] in-process refresh failed: {e}")

    # 2) Prepare dataset
    try:
        from prepare_dataset import prepare_dataset
        print(f"[prepare] running in-process prepare_dataset()")
        df_final, out_path = prepare_dataset()  # uses default paths
    except Exception as e:
        raise RuntimeError(f"[prepare] in-process prepare failed: {e}")

def train_all_units(cfg: Config):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    cfg.ensure_dirs(); _apply_warning_filters(cfg.quiet_warnings)

    if cfg.refresh == 'true':
        _run_refresh()

    df_raw = pd.read_csv(cfg.train_csv)
    ref_all = pd.read_csv(cfg.ref_csv)
    units = _plan_all_units(df_raw, cfg)

    print(f"\n[all] training {len(units)} unit(s): " + ", ".join(f"{u['type']}:{u['name']}" for u in units))
    results = {}

    for u in units:
        sts = u["statuses"]
        df_u = df_raw[df_raw[cfg.status_col].isin(sts)].copy()
        if df_u.empty:
            print(f"[skip:{u['name']}] no rows"); continue

        X, y = build_Xy(df_u, cfg)
        if y.nunique() < 2:
            print(f"[skip:{u['name']}] only one target class"); continue
        
        print(f"\n[train:{u['name']}] {len(X)} rows, {X.shape[1]} features, {y.nunique()} classes ({', '.join(str(c) for c in sorted(y.unique()))})")

        X_ref_u = ref_all[ref_all[cfg.status_col].isin(sts)].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=cfg.test_size, random_state=cfg.random_state
        )

        # best params per unit
        best = _best_hparams_for_unit(u, cfg, study_name_override=None)
        if best:
            k_features = int(best.get("k_features") or cfg.k_features)
            rf_params  = _apply_defaults_to_rf(best.get("rf_params", {}), cfg.random_state)
        else:
            k_features = cfg.k_features
            rf_params  = None

        est = LeadTimePipeline(k_features=k_features, rf_params=rf_params, model_type="classification", X_ref=X_ref_u)
        cv = StratifiedKFold(n_splits=_safe_cv(y_train, cfg.cv_folds), shuffle=True, random_state=cfg.random_state)

        y_oof = cross_val_predict(
            est, X_train, y_train, cv=cv, method="predict",
            n_jobs=5, verbose=2
        )
        acc_oof = accuracy_score(y_train, y_oof)
        f1w_oof = f1_score(y_train, y_oof, average="weighted")

        final_model = LeadTimePipeline(k_features=k_features, rf_params=rf_params, model_type="classification", X_ref=X_ref_u)
        final_model.fit(X_train, y_train)
        y_test_pred = final_model.predict(X_test)
        acc_test = accuracy_score(y_test, y_test_pred)
        f1w_test = f1_score(y_test, y_test_pred, average="weighted")

        kind = u["type"]
        tag  = f"{kind}_{u['slug']}"
        model_path = os.path.join(cfg.model_dir, f"lead_time_pipeline_{tag}.joblib")
        final_model.save(model_path)

        metrics = {
            "unit_type": kind, "unit_name": u["name"], "statuses": sts,
            "cv_folds": cv.get_n_splits(), "k_features": cfg.k_features,
            "acc_oof": float(acc_oof), "f1w_oof": float(f1w_oof),
            "acc_test": float(acc_test), "f1w_test": float(f1w_test),
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "model_path": model_path,
        }
        results[u["name"]] = metrics
        with open(os.path.join(cfg.metrics_dir, f"metrics_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"[{kind}:{u['name']}] OOF acc={acc_oof:.3f} f1w={f1w_oof:.3f} | Test acc={acc_test:.3f} f1w={f1w_test:.3f}")
        print(f"[{kind}:{u['name']}] Saved model -> {model_path}")

def predict_all_units(cfg: Config):
    _apply_warning_filters(cfg.quiet_warnings); cfg.ensure_dirs()

    X_new_all = pd.read_csv(cfg.ref_csv)
    units = _plan_all_units(X_new_all, cfg)
    print(f"\n[all] predicting {len(units)} unit(s): " + ", ".join(f"{u['type']}:{u['name']}" for u in units))

    for u in units:
        sts = u["statuses"]
        X_new = X_new_all[X_new_all[cfg.status_col].isin(sts)].copy()
        if X_new.empty:
            print(f"[skip:{u['name']}] no rows"); continue

        tag = f"{u['type']}_{u['slug']}"
        model_path = os.path.join(cfg.model_dir, f"lead_time_pipeline_{tag}.joblib")
        if not os.path.exists(model_path):
            print(f"[skip:{u['name']}] model not found: {model_path}"); continue
        
        print(f"\n[Predict:{u['name']}] {len(X_new)} rows")

        pipe = LeadTimePipeline.load(model_path)
        y_pred  = pipe.predict(X_new)
        y_proba = pipe.predict_proba(X_new)
        classes = pipe.pipeline_.named_steps["classifier"].classes_

        meta = pd.DataFrame({
            "epic_key": X_new.get(cfg.id_col, pd.Series(index=X_new.index)),
            cfg.status_col: X_new.get(cfg.status_col, pd.Series(index=X_new.index)),
            "predicted_lead_time": pd.Series(y_pred, index=X_new.index),
            "Proba_Max": pd.Series(y_proba.max(axis=1), index=X_new.index)
        })
        proba_df = pd.DataFrame(y_proba, columns=[f"proba_{c}" for c in classes], index=X_new.index)
        df_result = pd.concat([meta, proba_df], axis=1)

        xlsx_path = os.path.join(cfg.results_dir, f"predictions_{tag}.xlsx")
        csv_path  = os.path.join(cfg.results_dir, f"predictions_{tag}.csv")
        try:
            if export_xlsx is not None:
                raw_feats = pipe.transform_for_classifier(X_new, as_dataframe=True)
                export_xlsx(df_result, xlsx_path, raw_features=raw_feats, model_name=tag)
                print(f"[{u['name']}] Wrote {xlsx_path}")
            else:
                df_result.to_csv(csv_path, index=False)
                print(f"[{u['name']}] Wrote {csv_path} (CSV fallback)")
        except Exception as e:
            df_result.to_csv(csv_path, index=False)
            print(f"[{u['name']}] Excel export failed ({e}). Wrote CSV -> {csv_path}")

def tune_all_units(cfg: Config, n_trials: int, timeout: Optional[int], storage: Optional[str], study_name: Optional[str]):
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.model_selection import StratifiedKFold, cross_validate

    cfg.ensure_dirs(); _apply_warning_filters(cfg.quiet_warnings)

    if cfg.refresh == 'true':
        _run_refresh()

    df_raw = pd.read_csv(cfg.train_csv)
    ref_all = pd.read_csv(cfg.ref_csv)
    units = _plan_all_units(df_raw, cfg)
    print(f"\n[all] tuning {len(units)} unit(s): " + ", ".join(f"{u['type']}:{u['name']}" for u in units))

    for u in units:
        sts = u["statuses"]
        df_u = df_raw[df_raw[cfg.status_col].isin(sts)].copy()
        if df_u.empty:
            print(f"[skip:{u['name']}] no rows"); continue
        X, y = build_Xy(df_u, cfg)
        if y.nunique() < 2:
            print(f"[skip:{u['name']}] one target class"); continue
        X_ref_u = ref_all[ref_all[cfg.status_col].isin(sts)].copy()
        
        print(f"\n[Tune:{u['name']}] {len(X)} rows, {X.shape[1]} features, {y.nunique()} classes ({', '.join(str(c) for c in sorted(y.unique()))})")

        n_splits = _safe_cv(y, cfg.cv_folds)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

        sampler = TPESampler(seed=cfg.random_state)
        study = optuna.create_study(
            direction="maximize",
            study_name=(study_name or f"leadtime_{u['type']}_{u['slug']}"),
            storage=storage, load_if_exists=bool(storage),
            sampler=sampler
        )
        study.set_user_attr("data_rows", int(len(X)))
        study.set_user_attr("class_hist", dict(y.value_counts().to_dict()))
        study.set_user_attr("cv_folds", int(n_splits))
        study.set_user_attr("code_version", "leadtime-2025-09-24")

        def objective(trial):
            k_features = trial.suggest_int("k_features", 60, 260, step=20)
            rf_params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 10, 60, step=5),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "random_state": cfg.random_state, "class_weight": "balanced", "n_jobs": 5,
            }
            est = LeadTimePipeline(k_features=k_features, rf_params=rf_params, model_type="classification", X_ref=X_ref_u)
            scores = cross_validate(
                est, X, y, cv=cv,
                scoring={"f1w": "f1_weighted", "acc": "accuracy"},
                return_train_score=False, error_score="raise", n_jobs=5
            )
            trial.set_user_attr("acc", float(np.mean(scores["test_acc"])))
            return float(np.mean(scores["test_f1w"]))

        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        best = study.best_trial
        bp   = best.params

        tuned_rf = {
            "n_estimators": bp["n_estimators"], "max_depth": bp["max_depth"],
            "min_samples_split": bp["min_samples_split"], "min_samples_leaf": bp["min_samples_leaf"],
            "max_features": bp["max_features"],
            "random_state": cfg.random_state, "class_weight": "balanced", "n_jobs": 5,
        }
        tuned_k = bp["k_features"]

        tag = f"{u['type']}_{u['slug']}"
        study_dir = os.path.join(cfg.metrics_dir, f"optuna_{tag}")
        os.makedirs(study_dir, exist_ok=True)

        out = {
            "unit_type": u["type"], "unit_name": u["name"], "statuses": sts,
            "cv_folds": n_splits, "best_f1w": float(best.value),
            "best_acc": float(best.user_attrs.get("acc", float("nan"))),
            "best_params": {"k_features": tuned_k, "rf_params": tuned_rf},
            "n_trials": len(study.trials),
        }
        with open(os.path.join(study_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        try:
            study.trials_dataframe(attrs=("number","value","params","user_attrs","state")).to_csv(
                os.path.join(study_dir, "trials.csv"), index=False
            )
        except Exception:
            pass
        print(f"[tune:{u['name']}] best f1w={out['best_f1w']:.4f}, acc={out['best_acc']:.4f}")

        final = LeadTimePipeline(k_features=tuned_k, rf_params=tuned_rf, model_type="classification", X_ref=X_ref_u)
        final.fit(X, y)
        model_path = os.path.join(cfg.model_dir, f"lead_time_pipeline_{tag}.joblib")
        final.save(model_path)
        print(f"[tune:{u['name']}] saved tuned model -> {model_path}")

# ---------- CLI ----------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LeadTime training/prediction driver")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_shared(sp):
        sp.add_argument("--train-csv", default=os.getenv("LT_TRAIN_CSV"))
        sp.add_argument("--ref-csv",   default=os.getenv("LT_REF_CSV"))
        sp.add_argument("--model-dir", default=os.getenv("LT_MODEL_DIR"))
        sp.add_argument("--results-dir", default=os.getenv("LT_RESULTS_DIR"))
        sp.add_argument("--metrics-dir", default=os.getenv("LT_METRICS_DIR"))
        sp.add_argument("--segment", choices=["pre","post","all"], default=os.getenv("LT_SEGMENT","pre"))
        sp.add_argument("--quiet-warnings", action="store_true", default=True)
        sp.add_argument("--all-include", default=os.getenv("LT_ALL_INCLUDE", ""))   # comma-separated
        sp.add_argument("--all-exclude", default=os.getenv("LT_ALL_EXCLUDE", ""))   # comma-separated
        sp.add_argument("--all-groups",  default=os.getenv("LT_ALL_GROUPS", ""))    # "name=a,b;name2=c,d"
        sp.add_argument("--ungrouped", choices=["skip","perstatus"], default=os.getenv("LT_UNGROUPED", "skip"))
        sp.add_argument("--storage", default=os.getenv("LT_OPTUNA_STORAGE"))
        sp.add_argument("--study-name", default="sqlite://///ncehospms01/AHP_Factory/prediction_model/LeadTime/Result/metrics/optuna_all_status.db")  # optional explicit override


    sp_t = sub.add_parser("train", help="Train and save model(s)")
    add_shared(sp_t)
    sp_t.add_argument("--k-features", type=int, default=int(os.getenv("LT_K_FEATURES", "140")))
    sp_t.add_argument("--cv-folds",   type=int, default=int(os.getenv("LT_CV_FOLDS", "10")))
    sp_t.add_argument("--test-size",  type=float, default=float(os.getenv("LT_TEST_SIZE", "0.2")))
    sp_t.add_argument("--random-state", type=int, default=int(os.getenv("LT_RANDOM_STATE", "42")))
    sp_t.add_argument("--refresh", type=str, default="true")

    sp_p = sub.add_parser("predict", help="Predict with a saved model")
    add_shared(sp_p)
    sp_p.add_argument("--model-path", default=os.getenv("LT_MODEL_PATH"))

    sp_u = sub.add_parser("tune", help="Optuna hyperparameter search")
    add_shared(sp_u)
    sp_u.add_argument("--n-trials", type=int, default=int(os.getenv("LT_N_TRIALS", "40")))
    sp_u.add_argument("--timeout", type=int, default=int(os.getenv("LT_TIMEOUT", "0")))
    sp_u.add_argument("--refresh", type=str, default="true")

    return p.parse_args(argv)

def cfg_from_args(a) -> Config:
    cfg = Config()
    if a.train_csv:   cfg.train_csv   = a.train_csv
    if a.ref_csv:     cfg.ref_csv     = a.ref_csv
    if a.model_dir:   cfg.model_dir   = a.model_dir
    if a.results_dir: cfg.results_dir = a.results_dir
    if a.metrics_dir: cfg.metrics_dir = a.metrics_dir
    if hasattr(a, "k_features"): cfg.k_features = a.k_features
    if hasattr(a, "cv_folds"):   cfg.cv_folds   = a.cv_folds
    if hasattr(a, "test_size"):  cfg.test_size  = a.test_size
    if hasattr(a, "random_state"): cfg.random_state = a.random_state
    if hasattr(a, "quiet_warnings"): cfg.quiet_warnings = a.quiet_warnings
    if hasattr(a, "all_include") and a.all_include: cfg.ALL_INCLUDE = _csv_list(a.all_include)
    if hasattr(a, "all_exclude") and a.all_exclude: cfg.ALL_EXCLUDE = _csv_list(a.all_exclude)
    if hasattr(a, "all_groups") and a.all_groups: cfg.GROUPS = _parse_groups_arg(a.all_groups)
    if hasattr(a, "ungrouped") and a.ungrouped: cfg.UNGROUPED_POLICY = a.ungrouped
    if getattr(a, "storage", None): cfg.optuna_storage = a.storage # note: study-name is used at call sites (not a global default)
    if hasattr(a, "refresh"): cfg.refresh = a.refresh


    # parse ALL_EXCLUDE override
    if hasattr(a, "all_exclude") and a.all_exclude:
        cfg.ALL_EXCLUDE = [s.strip() for s in a.all_exclude.split(",") if s.strip()]
    return cfg

def main(argv=None):
    args = parse_args(argv)
    cfg = cfg_from_args(args)

    if args.cmd == "train":
        train_all_units(cfg)

    elif args.cmd == "predict":
        predict_all_units(cfg)

    elif args.cmd == "tune":
        tune_all_units(
                cfg,
                n_trials=getattr(args, "n_trials", 40),
                timeout=(getattr(args, "timeout", 0) or None),
                storage=getattr(args, "storage", None),
                study_name=getattr(args, "study_name", None),
            )

if __name__ == "__main__":
    sys.exit(main())