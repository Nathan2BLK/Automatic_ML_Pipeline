import os
import sys
import pathlib
from pathlib import Path
import copy
from collections import defaultdict
from datetime import datetime
import concurrent.futures
import sqlite3
import argparse
from typing import Optional, Tuple, List

# make project src importable if run standalone
SRC = pathlib.Path(__file__).resolve().parents[2]  # -> .../src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from tqdm import tqdm
from common import jira_utils, utils

# -------------------- Global-ish constants (logic expects these names) --------------------
issuetype = "initiative"
file = "Epics.csv"  # key to pick column mapping below

# Default I/O locations
DEFAULT_DB_PATH = Path(r"\\ncehospms01\AHP_Factory\prediction_model\LeadTime\train_data\equivalence.db")
DEFAULT_OUT_CSV = Path(r"\\ncehospms01\AHP_Factory\prediction_model\LeadTime\train_data\output_initiative.csv")

# Will be set inside refresh_initiatives()
DB_PATH: Optional[Path] = None
current_fields_to_considere: List[str] = []
change_fields_to_considere: List[str] = []
period = None
max_date: Optional[datetime] = None
target_date: Optional[List[str]] = None

# -------------------- Field mapping (unchanged) --------------------
def process_text(x): return x.replace(",", " ").replace('"', '') if x else x
def process_name(x): return x.name if x else x
def process_value(x): return x.value if x else x
def process_key(x): return x.key if x else x
def process_display_name(x): return process_text(x.displayName) if x else x
def process_email(x): return x.emailAddress.lower() if x else x
def process_list_names(x): return '#'.join(sorted([i.name for i in x])) if x else None
def process_list_values(x): return '#'.join(sorted([i.value for i in x])) if x else None
def process_string_list(x): return '#'.join(sorted(x)) if x else None
def process_split_list(x): return '#'.join(sorted(x.split())) if x else None
def process_float(x): return float(x) if x else x
def process_int(x): return int(x) if x else x
def process_time_estimate(x): return float(utils.get_days_from_sec(int(x), format='d', unit=False)) if x else 0.0
def process_timetracking_timespent(x): return float(utils.get_days_from_sec(x.timeSpentSeconds, format='d', unit=False)) if hasattr(x, 'timeSpentSeconds') else 0.0
def process_timetracking_timeestimate(x): return float(utils.get_days_from_sec(x.remainingEstimateSeconds, format='d', unit=False)) if hasattr(x, 'remainingEstimateSeconds') else 0.0

columns = {
    'Epics.csv': {
        'epic_agile_release_train': {'custom_field': 'customfield_16121'},
        'epic_agile_release_train_SAFe': {'custom_field': 'customfield_20203'},
        'epic_business_value': {'custom_field': 'customfield_10003'},
        'epic_components': {'custom_field': 'components'},
        'epic_customer_ref': {'custom_field': 'customfield_18051'},
        'epic_dev_cutoff': {'custom_field': ''},
        'epic_dev_roadmap': {'custom_field': 'customfield_18300'},
        'epic_exec_status': {'custom_field': 'customfield_15143'},
        'epic_external_hld_status': {'custom_field': 'customfield_15102'},
        'epic_ext_ref': {'custom_field': ''},
        'epic_feature_signoff': {'custom_field': 'customfield_16114', 'modify_to_choose': 'id'},
        'epic_fix_version': {'custom_field': 'fixVersions'},
        'epic_functional_area': {'custom_field': 'customfield_18055'},
        'epic_initiative_key': {'custom_field': 'customfield_21602'},
        'epic_jira_forecast_dev_etc_md': {'custom_field': 'timeoriginalestimate'},
        'epic_jira_remaining_dev_etc_md': {'custom_field': 'timetracking', 'processing': 1, 'name': 'timeestimate'},
        'epic_jira_spent_dev_etc_md': {'custom_field': 'timetracking', 'processing': 2, 'name': 'timespent'},
        'epic_labels': {'custom_field': 'labels'},
        'epic_level': {'custom_field': 'customfield_20200'},
        'epic_mgs_status': {'custom_field': 'customfield_14500'},
        'epic_owner': {'custom_field': 'customfield_18903'},
        'epic_owner_email': {'custom_field': 'customfield_18903', 'processing': 1, 'modprocessing': 'mod1'},
        'epic_PI': {'custom_field': 'fixVersions'},
        'epic_priority': {'custom_field': 'priority'},
        'epic_product_manager': {'custom_field': 'customfield_18901'},
        'epic_product_releases': {'custom_field': 'customfield_19004'},
        'epic_programs': {'custom_field': 'customfield_15700'},
        'epic_progress': {'custom_field': 'customfield_15114'},
        'epic_release_name': {'custom_field': 'fixVersions'},
        'epic_requirement_review_status': {'custom_field': 'customfield_15104'},
        'epic_risk_level': {'custom_field': 'customfield_15118'},
        'epic_scheduled_end': {'custom_field': 'customfield_13211'},
        'epic_scheduled_start': {'custom_field': 'customfield_13210', 'modify_to_choose': 'id'},
        'epic_service_order': {'custom_field': 'customfield_16803'},
        'epic_solution_manager': {'custom_field': 'customfield_18902'},
        'epic_solution_review_status': {'custom_field': 'customfield_15105'},
        'epic_status': {'custom_field': 'status'},
        'epic_strategy': {'custom_field': 'customfield_17004'},
        'epic_summary': {'custom_field': 'summary'},
        'epic_sync_parameters': {'custom_field': 'customfield_15101'},
        'epic_target_end': {'custom_field': 'customfield_12808', 'modify_to_choose': 'id'},
        'epic_target_start': {'custom_field': 'customfield_12807', 'modify_to_choose': 'id'},
        'epic_team': {'custom_field': 'customfield_12300'},
        'epic_versions': {'custom_field': 'versions'},
        'epic_sizing_task_records': {'custom_field': 'customfield_15401'}
    }
}

mapping = {
    'issuekey': {'name': 'key'},
    'summary': {'name': 'summary', 'default process': process_text, 'processmod0': process_text},
    'customfield_18903': {'name': 'Epic Owner', 'default process': process_display_name, 'process1': process_email,
                        'processmod1': lambda x: jira_utils.get_user_email_from_name(x, ji,) if x else x},
    'customfield_21602': {'name': 'Initiative KEY', 'default process': process_key},
    'fixVersions': {'name': 'Fix Version', 'default process': process_list_names, 'type': 'list'},
    'status': {'name': 'status', 'default process': process_name},
    'customfield_15104': {'name': 'RRM Status', 'default process': process_value},
    'customfield_15105': {'name': 'SRM Status', 'default process': process_value},
    'customfield_15401': {'name': 'Management Suite Record ID'},
    'customfield_16803': {'name': 'Service Order ID'},
    'versions': {'name': 'Version', 'default process': process_list_names, 'type': 'list'},
    'customfield_12807': {'name': 'Target start'},
    'labels': {'name': 'labels', 'default process': process_string_list, 'processmod0': process_split_list},
    'customfield_15101': {'name': 'Ext System Sync Parameters', 'default process': process_string_list, 'processmod0': process_split_list},
    'customfield_15143': {'name': 'Lead time (SSA)'},
    'customfield_15118': {'name': 'Status report (SSA)'},
    'customfield_12808': {'name': 'Target end'},
    'customfield_14500': {'name': 'Requirements', 'default process': process_string_list, 'processmod0': process_split_list},
    'customfield_16114': {'name': 'Feature Sign-off date'},
    'customfield_18051': {'name': 'Customer'},
    'customfield_18300': {'name': 'Development Roadmap', 'default process': process_list_names},
    'customfield_17004': {'name': 'Strategy', 'default process': process_string_list, 'processmod0': process_split_list},
    'customfield_10003': {'name': 'Business Value', 'processmod0': process_float},
    'components': {'name': 'Component', 'default process': process_list_names, 'type': 'list'},
    'customfield_18902': {'name': 'Solution Manager', 'default process': process_display_name},
    'customfield_19004': {'name': 'Product Release', 'default process': process_list_names, 'processmod0': process_split_list},
    'customfield_13211': {'name': 'Scheduled End Date'},
    'priority': {'name': 'priority', 'default process': process_name},
    'customfield_10900': {'name': 'Rank'},
    'customfield_18055': {'name': 'Functional area', 'default process': process_list_values, 'type': 'list'},
    'customfield_18901': {'name': 'Product Manager', 'default process': process_display_name},
    'customfield_13210': {'name': 'Scheduled Start Date'},
    'customfield_15114': {'name': 'Progress Indicator (%)', 'processmod0': process_int},
    'timeoriginalestimate': {'name': 'timeoriginalestimate', 'default process': process_time_estimate, 'processmod0': process_time_estimate},
    'timetracking': {'name': ['timespent', 'timeestimate'], 'process1': process_timetracking_timeestimate, 'process2': process_timetracking_timespent, 'processmod0': process_time_estimate},
    'customfield_20200': {'name': 'Epic Level', 'default process': process_value},
    'customfield_15102': {'name': 'External HLD Status', 'default process': process_value},
    'customfield_12300': {'name': 'Team', 'default process': process_name},
    'customfield_16121': {'name': 'Agile Release Train', 'default process': process_value},
    'customfield_20203': {'name': 'Agile Release Train (SAFe)', 'default process': process_list_values},
    'customfield_15700': {'name': 'Customer Programs', 'default process': process_list_values, 'processmod0': process_split_list},
}

# -------------------- Helpers (mostly unchanged) --------------------
def check_equivalence(a, b):
    """Automated: treat as equivalent, and persist into SQLite (thread-safe connection)."""
    if a == b or (a in (None, '') and b in (None, '')):
        return True

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM equivalences WHERE (var1=? AND var2=?) OR (var1=? AND var2=?)",
        (a, b, b, a)
    )
    if cursor.fetchone():
        conn.close()
        return True

    # Non-interactive: automatically record as equivalent (was default 'y' in your script)
    cursor.execute("INSERT OR IGNORE INTO equivalences (var1, var2) VALUES (?, ?)", (a, b))
    conn.commit()
    conn.close()
    return True

def apply_processing(field, value, num):
    if field in mapping:
        processing = 'default process' if num == 0 else f'process{num}'
        if processing in mapping[field]:
            return mapping[field][processing](value)
    return value

def set_current(issue):
    states = {'issuekey': issue.key}
    for fld, spec in columns[file].items():
        cf = spec.get('custom_field')
        num = spec.get('processing', 0)
        if cf and cf in current_fields_to_considere and cf not in states:
            states[fld] = apply_processing(cf, getattr(issue.fields, cf), num)
    return dict(sorted(states.items()))

def extract_next_update(info, current):
    updated = copy.deepcopy(info)
    not_found = []
    for item in current.items:
        if item.field in change_fields_to_considere:
            custom_field = next(
                (key for key, val in mapping.items()
                if val['name'] == item.field or (isinstance(val['name'], list) and item.field in val['name'])),
                None
            )
            if custom_field is None:
                not_found.append(item.field)
                continue

            headers_to_modify = []
            for hdr, custom in columns[file].items():
                if custom.get('custom_field') == custom_field:
                    headers_to_modify.append(hdr)
                if custom.get('name') == item.field:
                    headers_to_modify = [hdr]
                    break

            for fld in headers_to_modify:
                # choose id or string depending on mapping
                choose_id = columns[file][fld].get('modify_to_choose') == 'id'
                before = getattr(item, 'from')
                after  = getattr(item, 'to')
                v_before = before if choose_id else item.fromString
                v_after  = after  if choose_id else item.toString

                if 'type' not in mapping[custom_field] or mapping[custom_field]['type'] == 'string':
                    mod = columns[file][fld].get('modprocessing', 'mod0')
                    new_val = apply_processing(custom_field, v_after, mod)
                    if check_equivalence(updated[fld], new_val):
                        updated[fld] = apply_processing(custom_field, v_before, mod)
                    else:
                        not_found.append(item.field)
                elif mapping[custom_field]['type'] == 'list':
                    curr = updated[fld].split('#') if updated[fld] not in ([], None) else []
                    curr = [x for x in curr if x != '']
                    if v_before in [None, ''] and v_after not in [None, ''] and v_after in curr:
                        curr.remove(v_after)
                        updated[fld] = "#".join(curr)
                    elif v_before not in [None, ''] and v_after in [None, ''] and v_before not in curr:
                        curr.append(v_before)
                        updated[fld] = "#".join(curr)
                    else:
                        not_found.append(item.field)
                else:
                    not_found.append(item.field)
    return dict(sorted(updated.items())), not_found

def clean_up(datas: dict):
    for data in datas.values():
        for k, v in list(data.items()):
            if v == '':
                data[k] = None
    return datas

def process_single_issue(issue, jiraInstance):
    issue = jira_utils.search_issues([f'key = {issue.key}'], jiraInstance=jiraInstance,
                                    fields=current_fields_to_considere + ['created'], expand='changelog')
    issue = issue[0] if isinstance(issue, list) else issue
    infos = {issue.changelog.histories[-1].created: set_current(issue)}
    not_founds_local = []

    for i in reversed(range(0, len(issue.changelog.histories))):
        prev_date = issue.changelog.histories[i - 1].created if i >= 1 else issue.fields.created
        current = issue.changelog.histories[i]
        refined_date = datetime.strptime(prev_date[:10], "%Y-%m-%d")
        if max_date is None or refined_date >= max_date:
            data, nf = extract_next_update(infos[current.created], current)
            infos[prev_date] = dict(sorted(data.items()))
            not_founds_local += nf

    if not_founds_local:
        print("\n".join(sorted(set(not_founds_local))))
    infos = clean_up(infos)
    return issue.key, infos

def get_info_on_issue_multithreaded(issues, jiraInstance, max_workers=4):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut_to_issue = {executor.submit(process_single_issue, issue, jiraInstance): issue for issue in issues}
        for fut in tqdm(concurrent.futures.as_completed(fut_to_issue), total=len(issues), desc="Processing Issues"):
            try:
                issue_key, infos = fut.result()
                results[issue_key] = infos
            except Exception as e:
                print(f"Error processing issue: {fut_to_issue[fut]} - {e}")
    return results

# -------------------- Public entry point --------------------
def refresh_initiatives(
    *,
    jqls: str = "type = initiative and project in (AHPCOM, TECH)",
    start_date: str = "",
    end_date: str = "",
    target_dates: str = "",
    use_env_creds: bool = True,
    output_csv_path: Path = DEFAULT_OUT_CSV,
    db_path: Path = DEFAULT_DB_PATH,
    max_workers: int = 4,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Refresh initiatives dataset from Jira and write a daily-last-entry CSV.

    Returns: (df, output_path_str)
    """

    # --- auth & Jira instance ---
    jirauser = os.environ.get("JIRA_USERNAME") if use_env_creds else None
    jirapwd  = os.environ.get("JIRA_PASSWORD") if use_env_creds else None
    global ji
    ji = jira_utils.create_jira_instance(userName=jirauser, passWord=jirapwd)
    

    # --- DB init (thread-safe to open per-call later) ---
    global DB_PATH
    DB_PATH = Path(db_path)
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS equivalences (
            var1 TEXT,
            var2 TEXT,
            PRIMARY KEY (var1, var2)
        )
        """)
        conn.commit()

    # --- dates logic replicated from your script ---
    global period, max_date, target_date
    max_date = None
    period = None
    target_date = None

    if start_date != "" or end_date != "":
        try:
            sd = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.strptime('0001-01-01', "%Y-%m-%d")
            ed = datetime.strptime(end_date,   "%Y-%m-%d") if end_date   else datetime.strptime('9999-10-19', "%Y-%m-%d")
            period = (sd, ed)
            max_date = sd
        except Exception as e:
            raise SystemExit(f"ERROR handling dates: {e}")

    if target_dates:
        td = target_dates.replace(' ', '')
        target_date = td.split('|||')
        period = None
        max_date = datetime.strptime(max(target_date, key=lambda d: datetime.strptime(d, "%Y-%m-%d")), "%Y-%m-%d")

    # --- fields to consider (build globals used by helpers) ---
    global current_fields_to_considere, change_fields_to_considere
    current_fields_to_considere = list({spec.get('custom_field') for spec in columns[file].values() if spec.get('custom_field')})
    change_fields_to_considere = []
    for item in current_fields_to_considere:
        name = mapping[item]['name']
        if isinstance(name, list):
            change_fields_to_considere += list(set(name))
        else:
            change_fields_to_considere.append(name)
    change_fields_to_considere = list(set(change_fields_to_considere))

    # --- fetch issues ---
    jql_list = [jqls]  # keep behavior compatible with your jira_utils helper
    issues = jira_utils.search_issues(jql_list, jiraInstance=ji, fields=['key'])
    if verbose:
        print(f"Number of issues fetched: {len(issues)}")

    # --- process (parallel) ---
    infos = get_info_on_issue_multithreaded(issues, jiraInstance=ji, max_workers=max_workers)

    # --- reduce to daily last entries (+ optional target date snaps) ---
    filtered_worklogs = []
    for key, info in tqdm(infos.items(), desc='Processing Sorting of item By Date'):
        daily_entries = defaultdict(list)
        for dt, worklog in info.items():
            date_str = dt[:10]
            daily_entries[date_str].append((dt, worklog))

        dates = sorted(daily_entries.keys())
        date_objects = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

        # keep last per day (period window if set)
        for date, logs in daily_entries.items():
            refined_date = datetime.strptime(date, "%Y-%m-%d")
            latest_log = max(logs, key=lambda x: x[0])
            latest_log[1]['key'] = key
            latest_log[1]['extract_date'] = date
            if (period is not None and period[0] <= refined_date <= period[1]) or (period is None and max_date is None):
                filtered_worklogs.append(dict(sorted(latest_log[1].items())))

        # target snapshots
        if max_date is not None and target_date is not None:
            for target in target_date:
                target_dt = datetime.strptime(target, "%Y-%m-%d")
                if target_dt in date_objects:
                    closest_date = target_dt
                else:
                    closest_date = max((d for d in date_objects if d < target_dt), default=None)
                if closest_date:
                    closest_str = closest_date.strftime("%Y-%m-%d")
                    latest_log = max(daily_entries[closest_str], key=lambda x: x[0])
                    latest_log[1]['key'] = key
                    latest_log[1]['extract_date'] = closest_str
                    filtered_worklogs.append(dict(sorted(latest_log[1].items())))

    # --- dataframe & write ---
    priority_columns = ['extract_date', 'key']
    df = pd.DataFrame(list(filtered_worklogs))
    if not df.empty:
        df = df[priority_columns + [c for c in df.columns if c not in priority_columns]]

    out_path = Path(output_csv_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"Wrote {out_path} ({len(df)} rows)")
    return df, str(out_path)

# -------------------- Optional CLI (no args needed) --------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Refresh initiatives dataset (standalone or imported).")
    p.add_argument("--jql", default="", help="JQL to filter issues (optional)")
    p.add_argument("--start-date", default="", help="YYYY-MM-DD (optional)")
    p.add_argument("--end-date", default="", help="YYYY-MM-DD (optional)")
    p.add_argument("--target-dates", default="", help="YYYY-MM-DD|||YYYY-MM-DD (optional)")
    p.add_argument("--max-workers", type=int, default=10)
    p.add_argument("--no-env-creds", action="store_true", help="Do not read JIRA_USERNAME/PASSWORD from env")
    p.add_argument("--out", default=str(DEFAULT_OUT_CSV), help="Output CSV path")
    p.add_argument("--db",  default=str(DEFAULT_DB_PATH), help="SQLite equivalence DB path")
    return p.parse_args()

if __name__ == "__main__":
    a = _parse_args()
    refresh_initiatives(
        jqls=a.jql,
        start_date=a.start_date,
        end_date=a.end_date,
        target_dates=a.target_dates,
        use_env_creds=not a.no_env_creds,
        output_csv_path=Path(a.out),
        db_path=Path(a.db),
        max_workers=a.max_workers,
        verbose=True,
    )
