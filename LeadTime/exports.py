import os
import pandas as pd
import numpy as np
import math
from xlsxwriter.utility import xl_cell_to_rowcol

def export_predictions_to_excel_pretty(
        df_result: pd.DataFrame,
        path: str = "predictions.xlsx",
        *,
        raw_features: pd.DataFrame | None = None,  # same row order; used to compute data_quality
        # --- NEW master ledger options ---
        update_master: bool = True,
        master_path: str | None = r"\\ncehospms01\AHP_Factory\prediction_model\LeadTime\Result\all_predictions_master.xlsx",
        master_key_cols: list[str] | None = None,
        model_name: str | None = None,
    ):
    """
    Writes a nicely formatted Excel for this batch (as before),
    and optionally upserts the batch into a central 'master' workbook that
    keeps all formatting & the Insights sheet up to date.

    New columns:
      - model_name        : passed in param or 'default_model'
      - last_prediction_at: timezone-naive ISO string of the export time (Europe/Paris)
    """

    # ============================== INTERNAL HELPERS ==============================
    def _sorted_proba_cols(df: pd.DataFrame) -> list[str]:
        cols = [c for c in df.columns if c.startswith("proba_")]
        def key(c):
            lab = c[6:]
            if lab.endswith("+"):
                return (int(lab[:-1]), 1)
            try:
                return (int(lab), 0)
            except:
                return (10**9, 0)
        return [c for c in sorted(cols, key=key)]

    def _build_with_gaps(df: pd.DataFrame, groups: list[list[str]]):
        """
        Build a new dataframe with columns laid out as groups, inserting a blank
        spacer col between each group. Returns (df_out, gap_indices, gap_names).
        """
        cols_out, gap_idx, gap_names = [], [], []
        for gi, grp in enumerate(groups):
            existing = [c for c in grp if c in df.columns]
            cols_out.extend(existing)
            if gi < len(groups) - 1:
                gap_name = f"__GAP__{gi}"
                cols_out.append(gap_name)
                gap_idx.append(len(cols_out) - 1)
                gap_names.append(gap_name)
        df_out = df.reindex(columns=cols_out)
        for g in gap_names:
            df_out[g] = ""
        return df_out, gap_idx, gap_names

    def _compute_metrics(df_input: pd.DataFrame, raw_features_input: pd.DataFrame | None):
        """Return a new df with all metrics/indicators added (trust_score_v2 etc.)."""
        df = df_input.copy()
        proba_cols = [c for c in df.columns if str(c).startswith("proba_")]

        # ---------- Base indicators ----------
        if proba_cols:
            # p1/p2 (order-agnostic)
            arr_raw = df[proba_cols].to_numpy(dtype=float)
            top2 = np.partition(arr_raw, -2, axis=1)[:, -2:]
            p2 = top2[:, 0]
            p1 = top2[:, 1]
            if "Proba_Max" not in df.columns:
                df["Proba_Max"] = p1
            df["top2_proba"]  = p2
            df["top2_margin"] = p1 - p2

            # ---------- Ordinal-aware parts ----------
            raw_labels = [c[len("proba_"):] for c in proba_cols]
            def label_to_num(s: str) -> float:
                try:
                    return float(int(s))
                except Exception:
                    return float(int(s[:-1])) if s.endswith("+") else np.nan
            positions = np.array([label_to_num(s) for s in raw_labels], dtype=float)
            valid_mask = ~np.isnan(positions)
            if valid_mask.sum() >= 2:
                order_idx = np.argsort(positions[valid_mask])
                pos = positions[valid_mask][order_idx]
                arr = arr_raw[:, valid_mask][:, order_idx]  # shape [n, K_valid]

                Dmax = pos.max() - pos.min()
                Dmax = Dmax if Dmax > 0 else 1.0

                top_idx = np.argmax(arr, axis=1)

                # 1) proximity_score
                dist_all = np.abs(pos[None, :] - pos[:, None]) / Dmax  # [K,K]
                prox_penalty = np.einsum("ij,ij->i", arr, dist_all[top_idx, :])
                df["proximity_score"] = 1.0 - prox_penalty

                # 2) local_mass_pm1
                R = 1
                local = []
                for i, probs in enumerate(arr):
                    c = top_idx[i]
                    a, b = max(0, c-R), min(len(probs)-1, c+R)
                    local.append(float(probs[a:b+1].sum()))
                df["local_mass_pm1"] = local

                # 3) dispersion_score
                x = pos
                mu = (arr * x).sum(axis=1)
                var = (arr * (x - mu[:, None])**2).sum(axis=1)
                var_max = (Dmax**2) / 4.0 if Dmax > 0 else 1.0
                df["dispersion_score"] = 1.0 - (var / var_max)
            else:
                df["proximity_score"]  = np.nan
                df["local_mass_pm1"]   = np.nan
                df["dispersion_score"] = np.nan

            # entropy (normalized)
            with np.errstate(divide='ignore', invalid='ignore'):
                safe = np.where(arr_raw > 0, arr_raw, 1.0)
                ent = -(arr_raw * np.log2(safe))
                ent[arr_raw == 0] = 0.0
            df["entropy_bits"] = ent.sum(axis=1)
            K_valid = np.count_nonzero(~np.isnan(positions))
            Hmax = math.log2(K_valid) if K_valid > 1 else 1.0
            df["entropy_norm"] = (df["entropy_bits"] / Hmax).clip(0, 1)
            df.drop(columns=["entropy_bits"], inplace=True, errors="ignore")

        # ---------- Optional: data quality (coverage) ----------
        if raw_features_input is not None:
            rf = raw_features_input.reset_index(drop=True)
            if rf.shape[1] == 0:
                dq = np.ones(len(df), dtype=float)
            else:
                mask = pd.notna(rf).to_numpy()
                dq = mask.mean(axis=1).astype(float)
            dq = np.nan_to_num(dq, nan=0.0)
            dq = np.clip(dq, 0.0, 1.0)
            df["data_quality"] = pd.Series(dq, index=df.index)
        else:
            df["data_quality"] = 1.0

        # ---------- Composite trust score (v2) ----------
        def _nz(col):
            return df[col].astype(float) if col in df.columns else 0.0

        Proba_Max       = _nz("Proba_Max").clip(0, 1)
        top2_margin     = _nz("top2_margin").clip(0, 1)
        proximity_score = _nz("proximity_score").clip(0, 1)
        local_mass      = _nz("local_mass_pm1").clip(0, 1)
        dispersion      = _nz("dispersion_score").clip(0, 1)
        data_quality    = _nz("data_quality").clip(0, 1)

        base = (
            0.30 * Proba_Max +
            0.20 * top2_margin +
            0.20 * proximity_score +
            0.15 * local_mass +
            0.10 * dispersion +
            0.05 * data_quality
        )
        df["trust_score_v2"] = base

        # coerce numeric
        prob_like_cols = [
            *[c for c in df.columns if c.startswith("proba_")],
            "Proba_Max", "top2_proba", "top2_margin",
            "proximity_score", "local_mass_pm1", "dispersion_score",
            "entropy_bits", "cal_p1", "data_quality", "trust_score_v2", "entropy_norm"
        ]
        prob_like_cols = [c for c in prob_like_cols if c in df.columns]
        for c in prob_like_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # sort
        if "trust_score_v2" in df.columns:
            df.sort_values(
                by=["trust_score_v2", "Proba_Max", "top2_margin"],
                ascending=[False, False, False],
                inplace=True,
                kind="mergesort"
            )
        elif "Proba_Max" in df.columns:
            df.sort_values(by="Proba_Max", ascending=False, inplace=True)

        # bands
        df["trust_band"] = pd.cut(
            df["trust_score_v2"],
            bins=[0, 0.5, 0.75, 0.90, 1.0],
            labels=["Low", "Med", "High", "Very High"],
            include_lowest=True
        )

        return df

    def _order_with_gaps_for_export(df: pd.DataFrame):
        probas_sorted = _sorted_proba_cols(df)
        groups = [
            ["epic_key", "epic_status", "model_name", "last_prediction_at",
             "predicted_lead_time", "trust_score_v2", "trust_band"],
            ["Proba_Max", "top2_proba", "top2_margin"],
            probas_sorted,
            ["data_quality", "proximity_score", "local_mass_pm1", "dispersion_score", "entropy_norm"],
        ]
        return _build_with_gaps(df, groups)

    def _write_formatted_workbook(path_out: str, df_for_export: pd.DataFrame):
        """Write a single workbook (Predictions + Insights) with all formatting."""
        df_out, gap_idx, gap_names = _order_with_gaps_for_export(df_for_export)

        with pd.ExcelWriter(path_out, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Predictions")
            wb = writer.book
            ws = writer.sheets["Predictions"]

            percent_fmt = wb.add_format({"num_format": "0.00%"})
            header_fmt  = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
            cell_fmt    = wb.add_format({"border": 1})
            gap_head    = wb.add_format({"border": 0})
            gap_cell    = wb.add_format({"border": 0})

            # headers + widths (respect gaps)
            for j, col in enumerate(df_out.columns):
                if col in gap_names:
                    ws.write(0, j, "", gap_head)
                    ws.set_column(j, j, 2, gap_cell)
                else:
                    ws.write(0, j, col, header_fmt)
                    max_len = max(len(str(col)), *(len(str(v)) for v in df_out[col].astype(str).fillna("")))
                    ws.set_column(j, j, min(max_len + 2, 60), cell_fmt)

            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(df_out), len(df_out.columns) - 1)
            ws.hide_gridlines(2)
            ws.set_zoom(120)

            # % formatting + color scales
            nrows = len(df_out)
            pct_cols = [c for c in df_out.columns if c not in gap_names and (
                c.startswith("proba_") or c in [
                    "Proba_Max","top2_proba","top2_margin",
                    "proximity_score","local_mass_pm1","dispersion_score",
                    "entropy_norm",
                    "data_quality","trust_score_v2"
                ]
            )]
            for col in pct_cols:
                cidx = df_out.columns.get_loc(col)
                ws.set_column(cidx, cidx, None, percent_fmt)
                if col == "entropy_norm":
                    ws.conditional_format(1, cidx, nrows, cidx, {
                        "type": "3_color_scale",
                        "min_type": "num", "min_value": 0.0,  "min_color": "#388E3C",
                        "mid_type": "num", "mid_value": 0.5,  "mid_color": "#FFF59D",
                        "max_type": "num", "max_value": 1.0,  "max_color": "#D32F2F",
                    })
                else:
                    ws.conditional_format(1, cidx, nrows, cidx, {
                        "type": "3_color_scale",
                        "min_type": "num", "min_value": 0.0,  "min_color": "#D32F2F",
                        "mid_type": "num", "mid_value": 0.5,  "mid_color": "#FFF59D",
                        "max_type": "num", "max_value": 0.95, "max_color": "#388E3C",
                    })

            # trust_band column styling + header no border
            if "trust_band" in df_out.columns:
                c_band = df_out.columns.get_loc("trust_band")
                band_text_fmt = wb.add_format({"align": "center", "valign": "vcenter"})
                header_no_border = wb.add_format({"bold": True, "bg_color": "#F2F2F2"})
                ws.write(0, c_band, "trust_band", header_no_border)
                ws.set_column(c_band, c_band, 14, band_text_fmt)

                band_colors = {
                    "Low": "#D32F2F",
                    "Med": "#FFBB00",
                    "High": "#7BFF00",
                    "Very High": "#388E3C",
                }
                for label, color in band_colors.items():
                    fmt = wb.add_format({"bg_color": color, "align": "center", "valign": "vcenter"})
                    ws.conditional_format(1, c_band, nrows, c_band, {
                        "type": "cell", "criteria": "==", "value": f'"{label}"', "format": fmt
                    })

            # ===================== INSIGHTS (recomputed on df_for_export) =====================
            ws2 = wb.add_worksheet("Insights")
            ws2.hide_gridlines(2)
            ws2.set_zoom(120)

            ws2.set_column("A:A", 20)
            ws2.set_column("B:B", 10)
            ws2.set_column("C:C", 2)
            ws2.set_column("E:K", 14)
            ws2.set_column("L:L", 46)
            ws2.set_column("P:V", 14)
            ws2.set_column("W:W", 46)

            header_tbl = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})

            def write_table(start_row, title, series):
                ws2.write(start_row, 0, title, header_tbl)
                ws2.write(start_row, 1, "Count", header_tbl)
                for i, (k, v) in enumerate(series.items(), start=1):
                    ws2.write(start_row + i, 0, str(k))
                    ws2.write_number(start_row + i, 1, int(v))
                return start_row + len(series) + 2

            def style_chart(chart, title, x_name, y_name):
                chart.set_title({"name": title, "name_font": {"bold": True}})
                chart.set_x_axis({"name": x_name, "name_font": {"bold": True}, "num_font": {"size": 9}})
                chart.set_y_axis({"name": y_name, "name_font": {"bold": True}, "num_font": {"size": 9}})
                chart.set_legend({"position": "bottom", "font": {"size": 9}})
                chart.set_style(10)

            def note_box_at(row, col_letter, title, lines):
                text = "How to read\n" + title + "\n" + "\n".join(lines)
                _, col_idx = xl_cell_to_rowcol(f"{col_letter}1")
                ws2.insert_textbox(row, col_idx, text, {
                    "width": 420, "height": 130, "x_offset": 6, "y_offset": 2,
                    "font": {"size": 9}, "fill": {"color": "#FFFFF0"}, "line": {"color": "#CCCCCC"}
                })

            df = df_for_export
            proba_cols = [c for c in df.columns if str(c).startswith("proba_")]

            r = 1
            class_counts = df["predicted_lead_time"].value_counts(dropna=False).sort_index()
            r = write_table(r, "Lead Time Class", class_counts)

            # trust_score_v2 buckets
            bins   = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            labels = ["0–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
            proba_max_counts = (pd.cut(df["trust_score_v2"], bins=bins, labels=labels, include_lowest=True)
                                .value_counts().reindex(labels, fill_value=0))
            r = write_table(r, "trust_score_v2 Bucket", proba_max_counts)

            if proba_cols:
                arr = df[proba_cols].to_numpy(dtype=float)
                top2 = np.partition(arr, -2, axis=1)[:, -2:]
                p2 = top2[:, 0]; p1 = top2[:, 1]
                top2_margin = p1 - p2

                m_bins   = [0, 0.05, 0.10, 0.20, 0.30, 1.00]
                m_labels = ["<5pp", "5–10pp", "10–20pp", "20–30pp", "≥30pp"]
                m_counts = (pd.cut(top2_margin, bins=m_bins, labels=m_labels, include_lowest=True)
                            .value_counts().reindex(m_labels, fill_value=0))
                r = write_table(r, "Top-2 Margin", m_counts)

                K = max(1, len(proba_cols))
                e_bins   = np.linspace(0, math.log2(K), 7)
                e_labels = [f"{e_bins[i]:.2f}–{e_bins[i+1]:.2f}" for i in range(len(e_bins)-1)]
                def row_entropy(row):
                    vals = row[row > 0]
                    return float(-np.sum(vals * np.log2(vals))) if len(vals) else 0.0
                ent = np.apply_along_axis(row_entropy, 1, arr)
                e_counts = (pd.cut(ent, bins=e_bins, labels=e_labels, include_lowest=True)
                            .value_counts().reindex(e_labels, fill_value=0))
                r = write_table(r, "Entropy (bits)", e_counts)

            if "epic_status" in df.columns:
                by_status = df.groupby("epic_status", dropna=False)["trust_score_v2"].mean().sort_values(ascending=False)
                ws2.write(r, 0, "Epic Status", header_tbl); ws2.write(r, 1, "Avg trust_score_v2", header_tbl)
                for i, (st, val) in enumerate(by_status.items(), start=1):
                    ws2.write(r + i, 0, str(st)); ws2.write_number(r + i, 1, float(val))
                r += len(by_status) + 2

            anchors = {"col1": {"r1": (1, 4), "r2": (19, 4), "r3": (37, 4)},
                       "col2": {"r1": (1, 15), "r2": (19, 15), "r3": (37, 15)}}

            ch1 = wb.add_chart({"type": "column"})
            ch1.add_series({
                "name": "Predicted Lead Time Distribution",
                "categories": ["Insights", 2, 0, 1 + len(class_counts), 0],
                "values":     ["Insights", 2, 1, 1 + len(class_counts), 1],
                "fill": {"color": "#3A8BFF"},
            })
            style_chart(ch1, "Predicted Lead Time Distribution", "Lead Time Class", "Count")
            ws2.insert_chart(*anchors["col1"]["r1"], ch1, {"x_scale": 1.2, "y_scale": 1.0})
            note_box_at(anchors["col1"]["r1"][0], "L",
                        "Predicted Lead Time Distribution",
                        ["Count per predicted class.", "Taller bar ⇒ more items.",
                         "Insight: detect class imbalance/bias."])

            ch1b = wb.add_chart({"type": "pie"})
            ch1b.add_series({
                "name": "Class Proportions",
                "categories": ["Insights", 2, 0, 1 + len(class_counts), 0],
                "values":     ["Insights", 2, 1, 1 + len(class_counts), 1],
                "data_labels": {"percentage": True},
            })
            ch1b.set_title({"name": "Class Proportions"})
            ws2.insert_chart(*anchors["col1"]["r2"], ch1b, {"x_scale": 1.0, "y_scale": 1.0})
            note_box_at(anchors["col1"]["r2"][0], "L",
                        "Class Proportions (Pie)",
                        ["Share of each predicted class.", "Bigger slice ⇒ more predictions.",
                         "Insight: quick non-technical snapshot."])

            ch2 = wb.add_chart({"type": "column"})
            ch2.add_series({
                "name": "trust_score_v2 Distribution",
                "categories": ["Insights", 2 + len(class_counts) + 2, 0,
                               1 + len(class_counts) + 2 + len(proba_max_counts), 0],
                "values":     ["Insights", 2 + len(class_counts) + 2, 1,
                               1 + len(class_counts) + 2 + len(proba_max_counts), 1],
                "fill": {"color": "#7EA6FF"},
            })
            style_chart(ch2, "trust_score_v2 Distribution", "trust_score_v2 Range", "Count")
            ws2.insert_chart(*anchors["col1"]["r3"], ch2, {"x_scale": 1.2, "y_scale": 1.0})
            note_box_at(anchors["col1"]["r3"][0], "L",
                        "trust_score_v2 Distribution",
                        ["Count by trust_score_v2 bucket.", "More on right ⇒ model often confident.",
                         "Insight: many 60–70% ⇒ review needed."])

            if [c for c in df.columns if c.startswith("proba_")]:
                base_row_margin = 2 + len(class_counts) + 2 + len(proba_max_counts) + 2
                m_counts = ws2.table  # placeholder to satisfy linter, not used

                # recompute margin buckets for anchors
                arr = df[[c for c in df.columns if c.startswith("proba_")]].to_numpy(dtype=float)
                top2 = np.partition(arr, -2, axis=1)[:, -2:]
                m = top2[:, 1] - top2[:, 0]
                m_bins   = [0, 0.05, 0.10, 0.20, 0.30, 1.00]
                m_labels = ["<5pp", "5–10pp", "10–20pp", "20–30pp", "≥30pp"]
                m_counts = (pd.cut(m, bins=m_bins, labels=m_labels, include_lowest=True)
                            .value_counts().reindex(m_labels, fill_value=0))

                # write temporary table for margin
                r0 = base_row_margin
                ws2.write(r0, 0, "Top-2 Margin", header_tbl); ws2.write(r0, 1, "Count", header_tbl)
                for i, (lab, val) in enumerate(m_counts.items(), start=1):
                    ws2.write(r0 + i, 0, lab); ws2.write_number(r0 + i, 1, int(val))

                ch3 = wb.add_chart({"type": "column"})
                ch3.add_series({
                    "name": "Ambiguity (Top-2 Margin)",
                    "categories": ["Insights", r0 + 1, 0, r0 + len(m_counts), 0],
                    "values":     ["Insights", r0 + 1, 1, r0 + len(m_counts), 1],
                    "fill": {"color": "#C5D5F9"},
                })
                style_chart(ch3, "Ambiguity (Top-2 Margin)", "Margin (pp)", "Count")
                ws2.insert_chart(*anchors["col2"]["r1"], ch3, {"x_scale": 1.2, "y_scale": 1.0})
                note_box_at(anchors["col2"]["r1"][0], "W",
                            "Ambiguity (Top-2 Margin)",
                            ["Gap between top-1 and top-2 probabilities.",
                             "Small (<5pp) ⇒ undecided between classes.",
                             "Insight: prime for triage/active learning."])

                # entropy buckets table + chart
                K = max(1, len([c for c in df.columns if c.startswith("proba_")]))
                e_bins   = np.linspace(0, math.log2(K), 7)
                e_labels = [f"{e_bins[i]:.2f}–{e_bins[i+1]:.2f}" for i in range(len(e_bins)-1)]
                def row_entropy(row):
                    vals = row[row > 0]
                    return float(-np.sum(vals * np.log2(vals))) if len(vals) else 0.0
                ent = np.apply_along_axis(row_entropy, 1, arr)
                e_counts = (pd.cut(ent, bins=e_bins, labels=e_labels, include_lowest=True)
                            .value_counts().reindex(e_labels, fill_value=0))
                r1 = r0 + len(m_counts) + 2
                ws2.write(r1, 0, "Entropy (bits)", header_tbl); ws2.write(r1, 1, "Count", header_tbl)
                for i, (lab, val) in enumerate(e_counts.items(), start=1):
                    ws2.write(r1 + i, 0, lab); ws2.write_number(r1 + i, 1, int(val))

                ch4 = wb.add_chart({"type": "column"})
                ch4.add_series({
                    "name": "Prediction Entropy (Uncertainty)",
                    "categories": ["Insights", r1 + 1, 0, r1 + len(e_counts), 0],
                    "values":     ["Insights", r1 + 1, 1, r1 + len(e_counts), 1],
                    "fill": {"color": "#B650FF"},
                })
                style_chart(ch4, "Prediction Entropy (Uncertainty)", "Entropy (bits)", "Count")
                ws2.insert_chart(*anchors["col2"]["r2"], ch4, {"x_scale": 1.2, "y_scale": 1.0})
                note_box_at(anchors["col2"]["r2"][0], "W",
                            "Prediction Entropy (Uncertainty)",
                            ["Uncertainty across all classes.",
                             "Low ⇒ one class dominates; high ⇒ spread.",
                             "Insight: flags weak features/hard items."])

            if "epic_status" in df.columns:
                # average chart
                by_status = df.groupby("epic_status", dropna=False)["trust_score_v2"].mean().sort_values(ascending=False)
                start = r - (len(by_status) + 2)
                ch5 = wb.add_chart({"type": "column"})
                ch5.add_series({
                    "name": "Average trust_score_v2 by Epic Status",
                    "categories": ["Insights", start + 1, 0, start + len(by_status), 0],
                    "values":     ["Insights", start + 1, 1, start + len(by_status), 1],
                    "fill": {"color": "#3A8BFF"},
                })
                style_chart(ch5, "Average trust_score_v2 by Epic Status", "Epic Status", "Avg trust_score_v2")
                ws2.insert_chart(37, 15, ch5, {"x_scale": 1.2, "y_scale": 1.0})
                note_box_at(37, "W",
                            "Average Proba_Max by Epic Status",
                            ["Mean trust_score_v2 per workflow status.",
                             "Compare bars to spot strong/weak contexts.",
                             "Insight: if 'Open' ≪ 'Done' ⇒ enrich early features."])

            # Trust band proportions (pie)
            if "trust_band" in df.columns:
                trust_order = ["Low", "Med", "High", "Very High"]
                trust_colors = {
                    "Low":       "#D32F2F",
                    "Med":       "#FFBB00",
                    "High":      "#7BFF00",
                    "Very High": "#388E3C",
                }
                trust_counts = (df["trust_band"]
                                .value_counts(dropna=False)
                                .reindex(trust_order, fill_value=0))
                r_tb = 55
                # table
                ws2.write(r_tb, 0, "Trust Band", header_tbl); ws2.write(r_tb, 1, "Count", header_tbl)
                for i, (lab, val) in enumerate(trust_counts.items(), start=1):
                    ws2.write(r_tb + i, 0, lab); ws2.write_number(r_tb + i, 1, int(val))

                # pie
                points = [{"fill": {"color": trust_colors[label]}} for label in trust_order]
                ch_tb = wb.add_chart({"type": "pie"})
                ch_tb.add_series({
                    "name":       "Trust Band Proportions",
                    "categories": ["Insights", r_tb + 1, 0, r_tb + len(trust_counts), 0],
                    "values":     ["Insights", r_tb + 1, 1, r_tb + len(trust_counts), 1],
                    "data_labels": {"percentage": True},
                    "points": points,
                })
                ch_tb.set_title({"name": "Trust Band Proportions"})
                ws2.insert_chart(55, 15, ch_tb, {"x_scale": 1.0, "y_scale": 1.0})
                note_box_at(55, "W",
                    "Trust Band Proportions",
                    [
                        "Share of predictions by trust band.",
                        "Bigger slice ⇒ more items in that confidence bucket.",
                        "Many 'Low' items may indicate weak features or drift.",
                    ],
                )

        return path_out

    def _now_paris_iso():
        # Europe/Paris localized → write as ISO string (naive display in Excel)
        # Avoid pytz dependency: get local time by pandas with tz then convert to naive string
        ts = pd.Timestamp.now(tz="Europe/Paris")
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    def _pick_master_keys(cols, user_keys):
        if user_keys:
            return [k for k in user_keys if k in cols]
        # default heuristic
        candidates = []
        if "epic_key" in cols: candidates.append("epic_key")
        if "model_name" in cols: candidates.append("model_name")
        # If only one exists, still okay; upsert will be on that single column
        return candidates if candidates else None

    def _upsert_concat(existing: pd.DataFrame, new: pd.DataFrame, keys: list[str] | None):
        if existing is None or len(existing) == 0:
            return new.copy()
        if keys and len(keys) > 0 and all(k in new.columns for k in keys) and all(k in existing.columns for k in keys):
            # keep the most recent version of duplicates (from 'new')
            merged = pd.concat([existing, new], axis=0, ignore_index=True)
            # drop duplicates keeping last occurrence
            merged = merged.drop_duplicates(subset=keys, keep="last")
            return merged
        # no usable keys: append and drop full-row duplicates
        merged = pd.concat([existing, new], axis=0, ignore_index=True)
        merged = merged.drop_duplicates(keep="last")
        return merged

    # ============================== BUILD THIS BATCH ==============================
    # stamp model & timestamp now
    model_name = model_name or "default_model"
    df_batch = df_result.copy()
    df_batch["model_name"] = model_name
    df_batch["last_prediction_at"] = _now_paris_iso()

    # compute metrics for this batch
    df_scored = _compute_metrics(df_batch, raw_features)

    # write the per-batch Excel (original behavior, but now includes model/timestamp)
    _write_formatted_workbook(path, df_scored)

    # ============================== UPDATE MASTER (optional) ==============================
    if update_master and master_path:
        # Load old master (values only); if absent, start fresh
        if os.path.exists(master_path):
            try:
                # Read the main data from 'Predictions' sheet (works even if columns evolve)
                df_old = pd.read_excel(master_path, sheet_name="Predictions")
            except Exception:
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()

        # Recompute metrics for df_scored? Already computed above; but ensure union of columns with old
        master_keys = _pick_master_keys(
            cols=set(df_scored.columns) | set(df_old.columns),
            user_keys=master_key_cols
        )

        # upsert new rows into master
        df_master_raw = _upsert_concat(df_old, df_scored, master_keys)

        # Recompute metrics/bands on the full master (safe: metrics function handles existing columns)
        # NOTE: Metrics may already exist in df_old; recomputing keeps them consistent.
        df_master_scored = _compute_metrics(df_master_raw, raw_features_input=None)

        # Write a fully formatted, fresh master workbook (keeps formatting & insights current)
        _write_formatted_workbook(master_path, df_master_scored)

    return path
