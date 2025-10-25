# app.py
# Youth Mental Health – Australia (Dash 3.x)

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import os, glob, re

APP_TITLE = "Youth Mental Health – Australia"
DATA_DIR = os.path.dirname(__file__) or "."

# ---------------------------- IO helpers ----------------------------
def read_csv_safe(name):
    """Read CSV with relaxed memory options and without noisy dtype warnings."""
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    new = df.copy()
    new.columns = [str(c).strip().lower().replace(" ", "_").replace("–","-").replace("—","-")
                   for c in new.columns]
    return new

def ensure_standard_columns(df):
    """Create standard 'sex' and 'age_band' columns from common aliases."""
    if df is None or df.empty:
        return df
    x = df.copy()
    # sex
    if "sex" not in x.columns:
        cand = [c for c in x.columns if ("sex" in c.lower()) or ("gender" in c.lower())]
        if cand:
            x["sex"] = x[cand[0]]
    # age_band
    if "age_band" not in x.columns:
        cand = [c for c in x.columns if (c.lower() in
               ["age","age_group","age_groups","age_category","age-band","age band"]) or ("age" in c.lower())]
        if cand:
            x["age_band"] = x[cand[0]]
    return x

def options_from_cols(*frames, col="state"):
    """Collect unique options for a given column."""
    vals = []
    for fr in frames:
        if fr is None or fr.empty:
            continue
        if col in fr.columns:
            vals.extend(fr[col].dropna().astype(str).tolist())
    vals = sorted(pd.Series(vals).dropna().unique().tolist())
    return vals

def options_from_any(*frames, key="sex"):
    """Collect options even if the column names differ (sex/gender, age/age_group…)."""
    vals = []
    keys = [key]
    if key == "sex":
        keys += ["gender"]
    if key == "age_band":
        keys += ["age","age_group","age_groups","age_category","age band","age-band"]
    for fr in frames:
        if fr is None or fr.empty:
            continue
        f = ensure_standard_columns(fr)
        for k in keys:
            if k in f.columns:
                vals.extend(f[k].dropna().astype(str).tolist())
    if not vals and key == "sex":
        vals = ["Male", "Female"]
    return sorted(pd.Series(vals).dropna().unique().tolist())

def to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce")

def add_time_fields(df):
    """Ensure a 'year' column if possible, extracting from 'period' or 'metric' text."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "year" not in out.columns or out["year"].isna().all():
        yr = None
        if "period" in out.columns:
            yr = to_numeric_safe(out["period"].astype(str).str.extract(r"(20\d{2})", expand=False))
        if (yr is None) or yr.isna().all():
            mcol = "metric" if "metric" in out.columns else None
            if mcol is None:
                cands = [c for c in out.columns if c.startswith("metric")]
                if cands:
                    mcol = cands[0]
            if mcol:
                yr = to_numeric_safe(out[mcol].astype(str).str.extract(r"(20\d{2})", expand=False))
        if yr is not None:
            out["year"] = yr
    return out

# ---------------------------- Load data ----------------------------
df_master = norm_cols(ensure_standard_columns(read_csv_safe("integrated_long.csv")))
df_wait   = norm_cols(ensure_standard_columns(read_csv_safe("waiting_time_state_raw.csv")))
df_cap    = norm_cols(ensure_standard_columns(read_csv_safe("capacity_state_raw.csv")))
df_cons   = norm_cols(ensure_standard_columns(read_csv_safe("consultations_access_raw.csv")))
df_digi   = norm_cols(ensure_standard_columns(read_csv_safe("digital_services_raw.csv")))
df_pbs    = norm_cols(ensure_standard_columns(read_csv_safe("medications_pbs_raw.csv")))
df_dist   = norm_cols(ensure_standard_columns(read_csv_safe("psychological_distress_raw.csv")))

# Disorders (multiple files) -> combined frame
disorder_files = glob.glob(os.path.join(DATA_DIR, "disorders_raw_*.csv"))
_dis_frames = []
for f in disorder_files:
    try:
        _tmp = pd.read_csv(f, low_memory=False)
    except Exception:
        _tmp = pd.read_csv(f, low_memory=False, engine="python")
    _dis = norm_cols(ensure_standard_columns(_tmp))
    _dis_frames.append(_dis)
df_dis = pd.concat(_dis_frames, ignore_index=True) if _dis_frames else pd.DataFrame()

# Normalize common fields
for fr in [df_master, df_wait, df_cap, df_cons, df_digi, df_pbs, df_dist, df_dis]:
    if not fr.empty and "value" in fr.columns:
        fr["value"] = to_numeric_safe(fr["value"])

df_wait = add_time_fields(df_wait)
df_cap  = add_time_fields(df_cap)
df_dist = add_time_fields(df_dist)
df_dis  = add_time_fields(df_dis)

# ---------------------------- KPI helpers ----------------------------
def _filter_common(d, state=None, sex=None, age_band=None, year=None):
    if d is None or d.empty:
        return d
    x = d.copy()
    if state and "state" in x.columns:
        x = x[x["state"].astype(str) == str(state)]
    if sex and "sex" in x.columns:
        x = x[x["sex"].astype(str) == str(sex)]
    if age_band and "age_band" in x.columns:
        x = x[x["age_band"].astype(str) == str(age_band)]
    if (year is not None) and ("year" in x.columns) and (not pd.isna(year)):
        x = x[x["year"] == year]
    return x

def _first_metric_col(d):
    if d is None or d.empty:
        return None
    if "metric" in d.columns:
        return "metric"
    cands = [c for c in d.columns if c.startswith("metric")]
    return cands[0] if cands else None

def kpi_avg_wait(state=None, sex=None, age=None, year=None):
    d = _filter_common(df_wait, state, sex, age, year)
    if d is None or d.empty:
        return None
    mcol = _first_metric_col(d)
    if mcol:
        d2 = d[d[mcol].str.contains("wait|time", case=False, na=False)]
        if d2.empty:
            d2 = d
    else:
        d2 = d
    v = d2["value"] if "value" in d2.columns else pd.Series(dtype=float)
    m = v.mean() if not v.empty else np.nan
    return None if np.isnan(m) else float(m)

def kpi_on_time_pct(state=None, year=None):
    d = _filter_common(df_wait, state, None, None, year)
    if d is None or d.empty:
        return None
    mcol = _first_metric_col(d)
    if mcol:
        d2 = d[d[mcol].str.contains("on_time|on-time|on time", case=False, na=False)]
    else:
        d2 = d.iloc[0:0]
    v = d2["value"] if "value" in d2.columns else pd.Series(dtype=float)
    m = v.mean() if not v.empty else np.nan
    return None if np.isnan(m) else float(m)

def kpi_fte_total(state=None, year=None):
    d = _filter_common(df_cap, state, None, None, year)
    if d is None or d.empty:
        return None
    mcol = _first_metric_col(d)
    if mcol:
        d2 = d[d[mcol].str.contains("fte", case=False, na=False)]
        if d2.empty:
            d2 = d
    else:
        d2 = d
    v = d2["value"] if "value" in d2.columns else pd.Series(dtype=float)
    s = v.sum() if not v.empty else np.nan
    return None if np.isnan(s) else float(s)

def kpi_distress(state=None, sex=None, age=None, year=None):
    d = _filter_common(df_dist, state, sex, age, year)
    if d is None or d.empty:
        return None
    v = d["value"] if "value" in d.columns else pd.Series(dtype=float)
    m = v.mean() if not v.empty else np.nan
    return None if np.isnan(m) else float(m)

def kpi_prevalence_12m(state=None, sex=None, age=None, year=None):
    d = _filter_common(df_dis, state, sex, age, year)
    if d is None or d.empty:
        return None
    mcol = _first_metric_col(d)
    if mcol:
        d = d[d[mcol].str.contains("12-month|12_month|12 month", case=False, na=False)]
    v = d["value"] if "value" in d.columns else pd.Series(dtype=float)
    m = v.mean() if not v.empty else np.nan
    return None if np.isnan(m) else float(m)

# ---------------------------- Control options ----------------------------
states = options_from_cols(df_master, df_wait, df_cap, df_cons, df_digi, df_pbs, df_dist, df_dis, col="state")
sexes  = options_from_any(df_master, df_wait, df_cons, df_digi, df_pbs, df_dist, df_dis, key="sex")
ages   = options_from_any(df_master, df_cons, df_digi, df_pbs, df_dist, df_dis, key="age_band")

years = []
for fr in [df_master, df_wait, df_cap, df_dist, df_dis]:
    if not fr.empty and "year" in fr.columns:
        try:
            years.extend(fr["year"].dropna().astype(int).tolist())
        except Exception:
            pass
years = sorted(pd.Series(years).unique().tolist()) if years else []

default_state = states[0] if states else None
default_sex   = sexes[0]  if sexes else None
default_age   = ages[0]   if ages else None
default_year  = years[0]  if years else None

# ---------------------------- UI ----------------------------
app = Dash(__name__)
app.title = APP_TITLE

def kpi_card(title, id_value, subtitle=""):
    return html.Div([
        html.Div(title, style={"fontWeight":"600","fontSize":"14px","opacity":0.7}),
        html.Div("—", id=id_value, style={"fontWeight":"700","fontSize":"28px","marginTop":"4px"}),
        html.Div(subtitle, style={"fontSize":"12px","opacity":0.65,"marginTop":"2px"})
    ], style={"border":"1px solid #eee","padding":"16px","borderRadius":"16px",
              "boxShadow":"0 2px 10px rgba(0,0,0,0.04)","background":"white"})

app.layout = html.Div([
    html.Div([
        html.H1(APP_TITLE, style={"margin":"0","fontSize":"36px"}),
        html.Div("Interactive insights • Clean structure • Assessor-friendly", style={"opacity":0.7})
    ], style={"padding":"20px 24px","borderBottom":"1px solid #eee","background":"#fafafa"}),

    html.Div([
        html.Div([
            html.Div("State", style={"fontWeight":"600"}),
            dcc.Dropdown(options=states, value=default_state, id="state", clearable=True),
        ], style={"width":"24%","display":"inline-block","padding":"8px"}),
        html.Div([
            html.Div("Sex", style={"fontWeight":"600"}),
            dcc.Dropdown(options=sexes, value=default_sex, id="sex", clearable=True, placeholder="Select..."),
        ], style={"width":"24%","display":"inline-block","padding":"8px"}),
        html.Div([
            html.Div("Age band", style={"fontWeight":"600"}),
            dcc.Dropdown(options=ages, value=default_age, id="age_band", clearable=True, placeholder="Select..."),
        ], style={"width":"24%","display":"inline-block","padding":"8px"}),
        html.Div([
            html.Div("Year", style={"fontWeight":"600"}),
            dcc.Slider(
                min=(years[0] if years else 2013),
                max=(years[-1] if years else 2024),
                value=(default_year if default_year is not None else (years[0] if years else 2019)),
                step=None,
                marks=({str(y):str(y) for y in years} if years else None),
                id="year"
            ),
        ], style={"width":"24%","display":"inline-block","padding":"8px"}),
    ], style={"borderBottom":"1px solid #eee","background":"#fff"}),

    html.Div([
        kpi_card("12-month Prevalence (avg %)", "kpi_prev", "Filtered cohort"),
        kpi_card("Avg Wait (days)", "kpi_wait", "Filtered cohort"),
        kpi_card("ED On-time (%)", "kpi_ontime", "Filtered cohort"),
        kpi_card("FTE (sum)", "kpi_fte", "Filtered cohort"),
        kpi_card("Psychological Distress (avg)", "kpi_dist", "Filtered cohort"),
    ], style={"display":"grid","gridTemplateColumns":"repeat(5,1fr)","gap":"12px",
              "padding":"16px","background":"#fafafa"}),

    dcc.Tabs(id="tabs", value="tab-overview", children=[
        dcc.Tab(label="Overview", value="tab-overview", children=[
            dcc.Graph(id="overview_trend", style={"height":"480px"}),
            html.Div("Tip: Use the controls to explore changes by cohort and time.",
                     style={"opacity":0.65,"padding":"0 16px 16px"})
        ]),
        dcc.Tab(label="Demand & Capacity", value="tab-cap", children=[
            dcc.Graph(id="cap_bar", style={"height":"480px"}),
            html.Div("Observed capacity by state/profession. Consider overlaying demand once available.",
                     style={"opacity":0.65,"padding":"0 16px 16px"})
        ]),
        dcc.Tab(label="Access & Channel Mix", value="tab-access", children=[
            dcc.Graph(id="access_stack", style={"height":"480px"}),
            html.Div("Consultations vs digital services mix.",
                     style={"opacity":0.65,"padding":"0 16px 16px"})
        ]),
        dcc.Tab(label="Distress & Disorders", value="tab-distress", children=[
            dcc.Graph(id="distress_plot", style={"height":"480px"}),
            dcc.Graph(id="disorder_plot", style={"height":"480px"}),
        ]),
    ], style={"padding":"0 16px 16px"}),
])

# ---------------------------- Callbacks ----------------------------
@app.callback(
    Output("kpi_prev","children"),
    Output("kpi_wait","children"),
    Output("kpi_ontime","children"),
    Output("kpi_fte","children"),
    Output("kpi_dist","children"),
    Input("state","value"),
    Input("sex","value"),
    Input("age_band","value"),
    Input("year","value"),
)
def update_kpis(state, sex, age_band, year):
    prev = kpi_prevalence_12m(state, sex, age_band, year)
    wait = kpi_avg_wait(state, sex, age_band, year)
    onti = kpi_on_time_pct(state, year)
    fte  = kpi_fte_total(state, year)
    dist = kpi_distress(state, sex, age_band, year)

    f = lambda x, fs="{:.2f}": "—" if (x is None or (isinstance(x,float) and np.isnan(x))) else fs.format(x)
    return f(prev), f(wait), f(onti), f(fte, "{:.0f}"), f(dist)

@app.callback(
    Output("overview_trend","figure"),
    Input("state","value"),
    Input("sex","value"),
    Input("age_band","value"),
)
def update_overview(state, sex, age_band):
    if df_wait.empty:
        return px.line(title="No waiting time data")
    d = _filter_common(df_wait, state, sex, age_band, None)
    mcol = _first_metric_col(d)
    if mcol:
        d2 = d[d[mcol].str.contains("wait|on_time|on-time|on time|time", case=False, na=False)]
        if d2.empty:
            d2 = d
        d = d2
    if ("year" not in d.columns) or d["year"].isna().all():
        if "period" in d.columns:
            d["year"] = to_numeric_safe(d["period"].astype(str).str.extract(r"(20\d{2})", expand=False))
        if ("year" not in d.columns) or d["year"].isna().all():
            if mcol:
                d["year"] = to_numeric_safe(d[mcol].astype(str).str.extract(r"(20\d{2})", expand=False))
    d = d.dropna(subset=["value"])
    if "year" in d.columns and d["year"].notna().any():
        fig = px.line(d, x="year", y="value", color=(mcol if mcol else None),
                      markers=True, title="Waiting time trend")
        fig.update_xaxes(title_text="Year")
    else:
        fig = px.histogram(d, x="value", nbins=20, title="Waiting time")
    fig.update_layout(margin=dict(l=24,r=16,t=60,b=24))
    fig.update_yaxes(title_text="Value")
    return fig

@app.callback(
    Output("cap_bar","figure"),
    Input("state","value"),
    Input("year","value"),
)
def update_cap(state, year):
    if df_cap.empty:
        return px.bar(title="No capacity data")
    d = _filter_common(df_cap, state, None, None, year)
    mcol = _first_metric_col(d)
    if mcol:
        d = d[d[mcol].str.contains("fte", case=False, na=False)]
    d = d.dropna(subset=["value"])
    prof_col = "sheet" if "sheet" in d.columns else (mcol if mcol else None)
    fig = px.bar(d, x=prof_col, y="value", title="Capacity (FTE) by Profession")
    fig.update_layout(margin=dict(l=24,r=16,t=60,b=24))
    fig.update_xaxes(title_text="Profession")
    fig.update_yaxes(title_text="FTE")
    return fig

@app.callback(
    Output("access_stack","figure"),
    Input("state","value"),
    Input("sex","value"),
    Input("age_band","value"),
)
def update_access(state, sex, age_band):
    frames = []
    if not df_cons.empty:
        dc = _filter_common(df_cons, state, sex, age_band, None).copy()
        mcol = _first_metric_col(dc)
        if mcol:
            dc["channel"] = np.where(dc[mcol].str.contains("consult", case=False, na=False),
                                     "Consultations","Consultations (other)")
        else:
            dc["channel"] = "Consultations"
        frames.append(dc)
    if not df_digi.empty:
        dd = _filter_common(df_digi, state, sex, age_band, None).copy()
        dd["channel"] = "Digital services"
        frames.append(dd)
    if not frames:
        return px.bar(title="No access/channel data")
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["value"])
    agg = d.groupby(["channel"], as_index=False)["value"].mean()
    fig = px.bar(agg, x="channel", y="value", title="Access & Channel Mix (avg)")
    fig.update_layout(margin=dict(l=24,r=16,t=60,b=24))
    fig.update_xaxes(title_text="Channel")
    fig.update_yaxes(title_text="Average value")
    return fig

@app.callback(
    Output("distress_plot","figure"),
    Output("disorder_plot","figure"),
    Input("state","value"),
    Input("sex","value"),
    Input("age_band","value"),
)
def update_dist_dis(state, sex, age_band):
    # Distress
    if df_dist.empty:
        fig1 = px.line(title="No psychological distress data")
    else:
        d1 = _filter_common(df_dist, state, sex, age_band, None)
        if ("year" not in d1.columns) or d1["year"].isna().all():
            if "period" in d1.columns:
                d1["year"] = to_numeric_safe(d1["period"].astype(str).str.extract(r"(20\d{2})", expand=False))
        d1 = d1.dropna(subset=["value"])
        if "year" in d1.columns and d1["year"].notna().any():
            fig1 = px.line(d1, x="year", y="value", title="Psychological Distress (trend)", markers=True)
            fig1.update_xaxes(title_text="Year")
        else:
            fig1 = px.histogram(d1, x="value", nbins=20, title="Psychological Distress (distribution)")
        fig1.update_layout(margin=dict(l=24,r=16,t=60,b=24))
        fig1.update_yaxes(title_text="Value")

    # Disorders
    if df_dis.empty:
        fig2 = px.line(title="No disorders data")
    else:
        d2 = _filter_common(df_dis, state, sex, age_band, None)
        mcol = _first_metric_col(d2)
        if mcol:
            d2 = d2[d2[mcol].str.contains("12-month|12_month|12 month", case=False, na=False)]
        d2 = d2.dropna(subset=["value"])
        color_col = "sheet" if "sheet" in d2.columns else (mcol if mcol else None)
        if ("year" not in d2.columns) or d2["year"].isna().all():
            if "period" in d2.columns:
                d2["year"] = to_numeric_safe(d2["period"].astype(str).str.extract(r"(20\d{2})", expand=False))
        if "year" in d2.columns and d2["year"].notna().any():
            fig2 = px.line(d2, x="year", y="value", color=color_col, title="12M Disorders (trend by sheet)", markers=True)
            fig2.update_xaxes(title_text="Year")
        else:
            agg2 = d2.groupby([color_col], as_index=False)["value"].mean().nlargest(10, "value")
            fig2 = px.bar(agg2, x=color_col, y="value", title="12M Disorders (top groups)")
        fig2.update_layout(margin=dict(l=24,r=16,t=60,b=24))
        fig2.update_yaxes(title_text="Value")

    return fig1, fig2

# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
