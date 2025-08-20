# Portfolio-LAB
# Portfolio LAB
import os, math, warnings, socket
import numpy as np
import pandas as pd
import yfinance as yf

from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
import plotly.graph_objects as go
import plotly.express as px
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format
from pandas.tseries.offsets import DateOffset

from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt

warnings.filterwarnings("ignore")

# ------------ Universe (split UK vs German Pharma) ------------
UK_PHARMA     = ["AZN.L","GSK.L","HIK.L","SN.L"]
GERMAN_PHARMA = ["BAYN.DE","MRK.DE","SHL.DE","FME.DE","FRE.DE","SRT3.DE"]
US_TECH       = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","AVGO","ORCL","AMD","CRM"]
ASIA_BANKS    = ["0005.HK","1398.HK","0939.HK","3988.HK","8306.T","8316.T","8411.T","D05.SI","O39.SI","U11.SI"]

MANUAL_BENCH_CHOICES = {
    "EW All (internal)": "EW_ALL",
    "FTSE 100 (^FTSE)": "^FTSE",
    "DAX (^GDAXI)": "^GDAXI",
    "SPY (S&P 500 ETF)": "SPY",
    "URTH (MSCI World ETF)": "URTH",
    "QQQ (Nasdaq 100 ETF)": "QQQ",
    "^GSPC (S&P 500 Index)": "^GSPC"
}

# ------------------ Helpers & risk functions ------------------
def fetch_prices(tickers, start="2015-01-01"):
    """Download adjusted Close; handle single/multiple tickers."""
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
        px = df["Close"]
    else:
        px = df
    if isinstance(px, pd.Series): px = px.to_frame()
    if px is None or px.empty: return pd.DataFrame()
    px = px.sort_index().dropna(how="all")
    px.columns = [str(c) for c in px.columns]
    return px

def align_prices_equal_length(prices: pd.DataFrame):
    """
    Equalize length on a BusinessDay grid; forward-fill (use most recent prior close).
    Start from the latest first-valid date across tickers (avoid pre-IPO gaps).
    """
    if prices is None or prices.empty: return pd.DataFrame()
    valid_starts = [prices[c].first_valid_index() for c in prices.columns if prices[c].first_valid_index() is not None]
    if not valid_starts: return pd.DataFrame()
    start = max(valid_starts)
    px = prices.loc[start:].copy().dropna(axis=1, how="all")
    if px.empty: return pd.DataFrame()
    bidx = pd.date_range(px.index.min(), px.index.max(), freq="B")
    px = px.reindex(bidx).ffill()
    px = px.dropna(axis=1, how="all")
    return px

def to_returns(px: pd.DataFrame):
    r = px.pct_change()
    r = r.dropna(how="all").fillna(0.0)
    return r

def ew_bench(returns: pd.DataFrame):
    if returns is None or returns.empty: return pd.Series(dtype=float, name="EW_ALL")
    w = np.ones(returns.shape[1]) / max(1, returns.shape[1])
    return (returns @ w).rename("EW_ALL")

def drawdown_curve(r: pd.Series) -> pd.Series:
    nav = (1 + r).cumprod()
    return nav/np.maximum.accumulate(nav) - 1.0

# ---- metrics helpers (numeric) ----
def annualize_return(r: pd.Series, freq=252):
    if len(r)==0: return np.nan
    return float((1+r).prod() ** (freq/len(r)) - 1)

def annualize_vol(r: pd.Series, freq=252):
    if len(r)<2: return np.nan
    return float(r.std(ddof=1) * math.sqrt(freq))

def sharpe_ratio(r: pd.Series, rf_annual=0.0, freq=252):
    if len(r)<2: return np.nan
    rf_daily = (1+rf_annual)**(1/freq) - 1
    ex = r - rf_daily
    vol = ex.std(ddof=1)
    return float(np.nan) if vol==0 or np.isnan(vol) else float(ex.mean()/vol * math.sqrt(freq))

def sortino_ratio(r: pd.Series, rf_annual=0.0, freq=252):
    if len(r)<2: return np.nan
    rf_daily = (1+rf_annual)**(1/freq) - 1
    ex = r - rf_daily
    downside = ex[ex<0].std(ddof=1)
    return float(np.nan) if downside==0 or np.isnan(downside) else float(ex.mean()/downside * math.sqrt(freq))

def max_drawdown(r: pd.Series):
    dd = drawdown_curve(r)
    return float(dd.min()) if len(dd)>0 else np.nan

def ulcer_index(r: pd.Series):
    dd = drawdown_curve(r).values
    return float(np.sqrt(np.mean(dd**2))) if len(dd)>0 else np.nan

def info_ratio(r: pd.Series, rb: pd.Series, freq=252):
    idx = r.index.intersection(rb.index)
    if len(idx)<2: return np.nan
    ex = r.loc[idx] - rb.loc[idx]
    te = ex.std(ddof=1) * math.sqrt(freq)
    ex_ann = ex.mean()*freq
    return float(np.nan) if te==0 or np.isnan(te) else float(ex_ann/te)

def tracking_error(r: pd.Series, rb: pd.Series, freq=252):
    idx = r.index.intersection(rb.index)
    if len(idx)<2: return np.nan
    ex = r.loc[idx] - rb.loc[idx]
    return float(ex.std(ddof=1) * math.sqrt(freq))

def hist_var_es(r: pd.Series, alpha=0.95):
    x = r.dropna().values
    if len(x)==0: return np.nan, np.nan
    x = np.sort(x)
    k = int((1-alpha)*len(x))
    k = max(min(k, len(x)-1), 0)
    var = -x[k]
    es  = -x[:k+1].mean()
    return float(var), float(es)

# --------------- PyPortfolioOpt plumbing ----------------
def build_mu(prices: pd.DataFrame, rf_annual=0.0):
    # Returns ANNUALIZED expected returns
    return expected_returns.mean_historical_return(prices, frequency=252)

def build_cov(prices: pd.DataFrame, method: str):
    # Returns ANNUALIZED covariance matrix
    if method == "Sample":       return risk_models.sample_cov(prices, frequency=252)
    if method == "Ledoit-Wolf":  return risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    if method == "OAS":          return risk_models.CovarianceShrinkage(prices).oas()
    return risk_models.sample_cov(prices, frequency=252)

def optimize_weights(prices: pd.DataFrame, method: str, cov_method: str, rf_annual=0.0):
    cols = list(prices.columns)
    if len(cols) < 2:
        return pd.Series([1.0] if cols else [], index=cols)
    if method == "Equal-Weight":
        return pd.Series(np.ones(len(cols))/len(cols), index=cols)
    if method in ["Min-Variance","Max-Sharpe"]:
        mu, S = build_mu(prices, rf_annual), build_cov(prices, cov_method)
        ef = EfficientFrontier(mu, S, weight_bounds=(0,1))
        if method == "Min-Variance": ef.min_volatility()
        else: ef.max_sharpe(risk_free_rate=rf_annual)
        w = ef.clean_weights()
        return pd.Series({c: w.get(c,0.0) for c in cols})
    if method == "HRP":
        rets = to_returns(prices)
        hrp = HRPOpt(rets)
        w = hrp.optimize()
        return pd.Series({c: w.get(c,0.0) for c in cols})
    return pd.Series(np.ones(len(cols))/len(cols), index=cols)

# --------- Enhanced Efficient Frontier figure ----------
def efficient_frontier_fig(prices: pd.DataFrame, cov_method: str, rf_annual=0.0,
                           current_w: pd.Series=None, show_eqw=True, npts=40):
    fig = go.Figure()
    if prices.shape[1] < 2:
        fig.update_layout(title="Efficient Frontier (need ≥2 assets)")
        return fig

    mu, S = build_mu(prices, rf_annual), build_cov(prices, cov_method)  # annualized
    if np.any(~np.isfinite(S.values)) or np.any(~np.isfinite(mu.values)):
        fig.update_layout(title="Frontier unavailable (non-finite mu/cov)")
        return fig

    # Sweep target returns across mu quantiles
    tmin, tmax = np.nanquantile(mu.values, 0.05), np.nanquantile(mu.values, 0.95)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        fig.update_layout(title="Frontier unavailable (bad target range)")
        return fig

    vols, rets = [], []
    for t in np.linspace(tmin, tmax, npts):
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0,1))
            ef.efficient_return(target_return=float(t))
            w = np.array(list(ef.clean_weights().values()))
            vol = float(np.sqrt(w @ S.values @ w))   # already annualized
            vols.append(vol); rets.append(float(t))
        except Exception:
            continue

    if vols:
        fig.add_trace(go.Scatter(x=vols, y=rets, mode="lines", name="Efficient Frontier"))

    # Min-Variance
    try:
        ef_min = EfficientFrontier(mu, S, weight_bounds=(0,1))
        w_min = np.array(list(ef_min.min_volatility().clean_weights().values()))
        fig.add_trace(go.Scatter(x=[np.sqrt(w_min @ S.values @ w_min)],
                                 y=[w_min @ mu.values], mode="markers",
                                 name="Min-Vol", marker_symbol="square", marker_size=10))
    except Exception:
        pass

    # Max-Sharpe
    try:
        ef_ms = EfficientFrontier(mu, S, weight_bounds=(0,1))
        w_ms = np.array(list(ef_ms.max_sharpe(risk_free_rate=rf_annual).clean_weights().values()))
        fig.add_trace(go.Scatter(x=[np.sqrt(w_ms @ S.values @ w_ms)],
                                 y=[w_ms @ mu.values], mode="markers",
                                 name="Max-Sharpe", marker_symbol="star", marker_size=12))
    except Exception:
        pass

    # Equal-weight
    if show_eqw:
        w_eq = np.ones(len(mu))/len(mu)
        fig.add_trace(go.Scatter(x=[np.sqrt(w_eq @ S.values @ w_eq)],
                                 y=[w_eq @ mu.values], mode="markers",
                                 name="Equal-Weight", marker_symbol="circle", marker_size=9))

    # Current portfolio
    if current_w is not None and len(current_w)>0:
        wv = current_w.reindex(mu.index).fillna(0.0).values
        if wv.sum() > 0:
            fig.add_trace(go.Scatter(x=[np.sqrt(wv @ S.values @ wv)],
                                     y=[wv @ mu.values], mode="markers+text",
                                     name="Current", marker_symbol="diamond", marker_size=12,
                                     text=["Current"], textposition="top center"))
    fig.update_layout(title="Efficient Frontier (annualized)",
                      xaxis_title="Volatility (σ, annualized)",
                      yaxis_title="Expected Return (annualized)")
    return fig

# ---------------------- Metrics table ----------------------
def build_metrics_df(port: pd.Series, bench: pd.Series, rf_annual: float) -> pd.DataFrame:
    idx = port.index.intersection(bench.index)
    r = port.loc[idx].dropna()
    b = bench.loc[idx].dropna()
    met_p = {
        "AnnRet": annualize_return(r),
        "AnnVol": annualize_vol(r),
        "Sharpe": sharpe_ratio(r, rf_annual),
        "Sortino": sortino_ratio(r, rf_annual),
        "Calmar": (annualize_return(r) / abs(max_drawdown(r))) if max_drawdown(r) not in (0, np.nan) else np.nan,
        "Omega":  None if len(r)==0 else float(((r-0).clip(lower=0).sum()) / ((0-r).clip(lower=0).sum() or np.nan)),
        "MaxDD": max_drawdown(r),
        "UlcerIdx": ulcer_index(r),
        "TE": tracking_error(r, b),
        "InfoRatio": info_ratio(r, b),
    }
    var_p, es_p = hist_var_es(r, 0.95)
    met_p["VaR95"] = var_p; met_p["ES95"] = es_p

    met_b = {
        "AnnRet": annualize_return(b),
        "AnnVol": annualize_vol(b),
        "Sharpe": sharpe_ratio(b, rf_annual),
        "Sortino": sortino_ratio(b, rf_annual),
        "Calmar": (annualize_return(b) / abs(max_drawdown(b))) if max_drawdown(b) not in (0, np.nan) else np.nan,
        "Omega":  None if len(b)==0 else float(((b-0).clip(lower=0).sum()) / ((0-b).clip(lower=0).sum() or np.nan)),
        "MaxDD": max_drawdown(b),
        "UlcerIdx": ulcer_index(b),
        "TE": np.nan, "InfoRatio": np.nan,
    }
    var_b, es_b = hist_var_es(b, 0.95)
    met_b["VaR95"] = var_b; met_b["ES95"] = es_b

    df = pd.DataFrame([met_p, met_b], index=["Portfolio","Benchmark"])
    return df

# ----------------- Walk-Forward OOS Backtest -----------------
def _rebalance_dates(index: pd.DatetimeIndex, freq_code: str) -> pd.DatetimeIndex:
    if freq_code == "M":
        return index.to_series().resample("M").last().dropna().index
    if freq_code == "Q":
        return index.to_series().resample("Q").last().dropna().index
    if freq_code == "A":
        return index.to_series().resample("A").last().dropna().index
    return index.to_series().resample("M").last().dropna().index

def walk_forward_oos(prices: pd.DataFrame, optimizer: str, cov_method: str,
                     rf_annual: float, lookback_years: float = 3.0, freq_code: str = "M"):
    if prices.shape[1] < 2:
        return pd.Series(dtype=float), {}
    returns = to_returns(prices)
    rebal = _rebalance_dates(returns.index, freq_code)
    start_cut = returns.index.min() + DateOffset(years=lookback_years)
    rebal = rebal[rebal >= start_cut]
    if len(rebal) == 0:
        return pd.Series(dtype=float), {}
    weights_by_date, oos_parts = {}, []
    for i, t in enumerate(rebal):
        window_start = t - DateOffset(years=lookback_years)
        in_window = (prices.index > window_start) & (prices.index <= t)
        px_win = prices.loc[in_window]
        if px_win.shape[0] < 60:  # at least ~3 months
            continue
        try:
            w = optimize_weights(px_win, optimizer, cov_method, rf_annual).reindex(prices.columns).fillna(0.0)
        except Exception:
            continue
        weights_by_date[pd.Timestamp(t)] = w
        if i < len(rebal) - 1:
            t_next = rebal[i+1]
        else:
            t_next = returns.index.max()
        seg = returns.loc[(returns.index > t) & (returns.index <= t_next)]
        if seg.empty:
            continue
        oos_parts.append(seg @ w.values)
    if not oos_parts:
        return pd.Series(dtype=float), {}
    oos = pd.concat(oos_parts).sort_index().rename("OOS_Portfolio")
    return oos, weights_by_date

# --------------------------- Dash UI ---------------------------
percent2 = FormatTemplate.percentage(2)
ratio2   = Format(precision=2)
ratio3   = Format(precision=3)

metrics_columns = [
    {"name":"Series",   "id":"Series"},
    {"name":"AnnRet%",  "id":"AnnRet",  "type":"numeric", "format": percent2},
    {"name":"AnnVol%",  "id":"AnnVol",  "type":"numeric", "format": percent2},
    {"name":"Sharpe",   "id":"Sharpe",  "type":"numeric", "format": ratio2},
    {"name":"Sortino",  "id":"Sortino", "type":"numeric", "format": ratio2},
    {"name":"Calmar",   "id":"Calmar",  "type":"numeric", "format": ratio2},
    {"name":"Omega",    "id":"Omega",   "type":"numeric", "format": ratio2},
    {"name":"MaxDD%",   "id":"MaxDD",   "type":"numeric", "format": percent2},
    {"name":"UlcerIdx", "id":"UlcerIdx","type":"numeric", "format": ratio3},
    {"name":"TE%",      "id":"TE",      "type":"numeric", "format": percent2},
    {"name":"InfoRatio","id":"InfoRatio","type":"numeric","format": ratio2},
    {"name":"VaR95%",   "id":"VaR95",   "type":"numeric", "format": percent2},
    {"name":"ES95%",    "id":"ES95",    "type":"numeric", "format": percent2},
]

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Portfolio Lab — UK/German split • Downloads • OOS Backtest"),
    html.Div([
        html.Div([
            html.H4("Universe & Data"),
            html.Label("UK Pharma"),
            dcc.Checklist(id="tickers_uk", options=[{"label":t, "value":t} for t in UK_PHARMA], value=UK_PHARMA),
            html.Label("German Pharma"),
            dcc.Checklist(id="tickers_de", options=[{"label":t, "value":t} for t in GERMAN_PHARMA], value=GERMAN_PHARMA),
            html.Label("US Tech"),
            dcc.Checklist(id="tickers_us", options=[{"label":t, "value":t} for t in US_TECH], value=[]),
            html.Label("Asian Banks"),
            dcc.Checklist(id="tickers_asia", options=[{"label":t, "value":t} for t in ASIA_BANKS], value=[]),

            html.Label("Start date"), dcc.Input(id="start_date", type="text", value="2015-01-01"),

            html.Label("Benchmark mode"),
            dcc.RadioItems(id="bench_mode",
                           options=[{"label":"Auto (use ^FTSE for UK, ^GDAXI for German)", "value":"auto"},
                                    {"label":"Manual", "value":"manual"}],
                           value="auto"),
            dcc.Dropdown(id="bench_manual",
                         options=[{"label":k, "value":v} for k,v in MANUAL_BENCH_CHOICES.items()],
                         value="EW_ALL", clearable=False, style={"marginTop":"6px"}),

            html.Br(),
            html.Button("Fetch & Align Data", id="btn_fetch", n_clicks=0, style={"width":"100%"}),
            html.Pre(id="data_info", style={"fontFamily":"monospace","whiteSpace":"pre-wrap","marginTop":"6px"})
        ], style={"flex":"1","padding":"10px","border":"1px solid #ddd","borderRadius":"12px"}),

        html.Div([
            html.H4("Optimization"),
            html.Label("Portfolio Scope"),
            dcc.Dropdown(id="scope", clearable=False,
                         options=[{"label":"Custom (all selected)", "value":"custom"},
                                  {"label":"UK Pharma only", "value":"uk"},
                                  {"label":"German Pharma only", "value":"de"}],
                         value="custom"),
            html.Label("Optimizer"),
            dcc.Dropdown(id="optimizer", clearable=False,
                         options=[{"label":x,"value":x} for x in ["Equal-Weight","Min-Variance","Max-Sharpe","HRP"]],
                         value="Min-Variance"),
            html.Label("Covariance (for MV/Max-Sharpe)"),
            dcc.Dropdown(id="cov_method", clearable=False,
                         options=[{"label":x,"value":x} for x in ["Sample","Ledoit-Wolf","OAS"]],
                         value="Ledoit-Wolf"),
            html.Label("Risk-free (annual, %)"),
            dcc.Input(id="rf_pct", type="number", value=2.0, step=0.1, style={"width":"50%"}),
            html.Br(),
            html.Button("Build Portfolio", id="btn_build", n_clicks=0, style={"width":"100%"}),
            html.Pre(id="opt_info", style={"fontFamily":"monospace","whiteSpace":"pre-wrap","marginTop":"6px"})
        ], style={"flex":"1","padding":"10px","border":"1px solid #ddd","borderRadius":"12px","marginLeft":"12px"})
    ], style={"display":"flex","gap":"12px"}),

    html.Hr(),

    dcc.Tabs(id="tabs", value="tab-metrics", children=[
        dcc.Tab(label="Metrics", value="tab-metrics", children=[
            html.Div([
                html.Button("Download Metrics CSV", id="btn_dl_metrics", n_clicks=0, style={"marginRight":"8px"}),
                dcc.Download(id="dl_metrics"),
                html.Button("Download Weights CSV", id="btn_dl_weights", n_clicks=0),
                dcc.Download(id="dl_weights"),
            ], style={"marginBottom":"8px"}),
            dash_table.DataTable(id="table_metrics", data=[], columns=metrics_columns, page_size=10,
                                 sort_action="native", filter_action="native",
                                 style_table={"overflowX":"auto"})
        ]),
        dcc.Tab(label="Cumulative", value="tab-cum", children=[dcc.Loading(dcc.Graph(id="cum_chart"), type="circle")]),
        dcc.Tab(label="Drawdown", value="tab-dd", children=[dcc.Loading(dcc.Graph(id="dd_chart"), type="circle")]),
        dcc.Tab(label="Efficient Frontier", value="tab-frontier", children=[dcc.Loading(dcc.Graph(id="frontier_chart"), type="circle")]),
        dcc.Tab(label="Weights", value="tab-weights", children=[dcc.Loading(dcc.Graph(id="weights_chart"), type="circle")]),
        dcc.Tab(label="Walk-Forward OOS", value="tab-oos", children=[
            html.Div([
                html.Label("Lookback (years)"),
                dcc.Input(id="oos_lookback_years", type="number", value=3.0, step=0.5, style={"width":"25%","marginRight":"12px"}),
                html.Label("Rebalance"),
                dcc.Dropdown(id="oos_freq", clearable=False,
                             options=[{"label":"Monthly","value":"M"},
                                      {"label":"Quarterly","value":"Q"},
                                      {"label":"Annual","value":"A"}],
                             value="M", style={"width":"30%","display":"inline-block","marginRight":"12px"}),
                html.Button("Run OOS Backtest", id="btn_oos", n_clicks=0, style={"marginTop":"6px"}),
                html.Pre(id="oos_info", style={"fontFamily":"monospace","whiteSpace":"pre-wrap","marginTop":"8px"})
            ], style={"margin":"8px 0"}),
            dash_table.DataTable(id="table_oos_metrics", data=[], columns=metrics_columns, page_size=10,
                                 sort_action="native", filter_action="native",
                                 style_table={"overflowX":"auto", "marginBottom":"10px"}),
            dcc.Loading(dcc.Graph(id="oos_cum_chart"), type="circle"),
            dcc.Loading(dcc.Graph(id="oos_dd_chart"), type="circle"),
        ]),
    ]),

    dcc.Store(id="data_state"),
    dcc.Store(id="portfolio_state")
], style={"maxWidth":"1200px","margin":"10px auto","fontFamily":"Inter,system-ui"})

# ------------------------- Callbacks -------------------------
@app.callback(Output("bench_manual","disabled"), Input("bench_mode","value"))
def toggle_manual_bench(mode): return mode != "manual"

@app.callback(
    Output("data_state","data"),
    Output("data_info","children"),
    Input("btn_fetch","n_clicks"),
    State("tickers_uk","value"), State("tickers_de","value"),
    State("tickers_us","value"), State("tickers_asia","value"),
    State("start_date","value"),
    prevent_initial_call=False
)
def fetch_and_align(n_clicks, uk_sel, de_sel, us_sel, asia_sel, start_date):
    if n_clicks == 0: return no_update, ""
    try:
        tickers = list(dict.fromkeys((uk_sel or []) + (de_sel or []) + (us_sel or []) + (asia_sel or [])))
        if not tickers: return no_update, "Select at least one ticker."
        prices = fetch_prices(tickers, start=start_date)
        if prices.empty: return no_update, "No data fetched. Check tickers/start."
        aligned_prices = align_prices_equal_length(prices)
        if aligned_prices.empty: return no_update, "Alignment failed."
        returns = to_returns(aligned_prices)
        if returns.empty: return no_update, "Could not compute returns."
        info = (f"Selected {len(tickers)} tickers\n"
                f"Aligned grid: {aligned_prices.shape[0]} rows × {aligned_prices.shape[1]} cols\n"
                f"Date range: {aligned_prices.index.min().date()} → {aligned_prices.index.max().date()}")
        return {
            "tickers": tickers,
            "asset_prices": aligned_prices.to_json(date_format="iso", orient="split"),
            "asset_returns": returns.to_json(date_format="iso", orient="split"),
            "uk_list": UK_PHARMA, "de_list": GERMAN_PHARMA
        }, info
    except Exception as e:
        return no_update, f"Error: {e}"

def _subset_by_scope(prices: pd.DataFrame, returns: pd.DataFrame, tickers_selected: list, scope: str,
                     uk_list: list, de_list: list):
    if scope == "uk":   use = [t for t in tickers_selected if t in uk_list]
    elif scope == "de": use = [t for t in tickers_selected if t in de_list]
    else:               use = list(tickers_selected)
    use = [t for t in use if t in prices.columns]
    return prices[use], returns[use], use

def _auto_benchmark_for_scope(scope: str):
    if scope == "uk": return "^FTSE"
    if scope == "de": return "^GDAXI"
    return "EW_ALL"

@app.callback(
    Output("portfolio_state","data"),
    Output("opt_info","children"),
    Input("btn_build","n_clicks"),
    State("data_state","data"),
    State("scope","value"),
    State("optimizer","value"), State("cov_method","value"), State("rf_pct","value"),
    State("bench_mode","value"), State("bench_manual","value"),
    State("start_date","value")
)
def build_portfolio(n_clicks, ds, scope, optimizer, cov_method, rf_pct, bench_mode, bench_manual, start_date):
    if n_clicks == 0 or not ds: return no_update, ""
    try:
        prices_all = pd.read_json(ds["asset_prices"], orient="split")
        rets_all   = pd.read_json(ds["asset_returns"], orient="split")
        tickers_selected = ds["tickers"]
        prices, returns, used = _subset_by_scope(prices_all, rets_all, tickers_selected, scope, ds["uk_list"], ds["de_list"])
        if prices.shape[1] == 0: return no_update, f"No tickers in scope '{scope}'."
        rf_annual = (rf_pct or 0)/100.0

        bench_symbol = _auto_benchmark_for_scope(scope) if bench_mode == "auto" else bench_manual
        if bench_symbol == "EW_ALL":
            bench_ret = ew_bench(returns)
        else:
            bpx_raw = fetch_prices([bench_symbol], start=start_date)
            if bpx_raw.empty:
                bench_ret = ew_bench(returns); bench_ret.name = "EW_ALL (fallback)"
            else:
                bpx = align_prices_equal_length(bpx_raw).reindex(prices.index).ffill()
                bench_ret = bpx.pct_change().fillna(0.0).iloc[:,0]; bench_ret.name = bench_symbol

        w = optimize_weights(prices, optimizer, cov_method, rf_annual).reindex(prices.columns).fillna(0.0)
        idx = returns.index.intersection(bench_ret.index)
        port_ret = (returns.loc[idx] @ w.values).rename("Strategy")
        bench_ret = bench_ret.loc[idx]

        info = (f"Scope: {scope} | Optimizer: {optimizer} | Cov: {cov_method} | RF: {rf_annual:.2%}\n"
                f"Benchmark: {bench_ret.name} | Tickers: {len(used)} | Samples: n={len(idx)}")

        return {
            "scope": scope,
            "optimizer": optimizer,
            "weights_df": w.to_frame("weight").to_json(orient="split"),
            "portfolio_returns_df": port_ret.to_frame("Strategy").to_json(date_format="iso", orient="split"),
            "benchmark_returns_df": bench_ret.to_frame("Benchmark").to_json(date_format="iso", orient="split"),
            "asset_prices": prices.to_json(date_format="iso", orient="split"),
            "cov_method": cov_method,
            "rf_annual": rf_annual
        }, info
    except Exception as e:
        return no_update, f"Error building: {e}"

@app.callback(
    Output("table_metrics","data"),
    Output("cum_chart","figure"),
    Output("dd_chart","figure"),
    Output("frontier_chart","figure"),
    Output("weights_chart","figure"),
    Input("portfolio_state","data")
)
def update_views(ps):
    def _fig(title): return go.Figure(layout=dict(title=title))
    if not ps:
        return [], _fig("Cumulative"), _fig("Drawdown"), _fig("Frontier"), _fig("Weights")
    try:
        w_df  = pd.read_json(ps["weights_df"], orient="split")
        pr_df = pd.read_json(ps["portfolio_returns_df"], orient="split")
        rb_df = pd.read_json(ps["benchmark_returns_df"], orient="split")
        prices = pd.read_json(ps["asset_prices"], orient="split")
        cov_method = ps.get("cov_method", "Ledoit-Wolf")
        rf_annual = float(ps.get("rf_annual", 0.0))

        w  = w_df["weight"]; w.index = w_df.index
        pr = pr_df.iloc[:,0]; rb = rb_df.iloc[:,0]

        # Metrics (numeric)
        mdf = build_metrics_df(pr, rb, rf_annual)
        mdf.insert(0, "Series", mdf.index)
        metrics_table = mdf.reset_index(drop=True).to_dict("records")

        # Cumulative
        fig_cum = _fig("Cumulative Returns")
        if len(pr)>0:
            cum_p = (1+pr).cumprod()
            fig_cum.add_trace(go.Scatter(x=cum_p.index, y=cum_p.values, name="Portfolio", mode="lines"))
        if len(rb)>0:
            cum_b = (1+rb).cumprod()
            fig_cum.add_trace(go.Scatter(x=cum_b.index, y=cum_b.values, name=rb.name or "Benchmark", mode="lines", line=dict(dash="dash")))

        # Drawdown
        dd = drawdown_curve(pr)
        fig_dd = _fig("Drawdown")
        if len(dd)>0:
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", mode="lines"))

        # Enhanced Frontier
        fig_frontier = efficient_frontier_fig(prices, cov_method, rf_annual, current_w=w, show_eqw=True, npts=40)

        # Weights heatmap (sorted)
        w_sorted = w.sort_values(ascending=False)
        fig_w = px.imshow(w_sorted.to_frame("weight").values, x=["weight"], y=list(w_sorted.index),
                          labels=dict(color="Weight"), aspect="auto", origin="lower",
                          color_continuous_scale="Blues")
        fig_w.update_layout(title="Weights (sorted)")

        return metrics_table, fig_cum, fig_dd, fig_frontier, fig_w

    except Exception as e:
        return [], _fig(f"Cumulative (error: {e})"), _fig(f"Drawdown (error: {e})"), _fig(f"Frontier (error: {e})"), _fig(f"Weights (error: {e})")

# ------------------ Downloads (CSV: Metrics / Weights) ------------------
@app.callback(Output("dl_metrics","data"),
              Input("btn_dl_metrics","n_clicks"),
              State("portfolio_state","data"),
              prevent_initial_call=True)
def download_metrics(nc, ps):
    if not ps: return no_update
    pr_df = pd.read_json(ps["portfolio_returns_df"], orient="split")
    rb_df = pd.read_json(ps["benchmark_returns_df"], orient="split")
    pr = pr_df.iloc[:,0]; rb = rb_df.iloc[:,0]
    rf_annual = float(ps.get("rf_annual", 0.0))
    mdf = build_metrics_df(pr, rb, rf_annual)
    mdf.insert(0, "Series", mdf.index)
    return dcc.send_data_frame(mdf.reset_index(drop=True).to_csv, "metrics.csv", index=False)

@app.callback(Output("dl_weights","data"),
              Input("btn_dl_weights","n_clicks"),
              State("portfolio_state","data"),
              prevent_initial_call=True)
def download_weights(nc, ps):
    if not ps: return no_update
    w_df = pd.read_json(ps["weights_df"], orient="split")
    out = w_df.reset_index().rename(columns={"index":"Ticker"})
    return dcc.send_data_frame(out.to_csv, "weights.csv", index=False)

# ------------------ Walk-Forward OOS Backtest ------------------
@app.callback(
    Output("table_oos_metrics","data"),
    Output("oos_cum_chart","figure"),
    Output("oos_dd_chart","figure"),
    Output("oos_info","children"),
    Input("btn_oos","n_clicks"),
    State("portfolio_state","data"),
    State("oos_lookback_years","value"),
    State("oos_freq","value"),
    prevent_initial_call=True
)
def run_oos(nc, ps, lookback_years, freq_code):
    def _fig(title): return go.Figure(layout=dict(title=title))
    if not ps:
        return [], _fig("OOS Cumulative"), _fig("OOS Drawdown"), "Build a portfolio first."
    try:
        prices = pd.read_json(ps["asset_prices"], orient="split")
        cov_method = ps.get("cov_method", "Ledoit-Wolf")
        rf_annual = float(ps.get("rf_annual", 0.0))
        optimizer = ps.get("optimizer", "Min-Variance")
        oos_port, w_by_date = walk_forward_oos(prices, optimizer, cov_method, rf_annual,
                                               lookback_years=float(lookback_years or 3.0),
                                               freq_code=freq_code or "M")
        if oos_port.empty:
            return [], _fig("OOS Cumulative"), _fig("OOS Drawdown"), "OOS backtest produced no samples (insufficient history)."
        rb_df = pd.read_json(ps["benchmark_returns_df"], orient="split")
        bench = rb_df.iloc[:,0]
        idx = oos_port.index.intersection(bench.index)
        oos_p = oos_port.loc[idx]
        oos_b = bench.loc[idx]
        mdf = build_metrics_df(oos_p, oos_b, rf_annual)
        mdf.insert(0, "Series", mdf.index)
        table = mdf.reset_index(drop=True).to_dict("records")
        fig_cum = _fig("OOS Cumulative Returns")
        cum_p = (1+oos_p).cumprod(); fig_cum.add_trace(go.Scatter(x=cum_p.index, y=cum_p.values, name="OOS Portfolio", mode="lines"))
        cum_b = (1+oos_b).cumprod(); fig_cum.add_trace(go.Scatter(x=cum_b.index, y=cum_b.values, name=oos_b.name or "Benchmark", mode="lines", line=dict(dash="dash")))
        fig_dd = _fig("OOS Drawdown")
        dd = drawdown_curve(oos_p); fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="OOS DD", mode="lines"))
        info = (f"OOS: lookback={lookback_years}y, rebalance={dict(M='Monthly',Q='Quarterly',A='Annual').get(freq_code,'Monthly')}, "
                f"period: {oos_p.index.min().date()} → {oos_p.index.max().date()}, n={len(oos_p)}; rebalances={len(w_by_date)}")
        return table, fig_cum, fig_dd, info
    except Exception as e:
        return [], _fig(f"OOS Cumulative (error: {e})"), _fig(f"OOS Drawdown (error: {e})"), f"Error: {e}"

# -------------------- Main (external run) --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
