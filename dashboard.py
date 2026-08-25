"""
NBA Betting Strategy Dashboard
Run with: streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Betting Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
OUTPUTS = BASE_DIR / "outputs"

ANALYSIS_DIRS = sorted(
    [d.name for d in OUTPUTS.iterdir() if d.is_dir() and d.name.startswith("copilot_analysis_odds_")]
)

# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_results() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "results.tsv", sep="\t", parse_dates=["timestamp"])
    params_df = df["params"].apply(json.loads).apply(pd.Series)
    df = pd.concat([df.drop(columns=["params"]), params_df], axis=1)
    df["models_str"] = df["models"].apply(
        lambda x: "+".join(x) if isinstance(x, list) else str(x)
    )
    return df


@st.cache_data
def load_kelly_bets() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "kelly_bets_summary.csv", parse_dates=["GAME_DATE"])
    # Normalize bet_win to int for aggregations
    if df["bet_win"].dtype == object:
        df["bet_win"] = df["bet_win"].map({"True": 1, "False": 0, True: 1, False: 0}).astype(float)
    else:
        df["bet_win"] = pd.to_numeric(df["bet_win"], errors="coerce")
    return df


@st.cache_data
def load_backtest() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "backtest_results.csv", parse_dates=["GAME_DATE"])
    df = df.drop_duplicates(subset=["GAME_ID", "bet_side"])
    return df


@st.cache_data
def load_betting_summary() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "copilot_betting_summary.csv")


@st.cache_data
def load_model_rmse() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / "copilot_model_rmse.csv")
    return df[df["group_type"].fillna("overall") == "overall"].copy()


@st.cache_data
def load_analysis(folder: str) -> dict:
    base = OUTPUTS / folder
    dfs = {}
    for name in [
        "monthly_summary",
        "season_summary",
        "profitable_models_summary",
        "regime_to_profile_summary",
    ]:
        p = base / f"{name}.csv"
        if p.exists():
            dfs[name] = pd.read_csv(p)
    return dfs


# ── Controls (moved from sidebar) ─────────────────────────────────────────────
controls_col, _ = st.columns([1, 3])
with controls_col:
    st.title("🏀 NBA Betting")
    st.divider()
    st.header("Date Filter")

    kelly_df_raw = load_kelly_bets()
    date_min = kelly_df_raw["GAME_DATE"].min().date()
    date_max = kelly_df_raw["GAME_DATE"].max().date()

    date_range = st.date_input(
        "Game date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    st.divider()
    st.header("Deep Dive Folder")
    selected_folder = st.selectbox(
        "Analysis iteration",
        ANALYSIS_DIRS,
        index=len(ANALYSIS_DIRS) - 1,
    )

# Apply date filter globally
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    d_start = pd.Timestamp(date_range[0])
    d_end = pd.Timestamp(date_range[1])
else:
    d_start = pd.Timestamp(date_min)
    d_end = pd.Timestamp(date_max)

kelly_df = kelly_df_raw[
    (kelly_df_raw["GAME_DATE"] >= d_start) & (kelly_df_raw["GAME_DATE"] <= d_end)
].copy()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🔬 Experiments", "💰 Betting Performance", "🤖 Model Analysis", "🔍 Deep Dive"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Overview")

    results_df = load_results()
    summary_df = load_betting_summary()

    best_run = results_df.loc[results_df["score"].idxmax()]
    latest_run = results_df.sort_values("timestamp").iloc[-1]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Score", f"{best_run['score']:.4f}")
    c2.metric("Best RMSE", f"{results_df['ensemble_rmse'].min():.2f}")
    c3.metric("Latest RMSE", f"{latest_run['ensemble_rmse']:.2f}", delta=f"{latest_run['ensemble_rmse'] - results_df['ensemble_rmse'].min():.2f} above best")
    c4.metric("Total Runs", f"{len(results_df):,}")
    if "win_rate" in summary_df.columns:
        c5.metric("Best Model Win Rate", f"{summary_df['win_rate'].max():.1%}")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Objective Score Over Time")
        fig = px.line(
            results_df.sort_values("timestamp"),
            x="timestamp", y="score",
            hover_data=["exp_id", "ensemble_rmse", "models_str"],
            color_discrete_sequence=["#00b4d8"],
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#e63946", opacity=0.6, annotation_text="Break-even")
        fig.update_layout(xaxis_title="Run time", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Ensemble RMSE Over Time")
        fig2 = px.line(
            results_df.sort_values("timestamp"),
            x="timestamp", y="ensemble_rmse",
            hover_data=["exp_id", "score", "models_str"],
            color_discrete_sequence=["#f77f00"],
        )
        fig2.add_hline(
            y=15.5, line_dash="dash", line_color="#ffd60a", opacity=0.7,
            annotation_text="Target ≤15.5"
        )
        fig2.update_layout(xaxis_title="Run time", yaxis_title="RMSE")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Model Summary Table")
    fmt = {}
    for col in ["roi", "win_rate", "max_drawdown"]:
        if col in summary_df.columns:
            fmt[col] = "{:.1%}"
    for col in ["starting_bankroll"]:
        if col in summary_df.columns:
            fmt[col] = "${:,.0f}"
    if "ending_bankroll" in summary_df.columns:
        fmt["ending_bankroll"] = "{:.4f}"
    st.dataframe(summary_df.style.format(fmt), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Experiment Tracker
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Experiment Tracker")

    results_df = load_results()

    c1, c2, c3 = st.columns(3)
    model_options = sorted(results_df["models_str"].unique().tolist())
    sel_models = c1.multiselect("Model combo", model_options, default=model_options)

    rmse_min = float(results_df["ensemble_rmse"].min())
    rmse_max = float(results_df["ensemble_rmse"].max())
    rmse_range = c2.slider("RMSE range", rmse_min, rmse_max, (rmse_min, rmse_max), step=0.01)

    ev_options = sorted(results_df["ev_threshold"].unique().tolist()) if "ev_threshold" in results_df.columns else []
    sel_ev = c3.multiselect("EV threshold", ev_options, default=ev_options)

    mask = results_df["models_str"].isin(sel_models) & results_df["ensemble_rmse"].between(*rmse_range)
    if sel_ev:
        mask &= results_df["ev_threshold"].isin(sel_ev)
    filtered = results_df[mask].copy()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("RMSE vs Score")
        fig = px.scatter(
            filtered, x="ensemble_rmse", y="score",
            color="models_str",
            hover_data=["exp_id", "timestamp", "train_size", "ev_threshold", "fractional_kelly"],
            opacity=0.8,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#e63946", opacity=0.5)
        fig.add_vline(x=15.5, line_dash="dash", line_color="#ffd60a", opacity=0.5, annotation_text="RMSE target")
        fig.update_layout(legend_title="Models")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if "train_size" in filtered.columns:
            st.subheader("Score by Train Size")
            fig2 = px.box(
                filtered, x="train_size", y="score", color="models_str",
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="#e63946", opacity=0.5)
            fig2.update_layout(legend_title="Models")
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader(f"All Runs ({len(filtered):,} of {len(results_df):,})")
    display_cols = [
        "timestamp", "exp_id", "ensemble_rmse", "roi", "score",
        "train_size", "test_size", "ev_threshold", "fractional_kelly",
        "max_kelly", "top_n_bets", "models_str",
    ]
    avail_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[avail_cols]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
        .style.format({"ensemble_rmse": "{:.4f}", "roi": "{:.4f}", "score": "{:.6f}"}),
        use_container_width=True,
        height=420,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Betting Performance
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Betting Performance")
    st.caption(
        f"Date range: {d_start.date()} → {d_end.date()} | {len(kelly_df):,} bets "
        f"| Win rate: {kelly_df['bet_win'].mean():.1%} "
        f"| Total P&L: {kelly_df['pnl_kelly'].sum():.2f}"
    )

    if kelly_df.empty:
        st.warning("No bets in the selected date range.")
    else:
        # ── Bankroll curve ──
        st.subheader("Bankroll Over Time")
        bk_df = kelly_df.sort_values("GAME_DATE").copy()
        bk_df["rolling_peak"] = bk_df["bankroll"].cummax()

        fig_bk = go.Figure()
        fig_bk.add_trace(go.Scatter(
            x=bk_df["GAME_DATE"], y=bk_df["rolling_peak"],
            mode="lines", name="Peak",
            line=dict(color="#90e0ef", width=1, dash="dot"),
            fill=None,
        ))
        fig_bk.add_trace(go.Scatter(
            x=bk_df["GAME_DATE"], y=bk_df["bankroll"],
            mode="lines", name="Bankroll",
            line=dict(color="#00b4d8", width=2),
            fill="tonexty", fillcolor="rgba(230,57,70,0.15)",
        ))
        fig_bk.update_layout(
            xaxis_title="Date", yaxis_title="Bankroll ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_bk, use_container_width=True)

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Monthly P&L")
            bk_df["month"] = bk_df["GAME_DATE"].dt.to_period("M").astype(str)
            monthly = bk_df.groupby("month")["pnl_kelly"].sum().reset_index()
            monthly.columns = ["month", "total_pnl"]
            monthly["color"] = monthly["total_pnl"].apply(lambda x: "Profit" if x >= 0 else "Loss")
            fig_mo = px.bar(
                monthly, x="month", y="total_pnl",
                color="color",
                color_discrete_map={"Profit": "#2dc653", "Loss": "#e63946"},
            )
            fig_mo.update_layout(showlegend=False, xaxis_tickangle=-45, yaxis_title="P&L")
            st.plotly_chart(fig_mo, use_container_width=True)

        with col_r:
            st.subheader("Win Rate by Edge Bucket")
            edge_data = kelly_df.dropna(subset=["edge_bucket"]).copy()
            if not edge_data.empty:
                edge_stats = (
                    edge_data.groupby("edge_bucket")
                    .agg(win_rate=("bet_win", "mean"), bets=("bet_win", "count"))
                    .reset_index()
                )
                fig_edge = px.bar(
                    edge_stats, x="edge_bucket", y="win_rate",
                    text=edge_stats["bets"].apply(lambda n: f"n={n}"),
                    color_discrete_sequence=["#00b4d8"],
                )
                fig_edge.add_hline(y=0.5, line_dash="dash", line_color="#e63946", opacity=0.6)
                fig_edge.update_yaxes(tickformat=".0%")
                fig_edge.update_layout(yaxis_title="Win Rate")
                st.plotly_chart(fig_edge, use_container_width=True)

        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.subheader("HOME vs AWAY")
            if "bet_side" in kelly_df.columns:
                side_stats = (
                    kelly_df.groupby("bet_side")
                    .agg(win_rate=("bet_win", "mean"), bets=("bet_win", "count"), avg_pnl=("pnl_kelly", "mean"))
                    .reset_index()
                )
                fig_side = px.bar(
                    side_stats, x="bet_side", y="win_rate",
                    text=side_stats["bets"].apply(lambda n: f"n={n:,}"),
                    color="bet_side",
                    color_discrete_map={"HOME": "#00b4d8", "AWAY": "#f77f00"},
                )
                fig_side.add_hline(y=0.5, line_dash="dash", line_color="#e63946", opacity=0.6)
                fig_side.update_yaxes(tickformat=".0%")
                fig_side.update_layout(showlegend=False, yaxis_title="Win Rate")
                st.plotly_chart(fig_side, use_container_width=True)

        with col_r2:
            st.subheader("Season P&L")
            if "season" in kelly_df.columns:
                season_stats = (
                    kelly_df.groupby("season")
                    .agg(total_pnl=("pnl_kelly", "sum"), win_rate=("bet_win", "mean"), bets=("bet_win", "count"))
                    .reset_index()
                )
                season_stats["color"] = season_stats["total_pnl"].apply(lambda x: "Profit" if x >= 0 else "Loss")
                fig_season = px.bar(
                    season_stats, x="season", y="total_pnl",
                    color="color",
                    color_discrete_map={"Profit": "#2dc653", "Loss": "#e63946"},
                    text=season_stats["win_rate"].apply(lambda r: f"{r:.1%}"),
                    hover_data=["bets"],
                )
                fig_season.update_layout(showlegend=False, yaxis_title="P&L")
                st.plotly_chart(fig_season, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Model Analysis")

    rmse_df = load_model_rmse()
    summary_df = load_betting_summary()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("RMSE by Model")
        overall = rmse_df.drop_duplicates(subset="model_id")
        fig_rmse = px.bar(
            overall.sort_values("rmse_mean"),
            x="model_id", y="rmse_mean",
            error_y="rmse_std" if "rmse_std" in overall.columns else None,
            color_discrete_sequence=["#00b4d8"],
        )
        fig_rmse.add_hline(
            y=15.5, line_dash="dash", line_color="#ffd60a", opacity=0.7,
            annotation_text="Target ≤15.5",
        )
        fig_rmse.update_layout(yaxis_title="RMSE", xaxis_title="Model")
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col_r:
        st.subheader("Probability Calibration")
        cal_cols = {"win_prob_home_calibrated", "bet_win", "bet_side"}
        if cal_cols.issubset(kelly_df.columns):
            cal_df = kelly_df.dropna(subset=list(cal_cols)).copy()
            cal_df["pred_prob"] = cal_df.apply(
                lambda r: r["win_prob_home_calibrated"]
                if str(r["bet_side"]).upper() == "HOME"
                else 1 - r["win_prob_home_calibrated"],
                axis=1,
            )
            cal_df["prob_bin"] = pd.cut(cal_df["pred_prob"], bins=10)
            cal_stats = (
                cal_df.groupby("prob_bin", observed=True)
                .agg(mean_pred=("pred_prob", "mean"), actual_wr=("bet_win", "mean"), n=("bet_win", "count"))
                .reset_index()
                .dropna()
            )
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="#666"), name="Perfect calibration",
            ))
            fig_cal.add_trace(go.Scatter(
                x=cal_stats["mean_pred"], y=cal_stats["actual_wr"],
                mode="markers+lines",
                marker=dict(
                    size=cal_stats["n"] / cal_stats["n"].max() * 18 + 6,
                    color="#00b4d8",
                ),
                text=cal_stats["n"].apply(lambda n: f"n={n}"),
                name="Model",
            ))
            fig_cal.update_layout(
                xaxis_title="Predicted probability (bet side)",
                yaxis_title="Actual win rate",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_cal, use_container_width=True)

    st.subheader("Per-Model Betting Summary")
    if not summary_df.empty:
        fmt = {}
        for col in ["roi", "win_rate", "max_drawdown"]:
            if col in summary_df.columns:
                fmt[col] = "{:.1%}"
        for col in ["starting_bankroll"]:
            if col in summary_df.columns:
                fmt[col] = "${:,.0f}"
        if "ending_bankroll" in summary_df.columns:
            fmt["ending_bankroll"] = "{:.4f}"
        st.dataframe(summary_df.style.format(fmt), use_container_width=True)

    # Drawdown from backtest
    st.subheader("Drawdown Over Time (Backtest)")
    try:
        bt_df = load_backtest()
        bt_filtered = bt_df[
            (bt_df["GAME_DATE"] >= d_start) & (bt_df["GAME_DATE"] <= d_end)
        ].sort_values("GAME_DATE")
        if "drawdown" in bt_filtered.columns and not bt_filtered.empty:
            fig_dd = px.area(
                bt_filtered, x="GAME_DATE", y="drawdown",
                color_discrete_sequence=["#e63946"],
            )
            fig_dd.update_yaxes(tickformat=".0%", title="Drawdown")
            fig_dd.update_xaxes(title="Date")
            st.plotly_chart(fig_dd, use_container_width=True)
    except Exception as exc:
        st.info(f"Drawdown chart unavailable: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header(f"Deep Dive — {selected_folder}")

    analysis_data = load_analysis(selected_folder)
    analysis_dir = OUTPUTS / selected_folder

    # Static images
    img_files = [
        ("model_comparison.png", "Model Comparison"),
        ("experiment_matrix_comparison.png", "Experiment Matrix"),
        ("home_away_bets_wins_by_model.png", "Home/Away Bets & Wins by Model"),
    ]
    available_imgs = [(f, cap) for f, cap in img_files if (analysis_dir / f).exists()]
    if available_imgs:
        img_cols = st.columns(min(len(available_imgs), 2))
        for i, (fname, cap) in enumerate(available_imgs):
            with img_cols[i % 2]:
                st.image(str(analysis_dir / fname), caption=cap, use_container_width=True)
        st.divider()

    if not analysis_data:
        st.warning("No analysis CSV files found in selected folder.")
    else:
        inner_tabs = st.tabs(["📅 Monthly", "📆 Season", "🏆 Profitable Models", "🌊 Volatility Regime"])

        # ── Monthly ──
        with inner_tabs[0]:
            if "monthly_summary" in analysis_data:
                df = analysis_data["monthly_summary"]
                models = sorted(df["model"].unique().tolist()) if "model" in df.columns else []
                sel = st.selectbox("Model", models, key="dd_monthly_model")
                mdf = df[df["model"] == sel] if sel else df
                col_chart, col_table = st.columns([3, 2])
                with col_chart:
                    mdf = mdf.copy()
                    mdf["color"] = mdf["total_pnl"].apply(lambda x: "Profit" if x >= 0 else "Loss")
                    fig = px.bar(
                        mdf, x="month", y="total_pnl",
                        color="color",
                        color_discrete_map={"Profit": "#2dc653", "Loss": "#e63946"},
                        title=f"Monthly P&L — {sel}",
                    )
                    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                with col_table:
                    st.dataframe(
                        mdf[["month", "bets", "total_pnl", "avg_pnl_per_bet", "win_rate"]]
                        .style.format({"total_pnl": "{:.4f}", "avg_pnl_per_bet": "{:.5f}", "win_rate": "{:.1%}"}),
                        use_container_width=True,
                        height=400,
                    )

        # ── Season ──
        with inner_tabs[1]:
            if "season_summary" in analysis_data:
                df = analysis_data["season_summary"]
                col_chart, col_table = st.columns([3, 2])
                with col_chart:
                    fig = px.bar(
                        df, x="season", y="total_pnl",
                        color="model", barmode="group",
                        title="Season P&L by Model",
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="#e63946", opacity=0.5)
                    st.plotly_chart(fig, use_container_width=True)
                with col_table:
                    st.dataframe(
                        df.style.format({"total_pnl": "{:.4f}", "avg_pnl_per_bet": "{:.5f}", "win_rate": "{:.1%}"}),
                        use_container_width=True,
                    )

        # ── Profitable Models ──
        with inner_tabs[2]:
            if "profitable_models_summary" in analysis_data:
                df = analysis_data["profitable_models_summary"]
                fmt = {}
                for col in ["roi", "win_rate", "max_drawdown", "bankroll_growth"]:
                    if col in df.columns:
                        fmt[col] = "{:.2%}"
                for col in ["starting_bankroll", "ending_bankroll"]:
                    if col in df.columns:
                        fmt[col] = "{:.4f}"
                if "avg_edge" in df.columns:
                    fmt["avg_edge"] = "{:.2f}"
                st.dataframe(df.style.format(fmt), use_container_width=True)
                # ROI bar chart
                if "roi" in df.columns:
                    fig = px.bar(
                        df.sort_values("roi", ascending=False),
                        x="model", y="roi",
                        color=df.sort_values("roi", ascending=False)["roi"].apply(
                            lambda x: "Profit" if x >= 0 else "Loss"
                        ),
                        color_discrete_map={"Profit": "#2dc653", "Loss": "#e63946"},
                        title="ROI by Model",
                    )
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

        # ── Volatility Regime ──
        with inner_tabs[3]:
            if "regime_to_profile_summary" in analysis_data:
                df = analysis_data["regime_to_profile_summary"]
                col_chart, col_table = st.columns([3, 2])
                with col_chart:
                    fig = px.bar(
                        df, x="vol_bucket", y="avg_daily_pnl",
                        color="model", barmode="group",
                        title="Avg Daily P&L by Volatility Regime",
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="#e63946", opacity=0.5)
                    st.plotly_chart(fig, use_container_width=True)
                with col_table:
                    st.dataframe(
                        df.style.format({
                            "avg_daily_win_rate": "{:.1%}",
                            "avg_daily_pnl": "{:.4f}",
                        }),
                        use_container_width=True,
                    )
