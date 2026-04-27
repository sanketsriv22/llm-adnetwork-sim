"""
Streamlit demo dashboard for the LLM Ad Network simulation.
Run with: streamlit run app.py
"""

import copy
import collections
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pipeline import EmbeddingModel, load_ad_embeddings, match_candidates, score_candidates, run_auction
from events import EventLog, apply_feedback
from queries import QUERIES

IMPRESSIONS_PER_DAY = 10_000
FEEDBACK_INTERVAL   = 1_000

st.set_page_config(page_title="LLM Ad Network", layout="wide", page_icon="📡")


# ── Cached resources (load once per session) ──────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model and ad index...")
def load_pipeline():
    import io, sys
    buf = io.StringIO()
    sys.stdout = buf
    try:
        model = EmbeddingModel()
        ads, index = load_ad_embeddings(model)
    finally:
        sys.stdout = sys.__stdout__
    return model, ads, index


@st.cache_resource
def get_query_embeddings():
    model, _, _ = load_pipeline()
    return model.embed_batch(QUERIES)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(n_queries: int, seed: int, progress_bar):
    model, ads_template, index = load_pipeline()
    query_embs = get_query_embeddings()

    ads              = copy.deepcopy(ads_template)
    event_log        = EventLog()
    rng              = np.random.default_rng(seed)
    cumulative_spend = {ad.id: 0.0 for ad in ads}   # daily_spend resets; track total here
    snapshots        = []                             # (day, {ad_id: (ctr, cvr)})
    recent_auctions  = collections.deque(maxlen=500)
    update_every     = max(1, n_queries // 200)

    def take_snapshot(day):
        snapshots.append((day, {ad.id: (ad.base_ctr, ad.base_cvr) for ad in ads}))

    take_snapshot(0)
    day = 0

    for i in range(n_queries):
        if i > 0 and i % IMPRESSIONS_PER_DAY == 0:
            day += 1
            for ad in ads:
                ad.daily_spend = 0.0
            take_snapshot(day)

        if i % update_every == 0:
            progress_bar.progress(
                i / n_queries,
                text=f"Day {day} · query {i:,} / {n_queries:,}",
            )

        qi    = int(rng.integers(len(query_embs)))
        q_emb = query_embs[qi]
        query = QUERIES[qi]

        candidates = match_candidates(q_emb, ads, index)
        if not candidates:
            continue

        scores = score_candidates(candidates, rng)
        result = run_auction(scores)
        if not result.winner:
            continue

        w      = result.winner
        imp_id = event_log.log_impression(w.ad.id, query)
        cumulative_spend[w.ad.id] = round(cumulative_spend[w.ad.id] + result.price_impr, 6)

        clicked   = rng.random() < w.ad.base_ctr
        converted = clicked and (rng.random() < w.ad.base_cvr)
        if clicked:
            event_log.log_click(imp_id)
        if converted:
            event_log.log_conversion(imp_id, w.ad.avg_order_value)

        recent_auctions.append({
            "Query":      query[:65],
            "Winner":     w.ad.name,
            "Category":   w.ad.category,
            "Eff Bid":    f"${w.effective_bid:.5f}",
            "Price/Impr": f"${result.price_impr:.5f}",
            "Clicked":    "✓" if clicked else "",
            "Converted":  "✓" if converted else "",
        })

        if event_log.total_impressions % FEEDBACK_INTERVAL == 0:
            apply_feedback(ads, event_log)

    take_snapshot(day + 1)
    progress_bar.progress(1.0, text="Simulation complete.")
    return ads, event_log, snapshots, list(recent_auctions), cumulative_spend


# ── DataFrame builders ────────────────────────────────────────────────────────

def build_stats_df(ads, event_log, cumulative_spend):
    stats = event_log.stats_per_ad()
    rows  = []
    for ad in ads:
        s      = stats.get(ad.id, {})
        impr   = s.get("impressions", 0)
        clicks = s.get("clicks", 0)
        convs  = s.get("conversions", 0)
        rev    = s.get("revenue", 0.0)
        spend  = cumulative_spend.get(ad.id, 0.0)
        rows.append({
            "Ad":          ad.name,
            "Category":    ad.category,
            "Impressions": impr,
            "Clicks":      clicks,
            "Conversions": convs,
            "CTR %":       round(clicks / impr * 100,   2) if impr   else 0.0,
            "CVR %":       round(convs  / clicks * 100, 2) if clicks else 0.0,
            "Revenue ($)": round(rev, 2),
            "Spend ($)":   round(spend, 2),
            "Budget ($)":  int(ad.daily_budget),
        })
    df = pd.DataFrame(rows).sort_values("Impressions", ascending=False).reset_index(drop=True)
    df["ROAS"]    = (df["Revenue ($)"] / df["Spend ($)"].replace(0, np.nan)).round(1)
    df["CPA ($)"] = (df["Spend ($)"]   / df["Conversions"].replace(0, np.nan)).round(2)
    return df


def build_evolution_df(snapshots, ads):
    id_to_name = {ad.id: ad.name for ad in ads}
    rows = []
    for day, snap in snapshots:
        for ad_id, (ctr, cvr) in snap.items():
            rows.append({
                "Day":   day,
                "Ad":    id_to_name.get(ad_id, ad_id),
                "CTR %": round(ctr * 100, 3),
                "CVR %": round(cvr * 100, 3),
            })
    return pd.DataFrame(rows)


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("📡 LLM Ad Network — Simulation Dashboard")
st.caption(
    "Second-price auction · HNSW vector matching · target-CPA bidding · Bayesian CTR/CVR feedback"
)

with st.sidebar:
    st.header("Simulation")
    n_queries = st.select_slider(
        "Total Queries",
        options=[10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
        value=100_000,
    )
    seed    = st.number_input("Random Seed", value=42, min_value=0, step=1)
    run_btn = st.button("▶  Run", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1. Embed query — `all-MiniLM-L6-v2`")
    st.markdown("2. ANN search — HNSW cosine index")
    st.markdown("3. Score — `target_cpa × pCVR × pCTR`")
    st.markdown("4. Second-price auction")
    st.markdown("5. Event log + Bayesian update")
    st.divider()
    st.markdown(f"**Query pool:** {len(QUERIES)} diverse queries")
    st.markdown(f"**Ad catalog:** 20 ads across 9 categories")
    st.markdown(f"**Day length:** {IMPRESSIONS_PER_DAY:,} impressions")
    st.markdown(f"**Feedback every:** {FEEDBACK_INTERVAL:,} impressions")

if run_btn:
    # Clear widget state from any previous run so multiselects re-initialize
    for key in ["evo_selected_ads", "evo_metric"]:
        st.session_state.pop(key, None)

    bar    = st.progress(0, text="Starting...")
    result = run_simulation(n_queries, int(seed), bar)
    st.session_state["results"] = result

if "results" not in st.session_state:
    st.info("Set parameters in the sidebar and click **▶ Run** to start the simulation.")
    st.stop()

ads, event_log, snapshots, recent_auctions, cumulative_spend = st.session_state["results"]
stats_df = build_stats_df(ads, event_log, cumulative_spend)
evo_df   = build_evolution_df(snapshots, ads)

# ── KPI row ───────────────────────────────────────────────────────────────────
total_rev   = stats_df["Revenue ($)"].sum()
total_spend = stats_df["Spend ($)"].sum()
overall_ctr = event_log.total_clicks      / event_log.total_impressions * 100 if event_log.total_impressions else 0
overall_cvr = event_log.total_conversions / event_log.total_clicks      * 100 if event_log.total_clicks      else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Impressions",  f"{event_log.total_impressions:,}")
c2.metric("Clicks",       f"{event_log.total_clicks:,}")
c3.metric("Conversions",  f"{event_log.total_conversions:,}")
c4.metric("Revenue",      f"${total_rev:,.0f}")
c5.metric("Overall CTR",  f"{overall_ctr:.2f}%")
c6.metric("Overall CVR",  f"{overall_cvr:.2f}%")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Ad Performance", "Rate Evolution", "Budget & Spend", "Recent Auctions"])


# ── Tab 1: Ad Performance ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Per-Ad Performance")
    display_cols = ["Ad", "Category", "Impressions", "Clicks", "Conversions",
                    "CTR %", "CVR %", "Revenue ($)", "Spend ($)", "Budget ($)", "ROAS", "CPA ($)"]
    st.dataframe(
        stats_df[display_cols].style
            .background_gradient(subset=["Impressions"], cmap="Blues")
            .background_gradient(subset=["Revenue ($)"], cmap="Greens")
            .format({
                "CTR %":       "{:.2f}",
                "CVR %":       "{:.2f}",
                "Revenue ($)": "${:,.0f}",
                "Spend ($)":   "${:,.2f}",
                "ROAS":        "{:.1f}x",
                "CPA ($)":     "${:.2f}",
            }),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    col_l, col_r = st.columns(2)

    with col_l:
        palette   = px.colors.qualitative.Safe
        cats      = list(stats_df["Category"].unique())
        cat_color = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}
        bar_colors = [cat_color[c] for c in stats_df["Category"]]

        fig = go.Figure(go.Bar(
            x=stats_df["Ad"].tolist(),
            y=stats_df["Impressions"].tolist(),
            marker_color=bar_colors,
            text=stats_df["Impressions"].apply(lambda v: f"{v:,}"),
            textposition="outside",
        ))
        fig.update_layout(
            title="Impressions by Ad",
            xaxis_tickangle=-40,
            xaxis_type="category",
            height=420,
            yaxis_title="Impressions",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        scatter_df = stats_df[stats_df["Impressions"] > 0].copy()
        fig = px.scatter(
            scatter_df,
            x="CTR %", y="CVR %",
            size="Impressions", size_max=60,
            color="Category",
            hover_name="Ad",
            text="Ad",
            title="CTR % vs CVR %  (bubble size = impressions)",
            height=420,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_traces(textposition="top center", textfont_size=8)
        fig.update_layout(legend_title="Category")
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Rate Evolution ─────────────────────────────────────────────────────
with tab2:
    st.subheader("CTR & CVR Drift Over Simulated Days")
    st.caption("Dashed lines show each ad's initial prior rate. Solid lines show how rates shift as the Bayesian feedback loop accumulates data.")

    all_ad_names = sorted(evo_df["Ad"].unique().tolist())

    col_metric, col_filter = st.columns([1, 3])
    with col_metric:
        metric = st.radio("Metric", ["CTR %", "CVR %"],
                          key="evo_metric", horizontal=False)
    with col_filter:
        selected = st.multiselect(
            "Filter ads (all shown by default)",
            options=all_ad_names,
            default=all_ad_names,
            key="evo_selected_ads",
        )

    if not selected:
        st.info("Select at least one ad above.")
    else:
        filtered = evo_df[evo_df["Ad"].isin(selected)]
        name_to_ad = {ad.name: ad for ad in ads}

        fig = px.line(
            filtered, x="Day", y=metric, color="Ad",
            title=f"{metric} evolution over simulated days",
            height=520,
            color_discrete_sequence=px.colors.qualitative.Alphabet,
        )
        for ad_name in selected:
            ad = name_to_ad.get(ad_name)
            if ad:
                init_val = (ad._init_ctr if metric == "CTR %" else ad._init_cvr) * 100
                fig.add_hline(y=init_val, line_dash="dash", line_color="gray",
                              opacity=0.25)
        fig.update_layout(legend_title="Ad")
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Budget & Spend ─────────────────────────────────────────────────────
with tab3:
    n_days = max(1, event_log.total_impressions // IMPRESSIONS_PER_DAY)
    st.subheader(f"Budget & Spend  —  {n_days} simulated days  ·  cumulative totals")
    st.caption(
        "Spend is cumulative across all simulated days. "
        "Daily budget is the per-day cap; an ad that hit 100% every day "
        "would spend `daily_budget × n_days` total."
    )

    col_l, col_r = st.columns(2)

    with col_l:
        # Show total spend as % of maximum possible (daily_budget × n_days)
        budget_df = stats_df[stats_df["Spend ($)"] > 0].copy()
        budget_df["Max Possible ($)"] = budget_df["Budget ($)"] * n_days
        budget_df["Total Util %"]     = (budget_df["Spend ($)"] / budget_df["Max Possible ($)"] * 100).round(1)
        budget_df_sorted = budget_df.sort_values("Total Util %", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Spend ($)",
            x=budget_df_sorted["Ad"].tolist(),
            y=budget_df_sorted["Spend ($)"].tolist(),
            marker_color="steelblue",
            text=[f"${v:,.0f}" for v in budget_df_sorted["Spend ($)"]],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Max Budget ($)",
            x=budget_df_sorted["Ad"].tolist(),
            y=budget_df_sorted["Max Possible ($)"].tolist(),
            marker_color="lightgray",
        ))
        fig.update_layout(
            barmode="overlay",
            title=f"Total Spend vs Max Budget (daily × {n_days} days)",
            xaxis_tickangle=-40,
            xaxis_type="category",
            height=440,
            legend_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        rev_spend_df = stats_df[stats_df["Spend ($)"] > 0].sort_values("Revenue ($)", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Spend ($)",
            x=rev_spend_df["Ad"].tolist(),
            y=rev_spend_df["Spend ($)"].tolist(),
            marker_color="steelblue",
        ))
        fig.add_trace(go.Bar(
            name="Revenue ($)",
            x=rev_spend_df["Ad"].tolist(),
            y=rev_spend_df["Revenue ($)"].tolist(),
            marker_color="seagreen",
        ))
        fig.update_layout(
            barmode="group",
            title="Cumulative Spend vs Revenue per Ad",
            xaxis_tickangle=-40,
            xaxis_type="category",
            height=440,
            legend_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("ROAS & CPA Summary")
    roas_display = stats_df[stats_df["Spend ($)"] > 0][
        ["Ad", "Category", "Impressions", "Conversions", "Spend ($)", "Revenue ($)", "ROAS", "CPA ($)"]
    ].sort_values("ROAS", ascending=False).reset_index(drop=True)
    st.dataframe(
        roas_display.style.background_gradient(subset=["ROAS"], cmap="RdYlGn")
                          .format({
                              "Spend ($)":   "${:,.2f}",
                              "Revenue ($)": "${:,.0f}",
                              "ROAS":        "{:.1f}x",
                              "CPA ($)":     "${:.2f}",
                          }),
        use_container_width=True,
        hide_index=True,
    )


# ── Tab 4: Recent Auctions ────────────────────────────────────────────────────
with tab4:
    st.subheader(f"Last {len(recent_auctions)} Auctions")
    if recent_auctions:
        st.dataframe(
            pd.DataFrame(recent_auctions[::-1]),
            use_container_width=True,
            hide_index=True,
            height=600,
        )
    else:
        st.write("No auction data.")
