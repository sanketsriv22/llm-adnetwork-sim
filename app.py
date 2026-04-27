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

st.set_page_config(
    page_title="LLM Ad Network",
    layout="wide",
    page_icon="📡",
)


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

    ads       = copy.deepcopy(ads_template)
    event_log = EventLog()
    rng       = np.random.default_rng(seed)

    snapshots        = []                          # (day, {ad_id: (ctr, cvr)})
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

        clicked   = rng.random() < w.ad.base_ctr
        converted = clicked and (rng.random() < w.ad.base_cvr)
        if clicked:
            event_log.log_click(imp_id)
        if converted:
            event_log.log_conversion(imp_id, w.ad.avg_order_value)

        recent_auctions.append({
            "Query":     query[:65],
            "Winner":    w.ad.name,
            "Category":  w.ad.category,
            "Eff Bid":   f"${w.effective_bid:.5f}",
            "Price/Impr":f"${result.price_impr:.5f}",
            "Clicked":   "✓" if clicked else "",
            "Converted": "✓" if converted else "",
        })

        if event_log.total_impressions % FEEDBACK_INTERVAL == 0:
            apply_feedback(ads, event_log)

    take_snapshot(day + 1)
    progress_bar.progress(1.0, text="Simulation complete.")
    return ads, event_log, snapshots, list(recent_auctions)


# ── DataFrame builders ────────────────────────────────────────────────────────

def build_stats_df(ads, event_log):
    stats = event_log.stats_per_ad()
    rows  = []
    for ad in ads:
        s     = stats.get(ad.id, {})
        impr  = s.get("impressions", 0)
        clicks = s.get("clicks", 0)
        convs  = s.get("conversions", 0)
        rev    = s.get("revenue", 0.0)
        rows.append({
            "Ad":           ad.name,
            "Category":     ad.category,
            "Impressions":  impr,
            "Clicks":       clicks,
            "Conversions":  convs,
            "CTR %":        round(clicks / impr * 100, 2)   if impr   else 0,
            "CVR %":        round(convs / clicks * 100, 2)  if clicks else 0,
            "Revenue ($)":  round(rev, 2),
            "Spend ($)":    round(ad.daily_spend, 2),
            "Budget ($)":   int(ad.daily_budget),
            "Util %":       round(ad.daily_spend / ad.daily_budget * 100, 1) if ad.daily_budget else 0,
        })
    return pd.DataFrame(rows).sort_values("Impressions", ascending=False).reset_index(drop=True)


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

# Sidebar
with st.sidebar:
    st.header("Simulation")
    n_queries = st.select_slider(
        "Total Queries",
        options=[10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
        value=100_000,
    )
    seed = st.number_input("Random Seed", value=42, min_value=0, step=1)
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
    bar = st.progress(0, text="Starting...")
    result = run_simulation(n_queries, int(seed), bar)
    st.session_state["results"] = result

if "results" not in st.session_state:
    st.info("Set parameters in the sidebar and click **▶ Run** to start the simulation.")
    st.stop()

ads, event_log, snapshots, recent_auctions = st.session_state["results"]
stats_df = build_stats_df(ads, event_log)
evo_df   = build_evolution_df(snapshots, ads)

# ── KPI row ───────────────────────────────────────────────────────────────────
total_rev   = stats_df["Revenue ($)"].sum()
overall_ctr = event_log.total_clicks / event_log.total_impressions * 100 if event_log.total_impressions else 0
overall_cvr = event_log.total_conversions / event_log.total_clicks * 100 if event_log.total_clicks else 0

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
    st.dataframe(
        stats_df.style
            .background_gradient(subset=["Impressions"], cmap="Blues")
            .background_gradient(subset=["Revenue ($)"], cmap="Greens")
            .format({
                "CTR %": "{:.2f}",
                "CVR %": "{:.2f}",
                "Util %": "{:.1f}",
                "Revenue ($)": "${:,.0f}",
                "Spend ($)": "${:.2f}",
            }),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            stats_df, x="Ad", y="Impressions", color="Category",
            title="Impressions by Ad", height=400,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(xaxis_tickangle=-40, legend_title="Category")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.scatter(
            stats_df, x="CTR %", y="CVR %", size="Impressions",
            color="Category", hover_name="Ad", text="Ad",
            title="CTR % vs CVR %  (bubble = impressions)",
            height=400,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(legend_title="Category")
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Rate Evolution ─────────────────────────────────────────────────────
with tab2:
    st.subheader("CTR & CVR Drift Over Simulated Days")
    st.caption("Dotted lines show each ad's initial prior rate.")

    col_metric, col_filter = st.columns([1, 3])
    with col_metric:
        metric = st.radio("Metric", ["CTR %", "CVR %"], horizontal=False)
    with col_filter:
        all_ads = sorted(evo_df["Ad"].unique().tolist())
        selected = st.multiselect("Filter ads", all_ads, default=all_ads)

    filtered = evo_df[evo_df["Ad"].isin(selected)]

    fig = px.line(
        filtered, x="Day", y=metric, color="Ad",
        title=f"{metric} over simulated days",
        height=500,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )

    # dashed reference line per selected ad at its init rate
    id_to_ad = {ad.id: ad for ad in ads}
    name_to_ad = {ad.name: ad for ad in ads}
    for ad_name in selected:
        ad = name_to_ad.get(ad_name)
        if ad:
            init_val = (ad._init_ctr if metric == "CTR %" else ad._init_cvr) * 100
            fig.add_hline(
                y=init_val, line_dash="dot", line_color="gray",
                opacity=0.25,
            )

    fig.update_layout(legend_title="Ad")
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Budget & Spend ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Budget Utilization & Spend vs Revenue")

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            stats_df.sort_values("Util %", ascending=False),
            x="Ad", y="Util %", color="Category",
            title="Budget Utilization % (end of last simulated day)",
            height=420,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="red",
                      opacity=0.6, annotation_text="100%", annotation_position="right")
        fig.update_layout(xaxis_tickangle=-40, legend_title="Category")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = go.Figure()
        sorted_df = stats_df.sort_values("Revenue ($)", ascending=False)
        fig.add_trace(go.Bar(name="Spend ($)",   x=sorted_df["Ad"], y=sorted_df["Spend ($)"],
                             marker_color="steelblue"))
        fig.add_trace(go.Bar(name="Revenue ($)", x=sorted_df["Ad"], y=sorted_df["Revenue ($)"],
                             marker_color="seagreen"))
        fig.update_layout(
            barmode="group", title="Spend vs Revenue per Ad",
            xaxis_tickangle=-40, height=420, legend_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ROAS table
    st.subheader("Return on Ad Spend (ROAS)")
    roas_df = stats_df[stats_df["Spend ($)"] > 0].copy()
    roas_df["ROAS"] = (roas_df["Revenue ($)"] / roas_df["Spend ($)"]).round(1)
    roas_df["CPA ($)"] = (roas_df["Spend ($)"] / roas_df["Conversions"].replace(0, np.nan)).round(2)
    st.dataframe(
        roas_df[["Ad", "Category", "Spend ($)", "Revenue ($)", "ROAS", "CPA ($)"]].sort_values("ROAS", ascending=False),
        use_container_width=True, hide_index=True,
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
