"""
Monte Carlo simulation over the ad network pipeline.

Usage:
    python simulate.py                  # default 1,000,000 queries
    python simulate.py --queries 50000  # quick smoke test

Outputs:
    - per-ad stats table printed to stdout
    - rate_evolution.png  (CTR/CVR drift per ad over simulated days)
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from pipeline import AdPipeline, match_candidates, score_candidates, run_auction, AD_CATALOG
from events import apply_feedback
from queries import QUERIES


# ── Simulation constants ──────────────────────────────────────────────────────

IMPRESSIONS_PER_DAY = 10_000   # budget resets every this many total impressions
FEEDBACK_INTERVAL   = 1_000    # apply_feedback() every N impressions


# ── Stats helpers ─────────────────────────────────────────────────────────────

def print_stats(ads, event_log):
    stats = event_log.stats_per_ad()
    total_rev = sum(s["revenue"] for s in stats.values())

    header = (f"  {'Ad':<22} {'Cat':<10} {'Impr':>7} {'Clicks':>7} {'Convs':>6} "
              f"{'CTR%':>6} {'CVR%':>6} {'Rev($)':>9} {'Spend($)':>9} {'Budg($)':>8} {'Util%':>6}")
    print(f"\n{'═'*len(header)}")
    print(header)
    print(f"  {'─'*len(header)}")

    for ad in sorted(ads, key=lambda a: stats.get(a.id, {}).get("impressions", 0), reverse=True):
        s = stats.get(ad.id)
        if not s:
            continue
        ctr  = 100 * s["clicks"] / s["impressions"] if s["impressions"] else 0
        cvr  = 100 * s["conversions"] / s["clicks"] if s["clicks"] else 0
        util = 100 * ad.daily_spend / ad.daily_budget if ad.daily_budget else 0
        print(f"  {ad.name:<22} {ad.category:<10} {s['impressions']:>7,} {s['clicks']:>7,} "
              f"{s['conversions']:>6,} {ctr:>6.1f} {cvr:>6.1f} "
              f"{s['revenue']:>9,.1f} {ad.daily_spend:>9.2f} "
              f"{ad.daily_budget:>8.0f} {util:>6.1f}")

    print(f"\n  Total impressions : {event_log.total_impressions:,}")
    print(f"  Total clicks      : {event_log.total_clicks:,}")
    print(f"  Total conversions : {event_log.total_conversions:,}")
    print(f"  Total revenue     : ${total_rev:,.2f}")
    overall_ctr = 100 * event_log.total_clicks / event_log.total_impressions if event_log.total_impressions else 0
    overall_cvr = 100 * event_log.total_conversions / event_log.total_clicks if event_log.total_clicks else 0
    print(f"  Overall CTR       : {overall_ctr:.2f}%")
    print(f"  Overall CVR       : {overall_cvr:.2f}%")


def plot_rate_evolution(snapshots, ads, out_path="rate_evolution.png"):
    """
    snapshots: list of (day_idx, {ad_id: (ctr, cvr)})
    Produces one subplot per ad showing CTR and CVR over simulated days.
    """
    days = [s[0] for s in snapshots]
    n    = len(ads)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5), squeeze=False)
    fig.suptitle("CTR / CVR Evolution Over Simulated Days", fontsize=14, y=1.01)

    for i, ad in enumerate(ads):
        ax  = axes[i // cols][i % cols]
        ctr_vals = [s[1].get(ad.id, (ad._init_ctr, ad._init_cvr))[0] for s in snapshots]
        cvr_vals = [s[1].get(ad.id, (ad._init_ctr, ad._init_cvr))[1] for s in snapshots]

        ax.plot(days, [v * 100 for v in ctr_vals], label="CTR%", color="steelblue")
        ax.plot(days, [v * 100 for v in cvr_vals], label="CVR%", color="darkorange")
        ax.axhline(ad._init_ctr * 100, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(ad._init_cvr * 100, color="darkorange", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{ad.name}", fontsize=9)
        ax.set_xlabel("Day", fontsize=7)
        ax.set_ylabel("%", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    # hide unused subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\n  Saved chart → {out_path}")


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_simulation(n_queries: int):
    print(f"\n[ Monte Carlo Simulation ]  n={n_queries:,}  "
          f"days={n_queries // IMPRESSIONS_PER_DAY}  "
          f"feedback every {FEEDBACK_INTERVAL:,} impr\n")

    # ── init pipeline (loads model + index once) ──────────────────────────────
    from pipeline import EmbeddingModel, load_ad_embeddings, AD_CATALOG
    from events import EventLog

    embed_model = EmbeddingModel()
    ads, index  = load_ad_embeddings(embed_model)
    event_log   = EventLog()
    rng         = np.random.default_rng(42)

    # ── pre-embed all queries once ────────────────────────────────────────────
    print(f"\n  Pre-embedding {len(QUERIES)} sample queries...", end=" ", flush=True)
    query_texts = QUERIES
    query_embs  = embed_model.embed_batch(query_texts)
    print("OK")

    snapshots: list = []   # (day_idx, {ad_id: (ctr, cvr)})

    def take_snapshot(day_idx):
        snap = {ad.id: (ad.base_ctr, ad.base_cvr) for ad in ads}
        snapshots.append((day_idx, snap))

    take_snapshot(0)

    # ── simulation ────────────────────────────────────────────────────────────
    t0          = time.time()
    served      = 0
    day         = 0

    print(f"\n  Running...")

    for i in range(n_queries):
        # daily budget reset
        if i > 0 and i % IMPRESSIONS_PER_DAY == 0:
            day += 1
            for ad in ads:
                ad.daily_spend = 0.0
            take_snapshot(day)
            elapsed = time.time() - t0
            pct = 100 * i / n_queries
            print(f"  Day {day:>3}  ({pct:.0f}%)  served={served:,}  "
                  f"impr={event_log.total_impressions:,}  "
                  f"elapsed={elapsed:.1f}s")

        # pick a random pre-embedded query
        qi     = int(rng.integers(len(query_embs)))
        q_emb  = query_embs[qi]
        query  = query_texts[qi]

        # stage 1: match
        candidates = match_candidates(q_emb, ads, index)
        if not candidates:
            continue

        # stage 2: score
        scores = score_candidates(candidates, rng)

        # stage 3: auction
        result = run_auction(scores)
        if not result.winner:
            continue

        # stage 4: log events + simulate outcome
        w      = result.winner
        imp_id = event_log.log_impression(w.ad.id, query)
        served += 1

        clicked   = rng.random() < w.p_ctr
        converted = clicked and (rng.random() < w.p_cvr)
        if clicked:
            event_log.log_click(imp_id)
        if converted:
            event_log.log_conversion(imp_id, w.ad.avg_order_value)

        # feedback
        if event_log.total_impressions % FEEDBACK_INTERVAL == 0:
            apply_feedback(ads, event_log)

    # final snapshot
    take_snapshot(day + 1)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  ({n_queries / elapsed:,.0f} queries/s)")

    # ── output ────────────────────────────────────────────────────────────────
    print_stats(ads, event_log)
    plot_rate_evolution(snapshots, ads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=1_000_000,
                        help="Total number of queries to simulate (default: 1,000,000)")
    args = parser.parse_args()
    run_simulation(args.queries)
