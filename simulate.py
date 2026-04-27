"""
Monte Carlo simulation over the ad network pipeline.

Usage:
    python simulate.py                  # default 1,000,000 queries
    python simulate.py --queries 50000  # quick smoke test

Outputs:
    - per-ad stats table printed to stdout
"""

import argparse
import time
import numpy as np

from pipeline import match_candidates, score_candidates, run_auction
from events import apply_feedback
from queries import QUERIES


# ── Simulation constants ──────────────────────────────────────────────────────

IMPRESSIONS_PER_DAY = 10_000
FEEDBACK_INTERVAL   = 1_000


# ── Stats ─────────────────────────────────────────────────────────────────────

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


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_simulation(n_queries: int):
    print(f"\n[ Monte Carlo Simulation ]  n={n_queries:,}  "
          f"days={n_queries // IMPRESSIONS_PER_DAY}  "
          f"feedback every {FEEDBACK_INTERVAL:,} impr\n")

    from pipeline import EmbeddingModel, load_ad_embeddings
    from events import EventLog

    embed_model = EmbeddingModel()
    ads, index  = load_ad_embeddings(embed_model)
    event_log   = EventLog()
    rng         = np.random.default_rng(42)

    print(f"\n  Pre-embedding {len(QUERIES)} sample queries...", end=" ", flush=True)
    query_embs = embed_model.embed_batch(QUERIES)
    print("OK")

    t0     = time.time()
    served = 0
    day    = 0

    print(f"\n  Running...")

    for i in range(n_queries):
        if i > 0 and i % IMPRESSIONS_PER_DAY == 0:
            day += 1
            for ad in ads:
                ad.daily_spend = 0.0
            elapsed = time.time() - t0
            pct = 100 * i / n_queries
            print(f"  Day {day:>3}  ({pct:.0f}%)  served={served:,}  "
                  f"impr={event_log.total_impressions:,}  "
                  f"elapsed={elapsed:.1f}s")

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
        served += 1

        clicked   = rng.random() < w.ad.base_ctr
        converted = clicked and (rng.random() < w.ad.base_cvr)
        if clicked:
            event_log.log_click(imp_id)
        if converted:
            event_log.log_conversion(imp_id, w.ad.avg_order_value)

        if event_log.total_impressions % FEEDBACK_INTERVAL == 0:
            apply_feedback(ads, event_log)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  ({n_queries / elapsed:,.0f} queries/s)")
    print_stats(ads, event_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=1_000_000,
                        help="Total number of queries to simulate (default: 1,000,000)")
    args = parser.parse_args()
    run_simulation(args.queries)
