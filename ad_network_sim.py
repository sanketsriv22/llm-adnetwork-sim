"""
LLM Ad Network Simulator
Demonstrates: contextual matching → eligibility filtering → paced second-price auction
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Advertiser:
    id: str
    name: str
    daily_budget: float
    max_bid: float          # latent willingness-to-pay per impression
    keywords: List[str]     # targeting signal (proxy for campaign embeddings)
    # runtime state
    spend: float = 0.0
    wins: int = 0
    pacing_multiplier: float = 1.0
    spend_history: List[float] = field(default_factory=list)
    pacing_history: List[float] = field(default_factory=list)

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.daily_budget - self.spend)

    @property
    def utilization(self) -> float:
        return self.spend / self.daily_budget if self.daily_budget else 0.0


@dataclass
class Query:
    text: str
    topics: List[str]   # extracted intent tokens (proxy for LLM-derived context)


@dataclass
class AuctionResult:
    winner: Optional[Advertiser]
    price_paid: float
    eligible: List[Tuple[Advertiser, float]]   # (advertiser, relevance_score)
    effective_bids: Dict[str, float]


# ─── Stage 1: Contextual Matching ─────────────────────────────────────────────

class MatchingEngine:
    """
    Simulates the LLM-context → ad relevance scoring step.
    In production: embed query with an encoder model, do ANN search over
    campaign embeddings, then run an ML scorer. Here we use Jaccard similarity
    on topic tokens as a stand-in for cosine similarity on embeddings.
    """
    RELEVANCE_THRESHOLD = 0.12
    NOISE_STD = 0.04    # simulate embedding score variance

    def score(self, query: Query, adv: Advertiser) -> float:
        q_set = set(t.lower() for t in query.topics)
        a_set = set(k.lower() for k in adv.keywords)
        if not q_set or not a_set:
            return 0.0
        overlap = len(q_set & a_set)
        union = len(q_set | a_set)
        base = overlap / union
        return float(np.clip(base + np.random.normal(0, self.NOISE_STD), 0.0, 1.0))

    def get_candidates(
        self, query: Query, advertisers: List[Advertiser]
    ) -> List[Tuple[Advertiser, float]]:
        """Return (advertiser, relevance) pairs that pass threshold and have budget."""
        candidates = []
        for adv in advertisers:
            if adv.budget_remaining <= 0:
                continue
            rel = self.score(query, adv)
            if rel >= self.RELEVANCE_THRESHOLD:
                candidates.append((adv, rel))
        return candidates


# ─── Stage 2: Budget Pacing ────────────────────────────────────────────────────

class PacingController:
    """
    Adaptive throttle: if an advertiser is spending faster than the uniform
    daily target, reduce their pacing multiplier; if slower, relax it.
    Updates every N auctions to amortize overhead (mirrors production systems
    that update pacing state every few seconds rather than per-impression).
    """
    UPDATE_INTERVAL = 25

    def __init__(self, total_auctions: int):
        self.total = total_auctions
        self.tick = 0

    def update(self, advertisers: List[Advertiser]):
        self.tick += 1
        if self.tick % self.UPDATE_INTERVAL != 0:
            return
        time_fraction = self.tick / self.total
        for adv in advertisers:
            if adv.daily_budget == 0:
                continue
            pace_ratio = adv.utilization / (time_fraction + 1e-9)
            if pace_ratio > 1.1:        # overpacing → throttle down
                adv.pacing_multiplier = max(0.05, adv.pacing_multiplier * 0.80)
            elif pace_ratio < 0.85:     # underpacing → open up
                adv.pacing_multiplier = min(1.0, adv.pacing_multiplier * 1.15)
            adv.pacing_history.append(adv.pacing_multiplier)


# ─── Stage 3: Second-Price Auction ────────────────────────────────────────────

class SecondPriceAuction:
    """
    Generalized second-price (GSP) auction with quality normalization.

    Effective bid:  b_i = pacing_i × max_bid_i × relevance_i
    Winner:         argmax(b_i)
    Price paid:     second_highest_effective_bid / winner_relevance
                    (quality-normalized so higher-relevance ads pay less for
                     equivalent effective bids — incentivizes relevance)
    """
    RESERVE = 0.005     # floor price in $

    def run(self, candidates: List[Tuple[Advertiser, float]]) -> AuctionResult:
        if not candidates:
            return AuctionResult(None, 0.0, [], {})

        scored = []
        effective_bids: Dict[str, float] = {}
        for adv, rel in candidates:
            eb = adv.pacing_multiplier * adv.max_bid * rel
            scored.append((adv, eb, rel))
            effective_bids[adv.id] = round(eb, 4)

        scored.sort(key=lambda x: x[1], reverse=True)
        winner_adv, winner_eb, winner_rel = scored[0]

        if winner_eb < self.RESERVE:
            return AuctionResult(None, 0.0, candidates, effective_bids)

        if len(scored) > 1:
            second_eb = scored[1][1]
            price = second_eb / (winner_rel + 1e-9)
        else:
            price = self.RESERVE

        price = float(np.clip(price, self.RESERVE, winner_adv.max_bid))
        return AuctionResult(winner_adv, price, candidates, effective_bids)


# ─── Simulator ────────────────────────────────────────────────────────────────

class AdNetworkSimulator:

    def __init__(self, advertisers: List[Advertiser], queries: List[Query]):
        self.advertisers = advertisers
        self.queries = queries
        self.matcher = MatchingEngine()
        self.auction = SecondPriceAuction()
        self.pacer = PacingController(len(queries))
        self.log: List[dict] = []

    def run(self, verbose_first_n: int = 5):
        print(f"\nRunning {len(self.queries)} auctions across {len(self.advertisers)} advertisers...\n")

        for i, query in enumerate(self.queries):
            candidates = self.matcher.get_candidates(query, self.advertisers)
            result = self.auction.run(candidates)

            if result.winner:
                w = result.winner
                w.spend = min(w.spend + result.price_paid, w.daily_budget)
                w.wins += 1

            for adv in self.advertisers:
                adv.spend_history.append(adv.spend)

            self.log.append({
                "i": i,
                "query": query.text,
                "eligible": len(candidates),
                "winner": result.winner.name if result.winner else "—",
                "price": result.price_paid,
            })

            self.pacer.update(self.advertisers)

            if i < verbose_first_n:
                self._print_auction_trace(i, query, candidates, result)

    def _print_auction_trace(self, i, query, candidates, result):
        print(f"  Auction {i+1}: \"{query.text}\"")
        print(f"  Topics: {query.topics}")
        print(f"  Eligible bidders ({len(candidates)}):")
        for adv, rel in sorted(candidates, key=lambda x: x[1], reverse=True):
            eb = result.effective_bids.get(adv.id, 0)
            print(f"    {adv.name:<20} rel={rel:.2f}  pace={adv.pacing_multiplier:.2f}  "
                  f"max_bid=${adv.max_bid:.2f}  eff_bid=${eb:.3f}")
        if result.winner:
            print(f"  Winner: {result.winner.name}  → pays ${result.price_paid:.4f}")
        else:
            print(f"  No winner (below reserve or no candidates)")
        print()


# ─── Results & Visualization ──────────────────────────────────────────────────

def print_summary(sim: AdNetworkSimulator):
    n = len(sim.queries)
    filled = sum(1 for r in sim.log if r["winner"] != "—")
    avg_elig = np.mean([r["eligible"] for r in sim.log])
    prices = [r["price"] for r in sim.log if r["price"] > 0]

    print("=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"  Total auctions  : {n}")
    print(f"  Filled slots    : {filled} ({filled/n*100:.1f}%)")
    print(f"  Avg eligible    : {avg_elig:.1f} bidders/auction")
    print(f"  Avg clearing $  : ${np.mean(prices):.4f}" if prices else "")
    print()
    print(f"  {'Advertiser':<20} {'Budget':>7} {'Spend':>7} {'Util%':>6} "
          f"{'Wins':>5} {'WinRate%':>9} {'AvgCPC':>8} {'FinalPace':>10}")
    print("  " + "-" * 74)
    for adv in sorted(sim.advertisers, key=lambda x: x.wins, reverse=True):
        wr = adv.wins / n * 100
        cpc = adv.spend / max(1, adv.wins)
        print(f"  {adv.name:<20} ${adv.daily_budget:>5.0f}  ${adv.spend:>5.1f} "
              f"{adv.utilization*100:>5.1f}% {adv.wins:>5} {wr:>8.1f}% "
              f"${cpc:>6.3f} {adv.pacing_multiplier:>9.2f}")
    print()


def plot_results(sim: AdNetworkSimulator):
    advertisers = sim.advertisers
    n_auctions = len(sim.queries)
    x = list(range(n_auctions))

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(advertisers)))

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("LLM Ad Network Simulation", fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # 1. Spend curves
    ax1 = fig.add_subplot(gs[0, 0])
    for adv, c in zip(advertisers, colors):
        ax1.plot(x, adv.spend_history, label=adv.name, color=c, linewidth=1.4, alpha=0.85)
        # mark budget exhaustion
        exhausted = next((i for i, s in enumerate(adv.spend_history)
                          if s >= adv.daily_budget * 0.999), None)
        if exhausted:
            ax1.axvline(exhausted, color=c, linestyle=":", alpha=0.4, linewidth=0.8)
    ax1.set_title("Cumulative Spend Over Time", fontweight="bold")
    ax1.set_xlabel("Auction #")
    ax1.set_ylabel("Spend ($)")
    ax1.legend(fontsize=6.5, ncol=2)
    ax1.grid(alpha=0.25)

    # 2. Pacing multiplier over time (sampled at update intervals)
    ax2 = fig.add_subplot(gs[0, 1])
    for adv, c in zip(advertisers, colors):
        if adv.pacing_history:
            xs = np.linspace(0, n_auctions, len(adv.pacing_history))
            ax2.plot(xs, adv.pacing_history, label=adv.name, color=c, linewidth=1.4, alpha=0.85)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="max pace")
    ax2.set_title("Pacing Multiplier Over Time", fontweight="bold")
    ax2.set_xlabel("Auction #")
    ax2.set_ylabel("Multiplier")
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=6.5, ncol=2)
    ax2.grid(alpha=0.25)

    # 3. Win rate vs budget (bubble chart)
    ax3 = fig.add_subplot(gs[1, 0])
    for adv, c in zip(advertisers, colors):
        wr = adv.wins / n_auctions * 100
        ax3.scatter(adv.daily_budget, wr, s=adv.max_bid * 80,
                    color=c, alpha=0.85, edgecolors="white", linewidths=0.6)
        ax3.annotate(adv.name.split()[0], (adv.daily_budget, wr),
                     fontsize=7, ha="center", va="bottom")
    ax3.set_title("Win Rate vs Budget\n(bubble size = max bid)", fontweight="bold")
    ax3.set_xlabel("Daily Budget ($)")
    ax3.set_ylabel("Win Rate (%)")
    ax3.grid(alpha=0.25)

    # 4. Budget utilization bar
    ax4 = fig.add_subplot(gs[1, 1])
    names = [adv.name for adv in advertisers]
    utils = [adv.utilization * 100 for adv in advertisers]
    bars = ax4.barh(names, utils, color=colors, alpha=0.85, edgecolor="white")
    ax4.axvline(100, color="crimson", linestyle="--", linewidth=1, label="100% (exhausted)")
    for bar, u in zip(bars, utils):
        ax4.text(min(u + 1, 105), bar.get_y() + bar.get_height() / 2,
                 f"{u:.0f}%", va="center", fontsize=7)
    ax4.set_title("Budget Utilization", fontweight="bold")
    ax4.set_xlabel("Utilization (%)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.25, axis="x")

    plt.savefig("ad_network_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → ad_network_simulation.png")


# ─── Sample Data ──────────────────────────────────────────────────────────────

def make_advertisers() -> List[Advertiser]:
    return [
        Advertiser("a1",  "TechCloud Pro",   500,  2.50, ["cloud", "saas", "api", "infrastructure", "devops"]),
        Advertiser("a2",  "CodeLearn",        200,  1.80, ["learn", "coding", "python", "tutorial", "course"]),
        Advertiser("a3",  "DataSpark",        800,  3.20, ["data", "analytics", "ml", "ai", "pipeline"]),
        Advertiser("a4",  "DevTools Inc",     150,  1.20, ["developer", "tools", "sdk", "api", "debug"]),
        Advertiser("a5",  "AIAssist",        1000,  4.00, ["ai", "assistant", "llm", "chatbot", "automation"]),
        Advertiser("a6",  "CloudHost",        350,  2.00, ["hosting", "cloud", "server", "deployment", "infra"]),
        Advertiser("a7",  "PythonPro",        180,  1.50, ["python", "programming", "code", "developer", "script"]),
        Advertiser("a8",  "MLPlatform",       600,  3.50, ["ml", "machine learning", "model", "training", "ai"]),
        Advertiser("a9",  "SecurityFirst",    400,  2.80, ["security", "auth", "api", "oauth", "encryption"]),
        Advertiser("a10", "StartupBoost",     100,  0.90, ["startup", "saas", "mvp", "launch", "product"]),
    ]


QUERY_POOL = [
    ("How do I deploy a Python app to the cloud?",          ["python", "cloud", "deployment", "infrastructure"]),
    ("Best practices for ML model training at scale",       ["ml", "machine learning", "model", "training", "ai"]),
    ("How to integrate an LLM into my chatbot?",            ["llm", "ai", "chatbot", "integration", "api"]),
    ("Secure API authentication with OAuth 2.0",            ["security", "auth", "api", "oauth"]),
    ("Getting started with data analytics pipelines",       ["data", "analytics", "ml", "pipeline"]),
    ("Python tutorial for absolute beginners",              ["python", "tutorial", "learn", "programming", "course"]),
    ("Building a SaaS product from scratch",                ["saas", "startup", "mvp", "product", "cloud"]),
    ("How does vector search work for RAG?",                ["ml", "ai", "data", "python"]),
    ("Setting up a CI/CD pipeline with GitHub Actions",     ["developer", "tools", "infrastructure", "devops"]),
    ("What is prompt engineering and how do I get started?",["ai", "llm", "chatbot", "developer"]),
    ("Compare cloud providers for ML workloads",            ["cloud", "ml", "infrastructure", "ai"]),
    ("How to monitor model drift in production",            ["ml", "model", "ai", "data"]),
    ("Debug a slow Python API endpoint",                    ["python", "api", "developer", "debug"]),
    ("Encrypt user data at rest in a SaaS app",             ["security", "saas", "encryption", "cloud"]),
    ("Automate repetitive dev tasks with Python scripts",   ["python", "automation", "developer", "script"]),
]


def make_queries(n: int = 2000) -> List[Query]:
    return [Query(text=t, topics=top) for t, top in
            [random.choice(QUERY_POOL) for _ in range(n)]]


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    advertisers = make_advertisers()
    queries = make_queries(2000)

    sim = AdNetworkSimulator(advertisers, queries)
    sim.run(verbose_first_n=4)

    print_summary(sim)
    plot_results(sim)
