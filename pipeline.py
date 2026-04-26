"""
LLM Ad Network — Real Embedding Pipeline

Full pipeline per query:
  1. Embed query locally (sentence-transformers, all-MiniLM-L6-v2)
  2. ANN search via HNSW index → candidate shortlist by cosine similarity
  3. Mock DLRM scorer: relevance + ad stats → pCTR, pCVR → implied_cpc → effective_bid
  4. Second-price auction on effective bids

Advertisers provide only: daily_budget + target_cpa.
The platform derives per-impression bids from those inputs + model predictions.
"""

import pickle
import numpy as np
import hnswlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBED_CACHE = Path(__file__).parent / "ad_embeddings.pkl"
HNSW_CACHE  = Path(__file__).parent / "ad_index.bin"
MODEL_NAME  = "all-MiniLM-L6-v2"
EMBED_DIM   = 384

# HNSW index parameters
# M              : edges per node per layer — higher = better recall, more memory
# ef_construction: neighbor search breadth during insert — higher = better graph, slower build
# ef_search      : neighbor search breadth at query time — tune up for higher recall
HNSW_M               = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH       = 50


# ─── Ad Catalog ───────────────────────────────────────────────────────────────

@dataclass
class Ad:
    id:               str
    name:             str
    description:      str    # text that gets embedded
    category:         str
    target_cpa:       float  # advertiser's max cost-per-acquisition ($) — only input besides budget
    daily_budget:     float
    base_ctr:         float  # historical click-through rate
    base_cvr:         float  # historical conversion rate (given click)
    avg_order_value:  float  # avg revenue per conversion ($)
    embedding:        Optional[np.ndarray] = field(default=None, repr=False)
    daily_spend:      float = 0.0


AD_CATALOG: List[Ad] = [
    # Cloud / DevOps
    Ad("ad_01", "CloudBase Pro",
       "Deploy Python and Node apps to the cloud with one command. Auto-scaling, built-in monitoring, zero config.",
       "cloud", target_cpa=45, daily_budget=400, base_ctr=0.045, base_cvr=0.12, avg_order_value=180),

    Ad("ad_02", "ServerlessNow",
       "Run your backend without managing servers. Serverless functions with instant cold starts. Pay per use.",
       "cloud", target_cpa=38, daily_budget=300, base_ctr=0.038, base_cvr=0.10, avg_order_value=150),

    Ad("ad_03", "GitOps Pipeline",
       "Automate CI/CD workflows. One-click deployments, instant rollback, GitHub and GitLab integration.",
       "devops", target_cpa=30, daily_budget=250, base_ctr=0.042, base_cvr=0.09, avg_order_value=120),

    # ML / AI
    Ad("ad_04", "ModelDeploy",
       "Deploy machine learning models to production in minutes. REST API auto-generated. Supports PyTorch, TensorFlow, scikit-learn.",
       "ml", target_cpa=62, daily_budget=600, base_ctr=0.052, base_cvr=0.14, avg_order_value=250),

    Ad("ad_05", "DataSpark ML",
       "Visual ML pipeline builder. Drag-and-drop feature engineering, automated hyperparameter tuning, no-code model training.",
       "ml", target_cpa=55, daily_budget=500, base_ctr=0.048, base_cvr=0.13, avg_order_value=220),

    Ad("ad_06", "VectorDB Cloud",
       "Managed vector database for AI apps. Store and search billions of embeddings at sub-10ms latency. Built for LLMs.",
       "ml", target_cpa=75, daily_budget=700, base_ctr=0.055, base_cvr=0.15, avg_order_value=300),

    # Security
    Ad("ad_07", "AuthShield",
       "Add authentication to your app in 5 minutes. OAuth2, SAML, MFA, passwordless. SOC2 compliant.",
       "security", target_cpa=40, daily_budget=350, base_ctr=0.040, base_cvr=0.11, avg_order_value=160),

    Ad("ad_08", "SecureScan",
       "Automated security scanning for your codebase. Detect vulnerabilities and leaked secrets before they ship.",
       "security", target_cpa=35, daily_budget=280, base_ctr=0.035, base_cvr=0.09, avg_order_value=140),

    # Data / Analytics
    Ad("ad_09", "QueryFlow",
       "SQL analytics platform for data teams. Connect to any warehouse, build dashboards, schedule reports.",
       "analytics", target_cpa=48, daily_budget=320, base_ctr=0.041, base_cvr=0.10, avg_order_value=190),

    Ad("ad_10", "StreamPulse",
       "Real-time data streaming and analytics. Kafka-compatible, process millions of events per second.",
       "analytics", target_cpa=52, daily_budget=450, base_ctr=0.047, base_cvr=0.12, avg_order_value=210),

    # Developer Tools
    Ad("ad_11", "CodeReview AI",
       "AI-powered code review that catches bugs before your teammates do. GitHub, GitLab, Bitbucket integration.",
       "devtools", target_cpa=25, daily_budget=220, base_ctr=0.043, base_cvr=0.11, avg_order_value=100),

    Ad("ad_12", "DocGen Pro",
       "Auto-generate API documentation from your codebase. OpenAPI, GraphQL, gRPC. Always in sync with your code.",
       "devtools", target_cpa=20, daily_budget=180, base_ctr=0.037, base_cvr=0.08, avg_order_value=80),

    Ad("ad_13", "LogSense",
       "Intelligent log aggregation and alerting. Trace distributed requests, sub-1s search over terabytes of logs.",
       "devtools", target_cpa=32, daily_budget=270, base_ctr=0.039, base_cvr=0.10, avg_order_value=130),

    # LLM / AI Apps
    Ad("ad_14", "PromptLayer",
       "Track, version, and A/B test your LLM prompts. Monitor costs, debug responses. Works with OpenAI and Anthropic.",
       "llm", target_cpa=70, daily_budget=550, base_ctr=0.058, base_cvr=0.16, avg_order_value=280),

    Ad("ad_15", "LangChain Cloud",
       "Deploy LangChain and LlamaIndex apps to production. Managed vector stores, agent tracing, auto-scaling.",
       "llm", target_cpa=80, daily_budget=650, base_ctr=0.060, base_cvr=0.17, avg_order_value=320),

    Ad("ad_16", "EmbedAPI",
       "Fast, cheap text embeddings API. 1M tokens for $0.02. OpenAI-compatible format. 50+ languages.",
       "llm", target_cpa=65, daily_budget=500, base_ctr=0.054, base_cvr=0.14, avg_order_value=260),

    # Education
    Ad("ad_17", "DeepLearn.io",
       "Master machine learning and deep learning. 200+ hours of hands-on courses, real datasets, certificate included.",
       "education", target_cpa=30, daily_budget=200, base_ctr=0.050, base_cvr=0.18, avg_order_value=120),

    Ad("ad_18", "CodeCamp Pro",
       "Become a full-stack developer in 12 weeks. Python, React, databases, cloud. Job placement guarantee.",
       "education", target_cpa=25, daily_budget=160, base_ctr=0.048, base_cvr=0.15, avg_order_value=100),

    # Infrastructure
    Ad("ad_19", "K8sEasy",
       "Kubernetes made simple. Manage clusters visually, auto-scaling policies, cost optimization. No YAML required.",
       "infra", target_cpa=50, daily_budget=380, base_ctr=0.044, base_cvr=0.11, avg_order_value=200),

    Ad("ad_20", "EdgeDeploy",
       "Deploy to 200 edge locations worldwide. Sub-50ms global latency. DDoS protection and WAF included.",
       "infra", target_cpa=42, daily_budget=340, base_ctr=0.040, base_cvr=0.10, avg_order_value=170),
]


# ─── Embedding Layer ──────────────────────────────────────────────────────────

class EmbeddingModel:
    def __init__(self):
        print(f"  Loading {MODEL_NAME}...", end=" ", flush=True)
        self.model = SentenceTransformer(MODEL_NAME)
        dim = self.model.get_embedding_dimension()
        print(f"OK  ({dim}-dim vectors)")

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def load_ad_embeddings(model: EmbeddingModel) -> Tuple[List[Ad], hnswlib.Index]:
    # ── Step 1: embeddings ────────────────────────────────────────────────────
    if EMBED_CACHE.exists():
        print(f"  Loading cached ad embeddings...", end=" ", flush=True)
        with open(EMBED_CACHE, "rb") as f:
            saved = pickle.load(f)
        for ad, emb in zip(AD_CATALOG, saved):
            ad.embedding = emb
        print("OK")
    else:
        print(f"  Embedding {len(AD_CATALOG)} ads (one-time)...", end=" ", flush=True)
        texts = [f"{ad.name}: {ad.description}" for ad in AD_CATALOG]
        embeddings = model.embed_batch(texts)
        for ad, emb in zip(AD_CATALOG, embeddings):
            ad.embedding = emb
        with open(EMBED_CACHE, "wb") as f:
            pickle.dump(embeddings, f)
        print("OK  (cached)")

    # ── Step 2: HNSW index ────────────────────────────────────────────────────
    # Vectors are L2-normalized so inner product == cosine similarity.
    # Space "ip" (inner product) is faster than "cosine" for pre-normalized vecs.
    index = hnswlib.Index(space="ip", dim=EMBED_DIM)

    if HNSW_CACHE.exists():
        print(f"  Loading cached HNSW index...", end=" ", flush=True)
        index.load_index(str(HNSW_CACHE), max_elements=len(AD_CATALOG))
        index.set_ef(HNSW_EF_SEARCH)
        print("OK")
    else:
        print(f"  Building HNSW index  "
              f"(M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})...",
              end=" ", flush=True)
        index.init_index(
            max_elements=len(AD_CATALOG),
            M=HNSW_M,
            ef_construction=HNSW_EF_CONSTRUCTION,
        )
        # Each ad's integer label = its position in AD_CATALOG
        embeddings = np.stack([ad.embedding for ad in AD_CATALOG])
        index.add_items(embeddings, ids=list(range(len(AD_CATALOG))))
        index.set_ef(HNSW_EF_SEARCH)
        index.save_index(str(HNSW_CACHE))
        print("OK  (cached)")

    return AD_CATALOG, index


# ─── Stage 1: Vector Matching ─────────────────────────────────────────────────

RELEVANCE_THRESHOLD = 0.28

def match_candidates(
    query_emb: np.ndarray,
    ads: List[Ad],
    index: hnswlib.Index,
    threshold: float = RELEVANCE_THRESHOLD,
) -> List[Tuple[Ad, float]]:
    """
    ANN search via HNSW index.

    knn_query returns the k approximate nearest neighbors by navigating the
    graph — O(log n) vs O(n) brute force. Labels are AD_CATALOG positions.
    Similarities are inner products (== cosine sim for normalized vectors).

    Budget-exhausted ads still appear in the index; we filter them out here
    after retrieval. In production you'd rebuild or soft-delete from the index.
    """
    k = min(len(ads), 15)   # retrieve top-15, then threshold-filter
    labels, distances = index.knn_query(query_emb.reshape(1, -1), k=k)

    candidates = []
    for label, dist in zip(labels[0], distances[0]):
        ad  = ads[int(label)]
        # hnswlib "ip" space returns 1 - inner_product as distance, so flip it
        sim = round(1.0 - float(dist), 4)
        if sim < threshold:
            continue
        if ad.daily_spend >= ad.daily_budget:
            continue
        candidates.append((ad, sim))

    return sorted(candidates, key=lambda x: x[1], reverse=True)


# ─── Stage 2: DLRM-style Scorer ──────────────────────────────────────────────

@dataclass
class AdScore:
    ad:            Ad
    relevance:     float   # cosine similarity from matching stage
    p_ctr:         float   # predicted click-through rate
    p_cvr:         float   # predicted conversion rate (given click)
    exp_value:     float   # p_cvr × avg_order_value  ($)
    implied_cpc:   float   # target_cpa × p_cvr — platform-derived cost-per-click ceiling
    effective_bid: float   # implied_cpc × p_ctr — expected cost per impression (enters auction)


def score_candidates(
    candidates: List[Tuple[Ad, float]],
    rng: np.random.Generator,
) -> List[AdScore]:
    """
    Mock DLRM scorer.

    Advertisers provide only target_cpa and daily_budget.
    The platform derives a per-impression bid in two steps:

      Step 1 — implied_cpc = target_cpa × p_cvr
        "If I pay this much per click, and p_cvr% of clicks convert,
         my cost per acquisition equals target_cpa."

      Step 2 — effective_bid = implied_cpc × p_ctr
        "Expected cost to the advertiser per impression shown."
        This is what the auction ranks on.
    """
    scores = []
    for ad, relevance in candidates:
        noise = float(rng.normal(1.0, 0.06))   # ±6% prediction noise

        rel_boost_ctr = 1.0 + relevance * 0.9
        rel_boost_cvr = 1.0 + relevance * 0.5

        p_ctr = float(np.clip(ad.base_ctr * rel_boost_ctr * noise, 0.001, 0.99))
        p_cvr = float(np.clip(ad.base_cvr * rel_boost_cvr * noise, 0.001, 0.99))

        exp_value    = round(p_cvr * ad.avg_order_value, 4)
        implied_cpc  = round(ad.target_cpa * p_cvr, 6)
        effective_bid = round(implied_cpc * p_ctr, 6)

        scores.append(AdScore(ad, relevance, round(p_ctr, 4), round(p_cvr, 4),
                              exp_value, implied_cpc, effective_bid))

    return sorted(scores, key=lambda s: s.effective_bid, reverse=True)


# ─── Stage 4: Second-Price Auction ───────────────────────────────────────────

RESERVE = 0.0001   # floor price per impression ($)

@dataclass
class AuctionResult:
    winner:        Optional[AdScore]
    price_impr:    float   # price per impression paid by winner
    price_click:   float   # price_impr / p_ctr  (what advertiser sees as CPC)
    all_scores:    List[AdScore]


def run_auction(scores: List[AdScore]) -> AuctionResult:
    """
    Highest effective_bid wins.
    Winner pays the second-highest effective_bid (second-price rule).
    Price is then converted back to cost-per-click by dividing by pCTR,
    so the advertiser's dashboard shows a familiar $/click number.
    """
    if not scores or scores[0].effective_bid < RESERVE:
        return AuctionResult(None, 0.0, 0.0, scores)

    winner      = scores[0]
    second_bid  = scores[1].effective_bid if len(scores) > 1 else RESERVE
    price_impr  = max(RESERVE, second_bid)
    price_click = price_impr / (winner.p_ctr + 1e-9)

    winner.ad.daily_spend = round(winner.ad.daily_spend + price_impr, 6)
    return AuctionResult(winner, round(price_impr, 6), round(price_click, 4), scores)


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class AdPipeline:

    def __init__(self):
        print("\n[ Ad Network Pipeline ] Initializing\n")
        self.embed_model = EmbeddingModel()
        self.ads, self.index = load_ad_embeddings(self.embed_model)
        self.query_n     = 0
        self._rng        = np.random.default_rng(0)
        print("\n  Ready — type a query to run the pipeline.\n")

    def run(self, query: str) -> AuctionResult:
        self.query_n += 1
        W = 66

        print(f"\n{'═'*W}")
        print(f"  Query #{self.query_n}: \"{query}\"")
        print(f"{'═'*W}")

        # ── Stage 1: embed + match ──────────────────────────────────────────
        q_emb      = self.embed_model.embed(query)
        candidates = match_candidates(q_emb, self.ads, self.index)

        print(f"\n  STAGE 1 · Embedding + Vector Matching")
        print(f"  Cosine similarity threshold: {RELEVANCE_THRESHOLD}")
        print(f"  {'Ad':<26} {'Category':<12} {'Similarity':>10}")
        print(f"  {'─'*50}")
        if candidates:
            for ad, sim in candidates:
                bar = "█" * int(sim * 20)
                print(f"  {ad.name:<26} {ad.category:<12} {sim:>10.4f}  {bar}")
        else:
            print("  No candidates above threshold — no ad served.")
            return AuctionResult(None, 0.0, 0.0, [])

        # ── Stage 2: DLRM scoring ───────────────────────────────────────────
        scores = score_candidates(candidates, self._rng)

        print(f"\n  STAGE 2 · DLRM Scoring  (relevance → pCTR, pCVR → implied_cpc → effective_bid)")
        print(f"  {'Ad':<26} {'Rel':>5} {'pCTR':>6} {'pCVR':>6} {'EV($)':>7} {'impCPC':>8} {'EffBid':>8}")
        print(f"  {'─'*66}")
        for s in scores:
            marker = " ◄ winner" if s is scores[0] else ""
            print(f"  {s.ad.name:<26} {s.relevance:>5.3f} {s.p_ctr:>6.3f} "
                  f"{s.p_cvr:>6.3f} {s.exp_value:>7.2f} "
                  f"{s.implied_cpc:>8.4f} {s.effective_bid:>8.5f}{marker}")

        w0 = scores[0]
        print(f"\n  Bid derivation for '{w0.ad.name}':")
        print(f"    implied_cpc  = target_cpa × pCVR  = ${w0.ad.target_cpa} × {w0.p_cvr:.3f} = ${w0.implied_cpc:.4f}")
        print(f"    effective_bid = implied_cpc × pCTR = ${w0.implied_cpc:.4f} × {w0.p_ctr:.3f} = ${w0.effective_bid:.5f}/impr")

        # ── Stage 3: auction ────────────────────────────────────────────────
        result = run_auction(scores)

        print(f"\n  STAGE 3 · Second-Price Auction")
        if result.winner:
            w = result.winner
            second = scores[1].effective_bid if len(scores) > 1 else RESERVE
            print(f"  Winner         : {w.ad.name}")
            print(f"  Winning bid    : ${w.effective_bid:.5f}/impr")
            print(f"  Second-highest : ${second:.5f}/impr  ← winner pays this")
            print(f"  Price/impr     : ${result.price_impr:.5f}")
            print(f"  Price/click    : ${result.price_click:.4f}  (implied_cpc: ${w.implied_cpc:.4f}, target_cpa: ${w.ad.target_cpa})")
            print(f"  Daily spend    : ${w.ad.daily_spend:.4f} / ${w.ad.daily_budget:.0f}")

        else:
            print("  No winner (no candidates or all below reserve price).")

        print(f"\n{'─'*W}")
        return result


# ─── Sample Queries ───────────────────────────────────────────────────────────

SAMPLES = [
    "How do I deploy a machine learning model to production?",
    "What's the best way to add user authentication to my app?",
    "I want to build a chatbot using an LLM",
    "How do I set up Kubernetes for my startup?",
    "What is a vector database and when should I use one?",
    "I need to learn Python for data science",
    "How can I make my API faster and more reliable?",
    "How do I detect security vulnerabilities in my code?",
]


if __name__ == "__main__":
    pipeline = AdPipeline()

    print("  Press Enter for a sample query.  Type your own query.  'q' to quit.\n")
    sample_i = 0

    while True:
        try:
            raw = input("  Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if raw.lower() in ("q", "quit", "exit"):
            print("  Bye.")
            break

        if not raw:
            query = SAMPLES[sample_i % len(SAMPLES)]
            sample_i += 1
            print(f"  (sample: \"{query}\")")
        else:
            query = raw

        pipeline.run(query)
