import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Impression:
    id:     str
    ad_id:  str
    query:  str


@dataclass
class Click:
    id:            str
    impression_id: str


@dataclass
class Conversion:
    id:            str
    impression_id: str
    value:         float


class EventLog:
    """
    In-memory store for impression, click, and conversion events.

    Attribution model: last-click within session.
    Each conversion is tied directly to an impression_id, which is determined
    at log time by whoever calls log_conversion — in production this would be
    the attribution engine joining on click IDs and time windows.
    """

    def __init__(self):
        self.impressions:  Dict[str, Impression] = {}
        self.clicks:       Dict[str, Click]      = {}  # keyed by impression_id
        self.conversions:  List[Conversion]      = []

    def log_impression(self, ad_id: str, query: str) -> str:
        iid = uuid.uuid4().hex[:8]
        self.impressions[iid] = Impression(iid, ad_id, query)
        return iid

    def log_click(self, impression_id: str) -> Optional[str]:
        if impression_id not in self.impressions:
            return None
        cid = uuid.uuid4().hex[:8]
        self.clicks[impression_id] = Click(cid, impression_id)
        return cid

    def log_conversion(self, impression_id: str, value: float) -> Optional[str]:
        if impression_id not in self.impressions:
            return None
        cvid = uuid.uuid4().hex[:8]
        self.conversions.append(Conversion(cvid, impression_id, value))
        return cvid

    def stats_per_ad(self) -> Dict[str, dict]:
        """Aggregate impression/click/conversion counts per ad_id."""
        stats: Dict[str, dict] = {}

        for imp in self.impressions.values():
            if imp.ad_id not in stats:
                stats[imp.ad_id] = {"impressions": 0, "clicks": 0, "conversions": 0, "revenue": 0.0}
            stats[imp.ad_id]["impressions"] += 1

        for imp_id, click in self.clicks.items():
            ad_id = self.impressions[imp_id].ad_id
            stats[ad_id]["clicks"] += 1

        for conv in self.conversions:
            ad_id = self.impressions[conv.impression_id].ad_id
            stats[ad_id]["conversions"] += 1
            stats[ad_id]["revenue"] += conv.value

        return stats

    @property
    def total_impressions(self) -> int:
        return len(self.impressions)

    @property
    def total_clicks(self) -> int:
        return len(self.clicks)

    @property
    def total_conversions(self) -> int:
        return len(self.conversions)


PRIOR_STRENGTH = 100   # treat initial base_ctr/base_cvr as if observed from this many impressions


def apply_feedback(ads, event_log: EventLog) -> List[tuple]:
    """
    Bayesian update of base_ctr and base_cvr from observed event data.

    Instead of hard-replacing rates with observed values (which explodes with
    sparse data), we blend observed counts with a prior anchored to the ad's
    original base_ctr / base_cvr:

        new_ctr = (prior_clicks + observed_clicks) / (prior_strength + observed_impressions)

    With few impressions the estimate barely moves off the prior.
    With hundreds of impressions the observed data dominates.
    This prevents runaway values (0.99 or 0.001) from noisy early updates.

    Returns a list of (ad_name, old_ctr, new_ctr, old_cvr, new_cvr).
    """
    stats = event_log.stats_per_ad()
    updates = []

    for ad in ads:
        s = stats.get(ad.id)
        if not s:
            continue

        old_ctr = ad.base_ctr
        old_cvr = ad.base_cvr

        prior_clicks = ad._init_ctr * PRIOR_STRENGTH
        prior_convs  = ad._init_cvr * PRIOR_STRENGTH

        ad.base_ctr = (prior_clicks + s["clicks"]) / (PRIOR_STRENGTH + s["impressions"])
        ad.base_cvr = (prior_convs  + s["conversions"]) / (PRIOR_STRENGTH + s["clicks"]) \
                      if s["clicks"] > 0 else ad.base_cvr

        updates.append((ad.name, old_ctr, ad.base_ctr, old_cvr, ad.base_cvr))

    return updates
