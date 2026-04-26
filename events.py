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


def apply_feedback(ads, event_log: EventLog, min_impressions: int = 5) -> List[tuple]:
    """
    Recompute base_ctr and base_cvr from observed event data and update
    Ad objects in place.

    Only updates an ad if it has at least min_impressions — below that
    threshold the estimates are too noisy to trust over the prior.

    Returns a list of (ad_name, old_ctr, new_ctr, old_cvr, new_cvr)
    for any ad that was actually updated.
    """
    stats = event_log.stats_per_ad()
    updates = []

    for ad in ads:
        s = stats.get(ad.id)
        if not s or s["impressions"] < min_impressions:
            continue

        old_ctr = ad.base_ctr
        old_cvr = ad.base_cvr

        ad.base_ctr = s["clicks"] / s["impressions"]
        ad.base_cvr = s["conversions"] / s["clicks"] if s["clicks"] > 0 else ad.base_cvr

        updates.append((ad.name, old_ctr, ad.base_ctr, old_cvr, ad.base_cvr))

    return updates
