"""graph_analytics.py — Graph analytics on the Healthcare Provider/Patient KG.

Implements graph-based insights for:
  - Provider network analysis (centrality, community detection, referral rings)
  - Population health segmentation (patient clustering by condition profile)
  - Utilisation patterns (high-cost patient identification)
  - Fraud signal features (graph-derived anomaly indicators)

Uses NetworkX graphs exported by kg_builder.py.

Usage:
    python src/graph_analytics.py
    python src/graph_analytics.py --kg-dir kg/ --data-dir data/ --out-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("graph-analytics")


# ─────────────────────────────────────────────────────────────────────
#  1. Provider Network Analytics
# ─────────────────────────────────────────────────────────────────────

def provider_network_analysis(G: nx.DiGraph) -> dict:
    """
    Compute centrality measures, detect communities, and flag
    potential fraud-ring providers in the referral network.
    """
    results = {}

    # ── Centrality measures ────────────────────────────────────
    log.info("Computing centrality measures...")
    in_degree  = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))
    try:
        pagerank = nx.pagerank(G, weight="weight", alpha=0.85)
    except Exception:
        pagerank = {n: 1/G.number_of_nodes() for n in G.nodes}

    # Betweenness on undirected version (too slow on large graphs → sample)
    G_und = G.to_undirected()
    if G.number_of_nodes() <= 500:
        betweenness = nx.betweenness_centrality(G_und, weight="weight", normalized=True)
    else:
        sample = min(200, G.number_of_nodes())
        betweenness = nx.betweenness_centrality(G_und, k=sample, weight="weight")

    results["centrality"] = {
        node: {
            "in_degree_weighted":  round(in_degree.get(node, 0), 2),
            "out_degree_weighted": round(out_degree.get(node, 0), 2),
            "pagerank":            round(pagerank.get(node, 0), 6),
            "betweenness":         round(betweenness.get(node, 0), 6),
            "is_fraud_ring":       G.nodes[node].get("is_fraud_ring", False),
            "specialty":           G.nodes[node].get("specialty", "unknown"),
        }
        for node in G.nodes
    }

    # ── Community detection (Louvain on undirected) ────────────
    log.info("Running community detection...")
    try:
        communities = nx.community.louvain_communities(G_und, weight="weight", seed=42)
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
        results["communities"] = community_map
        log.info("  Found %d communities", len(communities))

        # Flag communities with high fraud-ring concentration
        fraud_communities: dict[int, dict] = {}
        for i, comm in enumerate(communities):
            fraud_count = sum(1 for n in comm if G.nodes[n].get("is_fraud_ring", False))
            if fraud_count > 0:
                fraud_communities[i] = {
                    "size":        len(comm),
                    "fraud_members": fraud_count,
                    "fraud_ratio":   round(fraud_count / len(comm), 3),
                }
        results["high_risk_communities"] = fraud_communities

    except Exception as e:
        log.warning("Community detection failed: %s", e)
        results["communities"] = {}

    # ── Referral concentration (Gini coefficient per provider) ─
    log.info("Computing referral concentration...")
    concentration: dict[str, float] = {}
    for node in G.nodes:
        edges = list(G.out_edges(node, data="weight"))
        if len(edges) < 2:
            concentration[node] = 0.0
            continue
        weights = sorted([d if d else 1 for _, _, d in edges])
        n = len(weights)
        gini = (2 * sum((i + 1) * w for i, w in enumerate(weights))
                / (n * sum(weights))) - (n + 1) / n
        concentration[node] = round(max(0.0, gini), 4)

    results["referral_concentration"] = concentration

    # ── Top providers by PageRank ─────────────────────────────
    top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    results["top_providers_pagerank"] = [
        {"provider_id": pid,
         "name": G.nodes[pid].get("name", pid),
         "specialty": G.nodes[pid].get("specialty", ""),
         "pagerank": round(pr, 6),
         "is_fraud_ring": G.nodes[pid].get("is_fraud_ring", False)}
        for pid, pr in top_pr
    ]

    return results


# ─────────────────────────────────────────────────────────────────────
#  2. Patient Risk Stratification
# ─────────────────────────────────────────────────────────────────────

def patient_risk_stratification(patients: list[dict], claims: list[dict]) -> dict:
    """
    Segment patients into risk tiers using HCC score + claim spend.
    Returns segment assignments and cohort statistics.
    """
    # Aggregate per-patient claim spend
    spend: dict[str, float] = {}
    claim_count: dict[str, int] = {}
    for clm in claims:
        pid = clm["patient_id"]
        spend[pid]       = spend.get(pid, 0) + clm["paid_amount"]
        claim_count[pid] = claim_count.get(pid, 0) + 1

    # Build feature matrix
    records = []
    for pat in patients:
        pid = pat["patient_id"]
        records.append({
            "patient_id":    pid,
            "hcc_risk":      pat["hcc_risk_score"],
            "annual_spend":  spend.get(pid, 0),
            "claim_count":   claim_count.get(pid, 0),
            "n_conditions":  len(pat["conditions"]),
            "n_chronic":     sum(1 for c in pat["conditions"] if c[2] == "chronic"),
        })

    # Rule-based risk tiering (mirrors payer stratification logic)
    tiers: dict[str, str] = {}
    cohort_stats: dict[str, dict] = {"high": {}, "medium": {}, "low": {}}

    for r in records:
        if r["hcc_risk"] >= 1.5 or r["annual_spend"] >= 5000:
            tier = "high"
        elif r["hcc_risk"] >= 0.7 or r["annual_spend"] >= 1000:
            tier = "medium"
        else:
            tier = "low"
        tiers[r["patient_id"]] = tier

    # Cohort aggregates
    for tier in ["high", "medium", "low"]:
        cohort = [r for r in records if tiers[r["patient_id"]] == tier]
        if not cohort:
            continue
        cohort_stats[tier] = {
            "count":            len(cohort),
            "avg_hcc_risk":     round(np.mean([c["hcc_risk"] for c in cohort]), 3),
            "avg_annual_spend": round(np.mean([c["annual_spend"] for c in cohort]), 2),
            "avg_claim_count":  round(np.mean([c["claim_count"] for c in cohort]), 1),
            "avg_chronic_conds":round(np.mean([c["n_chronic"] for c in cohort]), 2),
        }

    log.info("Risk stratification: high=%d medium=%d low=%d",
             sum(1 for v in tiers.values() if v == "high"),
             sum(1 for v in tiers.values() if v == "medium"),
             sum(1 for v in tiers.values() if v == "low"))

    return {"tiers": tiers, "cohort_stats": cohort_stats}


# ─────────────────────────────────────────────────────────────────────
#  3. Fraud Graph Features
# ─────────────────────────────────────────────────────────────────────

def compute_fraud_features(
    claims: list[dict],
    provider_analytics: dict,
) -> list[dict]:
    """
    Build a feature vector for each claim combining:
      - Claim-level features (amount, status, type)
      - Provider-level graph features (PageRank, betweenness, referral concentration)
      - Pattern flags (upcoding, unbundling)

    Returns list of dicts ready for ML training.
    """
    centrality   = provider_analytics.get("centrality", {})
    concentration= provider_analytics.get("referral_concentration", {})

    # Provider-level statistics for z-score computation
    prov_amounts: dict[str, list[float]] = {}
    for clm in claims:
        pid = clm["provider_id"]
        prov_amounts.setdefault(pid, []).append(clm["billed_amount"])

    prov_mean = {p: np.mean(v) for p, v in prov_amounts.items()}
    prov_std  = {p: max(np.std(v), 1.0) for p, v in prov_amounts.items()}

    feature_rows = []
    for clm in claims:
        prov  = clm["provider_id"]
        c_data = centrality.get(prov, {})

        # Z-score of billed amount vs provider's own historical mean
        bill_zscore = (clm["billed_amount"] - prov_mean.get(prov, 0)) / prov_std.get(prov, 1)

        row = {
            "claim_id":               clm["claim_id"],
            "is_fraud":               int(clm["is_fraud"]),
            "fraud_type":             clm.get("fraud_type", "none"),
            # Claim features
            "billed_amount":          clm["billed_amount"],
            "allowed_amount":         clm["allowed_amount"],
            "paid_amount":            clm["paid_amount"],
            "bill_allow_ratio":       round(clm["billed_amount"] / max(clm["allowed_amount"], 1), 3),
            "is_denied":              int(clm["claim_status"] == "denied"),
            "n_cpt_codes":            len(clm["cpt_codes"]),
            "bill_amount_zscore":     round(bill_zscore, 4),
            # Provider graph features
            "provider_pagerank":      c_data.get("pagerank", 0),
            "provider_betweenness":   c_data.get("betweenness", 0),
            "provider_in_degree":     c_data.get("in_degree_weighted", 0),
            "provider_out_degree":    c_data.get("out_degree_weighted", 0),
            "provider_ref_concentration": concentration.get(prov, 0),
            "provider_is_fraud_ring": int(c_data.get("is_fraud_ring", False)),
        }
        feature_rows.append(row)

    log.info("Built fraud feature set: %d rows, %d features",
             len(feature_rows), len(feature_rows[0]) if feature_rows else 0)
    return feature_rows


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-dir",   type=Path, default=Path("kg"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir",  type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def load_json(name):
        p = args.data_dir / f"{name}.json"
        return json.load(open(p)) if p.exists() else []

    patients  = load_json("patients")
    claims    = load_json("claims")

    # ── Provider network ──────────────────────────────────────
    gml_path = args.kg_dir / "provider_network.graphml"
    if not gml_path.exists():
        log.error("provider_network.graphml not found. Run kg_builder.py first.")
        return

    G = nx.read_graphml(str(gml_path))
    log.info("Provider network loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    provider_analytics = provider_network_analysis(G)
    with open(args.out_dir / "provider_analytics.json", "w") as f:
        # Convert to JSON-serialisable (community sets → lists)
        safe = {k: (list(v) if isinstance(v, set) else v)
                for k, v in provider_analytics.items()}
        json.dump(safe, f, indent=2, default=str)
    log.info("Saved provider analytics")

    # ── Patient risk stratification ───────────────────────────
    strat = patient_risk_stratification(patients, claims)
    with open(args.out_dir / "patient_risk_tiers.json", "w") as f:
        json.dump(strat, f, indent=2)
    log.info("Saved patient risk tiers")

    # ── Fraud features ────────────────────────────────────────
    fraud_features = compute_fraud_features(claims, provider_analytics)
    with open(args.out_dir / "fraud_features.json", "w") as f:
        json.dump(fraud_features, f, indent=2)
    log.info("Saved fraud features: %d rows", len(fraud_features))

    # ── Print summary ─────────────────────────────────────────
    print("\n" + "═"*60)
    print("  Healthcare KG — Graph Analytics Summary")
    print("═"*60)
    print(f"  Provider network nodes : {G.number_of_nodes()}")
    print(f"  Provider network edges : {G.number_of_edges()}")
    cs = strat["cohort_stats"]
    for tier in ["high", "medium", "low"]:
        if cs.get(tier):
            print(f"  {tier.capitalize()} risk patients: {cs[tier]['count']} "
                  f"(avg spend ${cs[tier]['avg_annual_spend']:,.0f})")
    fraud_count = sum(1 for r in fraud_features if r["is_fraud"])
    print(f"  Fraud claims in feature set: {fraud_count} / {len(fraud_features)}")

    top = provider_analytics.get("top_providers_pagerank", [])[:5]
    if top:
        print("\n  Top 5 providers by PageRank:")
        for p in top:
            ring = " ⚠️ FRAUD RING" if p["is_fraud_ring"] else ""
            print(f"    {p['name'][:30]:30s}  PR={p['pagerank']:.5f}{ring}")

    print("\n  ✅ Results saved to", args.out_dir)


if __name__ == "__main__":
    main()
