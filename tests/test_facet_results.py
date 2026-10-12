"""Facet results from rcsb_search_request: sorted, kept to a readable size, and annotated.

A terms facet returns every value its hits carry and the Search API sets no default cap.
Measured 2026-10-09 (~17-19 tokens a value): source taxonomy_lineage.name over 44 hits had
441 values (~7k tokens), over 2,468 viral hits 2,257 (~40k); annotation_lineage.name over
71,518 "kinase" hits 44,495 (~908k). On nested paths the values also come back unordered.
So terms facets are sorted by population and, past search._FACET_VALUES_SHOWN values when
the spec set no max_num_intervals, keep the most common with a note saying how to choose
the size. A facet response's total_count is the hit count, so the side-count notes
(missing values, chemical-index widening) now run for breakdowns too.

Network-free: the Search API is replaced by a stub.
"""

import asyncio

import pytest

from rcsb_mcp import query_doc, search

SHOWN = search._FACET_VALUES_SHOWN
LINEAGE = "rcsb_entity_source_organism.taxonomy_lineage.name"


def terms(name, attribute, **extra):
    return {"name": name, "aggregation_type": "terms", "attribute": attribute, **extra}


def body_with(*specs):
    return {"query": {"type": "terminal", "service": "full_text", "parameters": {"value": "x"}},
            "return_type": "entry", "request_options": {"facets": list(specs)}}


def raw_with(name, populations, sub=None):
    buckets = []
    for i, p in enumerate(populations):
        bucket = {"label": f"v{i}", "population": p}
        if sub:
            bucket["facets"] = [sub]
        buckets.append(bucket)
    return {"name": name, "buckets": buckets}


def shape(raw_facets, *specs):
    return search._format_facets({"total_count": 50, "facets": raw_facets}, body_with(*specs))


def test_terms_values_come_back_most_common_first():
    """Measured: nested paths (annotation_lineage, rcsb_ec_lineage, *_feature_summary)
    return their values in no particular order."""
    out = shape([raw_with("go", [3, 40, 1, 12])], terms("go", "rcsb_polymer_entity_annotation.annotation_lineage.name"))
    assert [b["population"] for b in out["facets"][0]["buckets"]] == [40, 12, 3, 1]
    assert "notes" not in out


def test_histogram_and_range_buckets_keep_their_bin_order():
    spec = {"name": "h", "aggregation_type": "histogram", "attribute": "rcsb_entry_info.resolution_combined",
            "interval": 0.5}
    out = shape([raw_with("h", [5, 90, 30])], spec)
    assert [b["population"] for b in out["facets"][0]["buckets"]] == [5, 90, 30]


def test_a_facet_with_more_values_than_shown_keeps_the_most_common_and_says_so():
    populations = list(range(1, 251))  # 250 values, least common first
    out = shape([raw_with("lineage", populations)], terms("lineage", LINEAGE))
    shown = [b["population"] for b in out["facets"][0]["buckets"]]
    assert shown == list(range(250, 250 - SHOWN, -1))
    note = out["notes"][0]
    assert "has 250 values" in note and f"the {SHOWN} most common" in note
    assert "max_num_intervals" in note and "min_interval_population" in note


def test_a_size_the_caller_asked_for_is_left_alone():
    out = shape([raw_with("org", [1] * 300)], terms("org", "rcsb_entity_source_organism.ncbi_scientific_name",
                                                    max_num_intervals=500))
    assert len(out["facets"][0]["buckets"]) == 300 and "notes" not in out


def test_a_population_floor_alone_does_not_bound_the_size():
    """min_interval_population drops rare values but still left 1,277 viral lineage names."""
    out = shape([raw_with("lineage", [2] * 150)], terms("lineage", LINEAGE, min_interval_population=2))
    assert len(out["facets"][0]["buckets"]) == SHOWN and out["notes"]


def test_sub_facets_are_shaped_too_with_one_note_per_facet():
    sub = raw_with("org", list(range(1, 151)))
    spec = terms("method", "exptl.method", facets=[terms("org", "rcsb_entity_source_organism.ncbi_scientific_name")])
    out = shape([raw_with("method", [10, 20], sub=sub)], spec)
    outer = out["facets"][0]["buckets"]
    assert [b["population"] for b in outer] == [20, 10]
    for bucket in outer:
        inner = [b["population"] for b in bucket["facets"][0]["buckets"]]
        assert len(inner) == SHOWN and inner[0] == 150
    assert len(out["notes"]) == 1 and "Facet 'org'" in out["notes"][0]


def test_a_metric_facet_passes_through():
    out = shape([{"name": "n", "value": 42}], {"name": "n", "aggregation_type": "cardinality", "attribute": "x"})
    assert out["facets"] == [{"name": "n", "value": 42}]


# --------------------------------------------------------------------------- #
# wired into rcsb_search_request
# --------------------------------------------------------------------------- #
PH = "exptl_crystal_grow.pH"


def term(attribute, operator="exact_match", value="x"):
    params = {"attribute": attribute, "operator": operator}
    if operator != "exists":
        params["value"] = value
    return {"type": "terminal", "service": "text", "parameters": params}


@pytest.fixture
def api(monkeypatch):
    """Search API stub: facet requests get a 250-value breakdown, counts get 669."""
    sent = []

    async def fake_post(body):
        sent.append(body)
        if body["request_options"].get("return_counts"):
            return {"total_count": 669}
        if body["request_options"].get("facets"):
            return {"total_count": 2, "facets": [raw_with("org", list(range(1, 251)))]}
        return {"total_count": 2, "group_set": [], "result_set": []}

    monkeypatch.setattr(search, "_post_search", fake_post)
    return sent


def run(node, **kw):
    return asyncio.run(search.rcsb_search_request(query_doc.sign(node), return_type="entry", **kw))


def test_a_breakdown_gets_the_cap_note_and_the_missing_value_note(api):
    """Measured: Viruses AND pH 7-8 AND resolution < 2 with an exptl.method facet had
    total_count 1,546, the same as the hit-returning request, and no note before."""
    node = {"type": "group", "logical_operator": "and",
            "nodes": [term(PH, "range", {"from": 7.0, "to": 8.0}), term("rcsb_entry_info.resolution_combined", "less", 2.0)]}
    r = run(node, facets=[terms("org", "rcsb_entity_source_organism.ncbi_scientific_name")])
    assert len(r["facets"][0]["buckets"]) == SHOWN
    assert any("has 250 values" in n for n in r["notes"])
    assert any(PH in n and "669" in n for n in r["notes"])
    assert len(api) == 2, "the breakdown plus one count"


def test_a_grouped_search_still_skips_the_side_counts(api):
    """Its total counts groups, not the hits the notes talk about."""
    node = {"type": "group", "logical_operator": "and",
            "nodes": [term(PH, "range", {"from": 7.0, "to": 8.0}), term("rcsb_entry_info.resolution_combined", "less", 2.0)]}
    asyncio.run(search.rcsb_search_request(query_doc.sign(node), return_type="polymer_entity", group_by="uniprot"))
    assert len(api) == 1
