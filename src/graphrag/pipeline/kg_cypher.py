"""Cypher fragments for KGTraversal (keeps traversal module under line limits)."""

from __future__ import annotations

YEAR_P = (
    " AND ($min_year IS NULL OR p.year >= $min_year)"
    " AND ($max_year IS NULL OR p.year <= $max_year)"
)

AUTHOR_PAPERS = (
    "MATCH (a:Author)-[:WROTE]->(p:Paper) WHERE toLower(a.name) CONTAINS toLower($name)"
    + YEAR_P
    + " OPTIONAL MATCH (p)-[:HAS_TOPIC]->(k:Keyword)"
    " OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)"
    " OPTIONAL MATCH (p)<-[:WROTE]-(co:Author)"
    " RETURN p.id AS id, p.title AS title, p.year AS year,"
    " collect(DISTINCT k.name) AS topics, v.name AS venue,"
    " collect(DISTINCT co.name) AS authors ORDER BY p.year DESC LIMIT 20"
)

TOPIC_PAPERS = (
    "MATCH (p:Paper)-[:HAS_TOPIC]->(k:Keyword) WHERE toLower(k.name) CONTAINS toLower($keyword)"
    + YEAR_P
    + " OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)"
    " OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)"
    " OPTIONAL MATCH (p)-[:HAS_TOPIC]->(k2:Keyword)"
    " RETURN p.id AS id, p.title AS title, p.year AS year,"
    " collect(DISTINCT a.name) AS authors, v.name AS venue,"
    " collect(DISTINCT k2.name) AS all_topics ORDER BY p.year DESC LIMIT 20"
)

CITE_PAPERS = (
    "MATCH (p:Paper)-[:CITES]->(cited:Paper) WHERE toLower(cited.title) CONTAINS toLower($title)"
    + YEAR_P
    + " OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)"
    " OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)"
    " RETURN p.id AS id, p.title AS title, p.year AS year,"
    " collect(DISTINCT a.name) AS authors, v.name AS venue, cited.title AS cites"
    " ORDER BY p.year DESC LIMIT 20"
)

VENUE_PAPERS = (
    "MATCH (p:Paper)-[:PUBLISHED_IN]->(v:Venue) WHERE toLower(v.name) CONTAINS toLower($venue_name)"
    + YEAR_P
    + " OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)"
    " OPTIONAL MATCH (p)-[:HAS_TOPIC]->(k:Keyword)"
    " RETURN p.id AS id, p.title AS title, p.year AS year,"
    " collect(DISTINCT a.name) AS authors, collect(DISTINCT k.name) AS topics, v.name AS venue"
    " ORDER BY p.year DESC LIMIT 20"
)

INST_PAPERS = (
    "MATCH (a:Author)-[:AFFILIATED_WITH]->(i:Institution) "
    "WHERE toLower(i.name) CONTAINS toLower($institution) MATCH (a)-[:WROTE]->(p:Paper) "
    "AND ($min_year IS NULL OR p.year >= $min_year) AND ($max_year IS NULL OR p.year <= $max_year) "
    "OPTIONAL MATCH (p)-[:HAS_TOPIC]->(k:Keyword) "
    "RETURN p.id AS id, p.title AS title, p.year AS year, "
    "collect(DISTINCT a.name) AS authors, collect(DISTINCT k.name) AS topics "
    "ORDER BY p.year DESC LIMIT 20"
)

FIND_AUTHORS = (
    "MATCH (a:Author)-[:WROTE]->(p:Paper)-[:HAS_TOPIC]->(k:Keyword) "
    "WHERE toLower(k.name) CONTAINS toLower($keyword) "
    "OPTIONAL MATCH (a)-[:AFFILIATED_WITH]->(i:Institution) "
    "RETURN a.id AS id, a.name AS name, collect(DISTINCT p.title) AS papers, "
    "i.name AS institution, count(p) AS paper_count ORDER BY paper_count DESC LIMIT 10"
)

EXPLORE = (
    "MATCH (n) WHERE (n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($kw)) OR "
    "(n.title IS NOT NULL AND toLower(n.title) CONTAINS toLower($kw)) RETURN n LIMIT 15"
)
