#!/usr/bin/env python3
"""Seed Neo4j with sample authors, papers, venues, keywords, and citations."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")

CYPHERS = """
CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT inst_id IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE;
CREATE CONSTRAINT kw_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.id IS UNIQUE;
CREATE CONSTRAINT venue_id IF NOT EXISTS FOR (v:Venue) REQUIRE v.id IS UNIQUE;
MERGE (:Institution {id:'i_mit', name:'MIT'});
MERGE (:Institution {id:'i_stanford', name:'Stanford University'});
MERGE (:Institution {id:'i_dm', name:'DeepMind'});
MERGE (:Author {id:'a1', name:'Geoffrey Hinton'});
MERGE (:Author {id:'a2', name:'Yoshua Bengio'});
MERGE (:Author {id:'a3', name:'David Silver'});
MERGE (:Author {id:'a4', name:'Jure Leskovec'});
MERGE (:Author {id:'a5', name:'Karen Simonyan'});
MATCH (a:Author {id:'a1'}), (i:Institution {id:'i_dm'})
MERGE (a)-[:AFFILIATED_WITH]->(i);
MATCH (a:Author {id:'a2'}), (i:Institution {id:'i_mit'})
MERGE (a)-[:AFFILIATED_WITH]->(i);
MATCH (a:Author {id:'a3'}), (i:Institution {id:'i_dm'})
MERGE (a)-[:AFFILIATED_WITH]->(i);
MATCH (a:Author {id:'a4'}), (i:Institution {id:'i_stanford'})
MERGE (a)-[:AFFILIATED_WITH]->(i);
MATCH (a:Author {id:'a5'}), (i:Institution {id:'i_dm'})
MERGE (a)-[:AFFILIATED_WITH]->(i);
MERGE (:Venue {id:'v_nips', name:'NeurIPS'});
MERGE (:Venue {id:'v_iclr', name:'ICLR'});
MERGE (:Venue {id:'v_icml', name:'ICML'});
MERGE (:Keyword {id:'k_tf', name:'transformers'});
MERGE (:Keyword {id:'k_rl', name:'reinforcement learning'});
MERGE (:Keyword {id:'k_gnn', name:'graph neural networks'});
MERGE (:Keyword {id:'k_diff', name:'diffusion models'});
MERGE (:Keyword {id:'k_llm', name:'large language models'});
MERGE (:Paper {id:'p1', title:'Attention Is All You Need', year:2017});
MERGE (:Paper {id:'p2', title:'Deep Residual Learning for Image Recognition', year:2016});
MERGE (:Paper {id:'p3', title:'Mastering the Game of Go with Deep RL', year:2016});
MERGE (:Paper {id:'p4', title:'Graph Attention Networks', year:2018});
MERGE (:Paper {id:'p5', title:'BERT: Pre-training of Deep Bidirectional Transformers', year:2019});
MERGE (:Paper {id:'p6', title:'Denoising Diffusion Probabilistic Models', year:2020});
MERGE (:Paper {id:'p7', title:'Neural ODEs', year:2018});
MERGE (:Paper {id:'p8', title:'Scaling Laws for Neural Language Models', year:2020});
MERGE (:Paper {id:'p9', title:'Inductive Representation Learning on Large Graphs', year:2017});
MERGE (:Paper {id:'p10', title:'Very Deep Convolutional Networks for Large-Scale Image Recognition', year:2015});
MATCH (a:Author {id:'a1'}), (p:Paper {id:'p1'}) MERGE (a)-[:WROTE]->(p);
MATCH (a:Author {id:'a2'}), (p:Paper {id:'p5'}) MERGE (a)-[:WROTE]->(p);
MATCH (a:Author {id:'a3'}), (p:Paper {id:'p3'}) MERGE (a)-[:WROTE]->(p);
MATCH (a:Author {id:'a4'}), (p:Paper {id:'p4'}) MERGE (a)-[:WROTE]->(p);
MATCH (a:Author {id:'a4'}), (p:Paper {id:'p9'}) MERGE (a)-[:WROTE]->(p);
MATCH (a:Author {id:'a5'}), (p:Paper {id:'p2'}) MERGE (a)-[:WROTE]->(p);
MATCH (p:Paper {id:'p1'}), (v:Venue {id:'v_nips'}) MERGE (p)-[:PUBLISHED_IN]->(v);
MATCH (p:Paper {id:'p2'}), (v:Venue {id:'v_iclr'}) MERGE (p)-[:PUBLISHED_IN]->(v);
MATCH (p:Paper {id:'p3'}), (v:Venue {id:'v_nips'}) MERGE (p)-[:PUBLISHED_IN]->(v);
MATCH (p:Paper {id:'p4'}), (v:Venue {id:'v_iclr'}) MERGE (p)-[:PUBLISHED_IN]->(v);
MATCH (p:Paper {id:'p5'}), (v:Venue {id:'v_nips'}) MERGE (p)-[:PUBLISHED_IN]->(v);
MATCH (p:Paper {id:'p6'}), (v:Venue {id:'v_icml'}) MERGE (p)-[:PUBLISHED_IN]->(v);
MATCH (p:Paper {id:'p7'}), (v:Venue {id:'v_iclr'}) MERGE (p)-[:PUBLISHED_IN]->(v);
MATCH (p:Paper {id:'p8'}), (v:Venue {id:'v_icml'}) MERGE (p)-[:PUBLISHED_IN]->(v);
MATCH (p:Paper {id:'p9'}), (v:Venue {id:'v_nips'}) MERGE (p)-[:PUBLISHED_IN]->(v);
MATCH (p:Paper {id:'p10'}), (v:Venue {id:'v_iclr'}) MERGE (p)-[:PUBLISHED_IN]->(v);
MATCH (p:Paper {id:'p1'}), (k:Keyword {id:'k_tf'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (p:Paper {id:'p5'}), (k:Keyword {id:'k_tf'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (p:Paper {id:'p5'}), (k:Keyword {id:'k_llm'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (p:Paper {id:'p8'}), (k:Keyword {id:'k_llm'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (p:Paper {id:'p3'}), (k:Keyword {id:'k_rl'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (p:Paper {id:'p4'}), (k:Keyword {id:'k_gnn'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (p:Paper {id:'p9'}), (k:Keyword {id:'k_gnn'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (p:Paper {id:'p6'}), (k:Keyword {id:'k_diff'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (p:Paper {id:'p2'}), (k:Keyword {id:'k_llm'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (p:Paper {id:'p7'}), (k:Keyword {id:'k_tf'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (p:Paper {id:'p10'}), (k:Keyword {id:'k_llm'}) MERGE (p)-[:HAS_TOPIC]->(k);
MATCH (c:Paper {id:'p5'}), (p:Paper {id:'p1'}) MERGE (c)-[:CITES]->(p);
MATCH (c:Paper {id:'p6'}), (p:Paper {id:'p2'}) MERGE (c)-[:CITES]->(p);
MATCH (c:Paper {id:'p4'}), (p:Paper {id:'p9'}) MERGE (c)-[:CITES]->(p);
MATCH (c:Paper {id:'p8'}), (p:Paper {id:'p5'}) MERGE (c)-[:CITES]->(p);
MATCH (c:Paper {id:'p7'}), (p:Paper {id:'p1'}) MERGE (c)-[:CITES]->(p);
"""


def main() -> None:
    """Run constraint and sample-data statements."""
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        for stmt in CYPHERS.strip().split(";"):
            q = stmt.strip()
            if q:
                session.run(q)
    driver.close()
    print("Seeded Neo4j with sample graph.")


if __name__ == "__main__":
    main()
