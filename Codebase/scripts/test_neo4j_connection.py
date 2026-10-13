#!/usr/bin/env python3
"""Quick Neo4j connectivity test using project .env values."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")


def main() -> int:
    """Connect, verify, and run a tiny query."""
    print(f"Testing Neo4j connection to {URI} as {USER}...")
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        with driver.session() as session:
            record = session.run("RETURN 1 AS ok").single()
        driver.close()
        print("Neo4j connection successful.")
        print(f"Sanity query result: ok={record['ok'] if record else None}")
        return 0
    except Exception as exc:
        print(f"Neo4j connection failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
