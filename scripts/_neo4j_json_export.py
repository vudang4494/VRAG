#!/usr/bin/env python3
"""Neo4j JSON export helper — parses cypher-shell node output to JSON lines.

Usage:
  cat cypher_output.txt | python3 _neo4j_json_export.py <label>
  docker exec rag-neo4j cypher-shell ... | python3 _neo4j_json_export.py <label>
"""
import json
import sys


def parse_cypher_node(line: str):
    """Parse a single node line from cypher-shell WITH n RETURN n output.

    Format variations from cypher-shell:
      - Node{prop: value, ...}
      - :<Label> {prop: value, ...}
      - {"prop": value, ...}

    Returns dict or None.
    """
    line = line.strip()
    if not line or line.startswith("+") or line.startswith("|") or line.startswith("n"):
        return None
    if "Node" in line:
        # node: <id> {properties}
        if "{" in line:
            props_str = line[line.index("{"):]
            if props_str.endswith("}"):
                try:
                    # Handle Cypher node representation
                    # Format: Node<123> {prop: value, ...}
                    # Remove Node<id> prefix
                    bracket_end = props_str.rindex("}")
                    inner = props_str[:bracket_end + 1]
                    return json.loads(inner)
                except (json.JSONDecodeError, ValueError):
                    pass
    return None


def main():
    sys.argv[1]  # label argument reserved for future use

    results = []
    for line in sys.stdin:
        node = parse_cypher_node(line)
        if node is not None:
            results.append(node)

    for r in results:
        print(json.dumps(r, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
