import json
from pathlib import Path
from typing import List, Dict

class TransitionGraph:
    def __init__(self, path: str = "artefacts/train_graph.json"):
        if not Path(path).exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        
        with open(path, "r") as f:
            self.graph: Dict[str, Dict[str, int]] = json.load(f)


    def outgoing(self, node: str) -> Dict[str, int]:
        """Raw outgoing transition counts"""
        return self.graph.get(node, {})

    def degree(self, node: str) -> int:
        """Number of outgoing neighbours"""
        return len(self.outgoing(node))

    def total_transitions(self, node: str) -> int:
        """Total outgoing edge weight"""
        return sum(self.outgoing(node).values())


    def outgoing_probs(self, node: str) -> Dict[str, float]:
        """Normalized outgoing transition probabilities"""
        transitions = self.outgoing(node)
        total = self.total_transitions(node)

        if total == 0:
            return {}

        return {
            k: v / total
            for k, v in transitions.items()
        }

    def top_k_transitions(self, node: str, k: int = 3) -> List[dict]:
        """Top-k outgoing transitions by probability"""
        probs = self.outgoing_probs(node)

        return [
            {"node": n, "probability": p}
            for n, p in sorted(
                probs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:k]
        ]


    def node_summary(self, node: str) -> dict:
        """Full summary for API responses"""
        transitions = self.outgoing(node)

        if not transitions:
            return {
                "node": node,
                "degree": 0,
                "total_transitions": 0,
                "outgoing_transitions": {},
                "outgoing_probabilities": {},
                "top_transitions": []
            }

        return {
            "node": node,
            "degree": self.degree(node),
            "total_transitions": self.total_transitions(node),
            "outgoing_transitions": transitions,
            "outgoing_probabilities": self.outgoing_probs(node),
            "top_transitions": self.top_k_transitions(node)
        }
