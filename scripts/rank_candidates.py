from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ma_rank.agents import ConsensusAgent, MatcherRankerAgent
from ma_rank.config import get_neo4j_config
from ma_rank.graph import GraphStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates for a Neo4j Job node.")
    parser.add_argument("job_id", help="Job.job_id value to rank candidates against.")
    parser.add_argument("--limit", type=int, default=10, help="Number of ranked candidates to print.")
    parser.add_argument(
        "--candidate-pool-limit",
        type=int,
        default=5000,
        help="Maximum candidates to pull from Neo4j before ranking.",
    )
    parser.add_argument(
        "--include-zero-skill-matches",
        action="store_true",
        help="Include candidates with zero required skill overlap.",
    )
    parser.add_argument(
        "--consensus",
        action="store_true",
        help="Run the Consensus Agent over the ranked shortlist.",
    )
    parser.add_argument(
        "--instructions",
        default="",
        help="Recruiter must-have instructions for the Consensus Agent.",
    )
    parser.add_argument(
        "--consensus-top-n",
        type=int,
        default=5,
        help="Number of ranked candidates whose resume text is reviewed by consensus.",
    )
    args = parser.parse_args()

    with GraphStore(get_neo4j_config()) as graph:
        graph.verify()
        agent = MatcherRankerAgent(graph)
        ranked = agent.rank_for_job(
            args.job_id,
            limit=args.limit,
            candidate_pool_limit=args.candidate_pool_limit,
            include_zero_skill_matches=args.include_zero_skill_matches,
        )
        if args.consensus:
            job = graph.get_job(args.job_id) or {"job_id": args.job_id}
            consensus = ConsensusAgent(graph).review(
                job,
                ranked,
                recruiter_instructions=args.instructions,
                top_n=args.consensus_top_n,
            )
            print(json.dumps({"ranked": ranked, "consensus": consensus}, indent=2))
            return

    print(json.dumps(ranked, indent=2))


if __name__ == "__main__":
    main()
