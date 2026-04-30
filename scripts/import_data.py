from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ma_rank.agents import ExtractorAgent, GraphWriterAgent, NormalizerAgent
from ma_rank.config import get_neo4j_config
from ma_rank.extractors import html_to_text
from ma_rank.graph import GraphStore
from ma_rank.normalizer import infer_education, normalize_skill
from ma_rank.preprocessing import (
    augment_jobs,
    augment_resumes,
    build_job_domain_dataset,
    write_augmented,
    write_job_domain_dataset,
)


def load_jobs(linkedin_dir: Path, limit: int, extractor: ExtractorAgent, normalizer: NormalizerAgent) -> list[dict]:
    postings = build_job_domain_dataset(pd.read_csv(linkedin_dir / "postings.csv", nrows=limit), linkedin_dir)
    jobs = []
    for row_number, (_, row) in enumerate(postings.iterrows(), start=1):
        job_id = str(row.get("job_id", "")).strip()
        description = str(row.get("description", "") or "")
        title = str(row.get("title", "") or "").strip() or "Untitled Job"
        company = str(row.get("company_name", "") or "").strip() or "Unknown Company"
        location = str(row.get("location", "") or "").strip()
        domain = str(row.get("ma_rank_domain", "") or "").strip().lower()
        if domain not in {"sales", "technology"}:
            continue
        print(f"Extracting job {len(jobs) + 1} from row {row_number}: {title[:80]}")
        try:
            extracted = extractor.extract_job(
                title=title,
                company=company,
                description=description,
                location=location,
                domain_hint=domain,
            )
        except Exception as exc:
            print(f"  Skipping job {job_id or row_number}: extraction failed: {exc}")
            continue
        extracted["job_id"] = job_id or f"job-{row_number}"
        extracted["title"] = title
        extracted["company"] = company
        extracted["location"] = extracted.get("location") or location
        extracted["description"] = description
        extracted["domain"] = domain
        extracted["linkedin_skill_abrs"] = row.get("linkedin_skill_abrs", [])
        extracted["linkedin_skill_names"] = row.get("linkedin_skill_names", [])
        extracted["posting_date"] = extracted.get("posting_date") or str(row.get("posted_date", "") or "")
        extracted["source"] = "linkedin"
        if not extracted.get("education"):
            extracted["education"] = infer_education(description)
        if not extracted.get("experience_years"):
            row_experience = row.get("Experience")
            if pd.notna(row_experience):
                try:
                    extracted["experience_years"] = int(float(row_experience))
                except (TypeError, ValueError):
                    pass
        jobs.append(normalizer.normalize_job(extracted))
    return jobs


def load_resumes(resume_csv: Path, limit: int, vocabulary: set[str], extractor: ExtractorAgent, normalizer: NormalizerAgent) -> list[dict]:
    df = augment_resumes(pd.read_csv(resume_csv, nrows=limit))
    candidates = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        candidate_id = str(row.get("ID", "")).strip()
        text = str(row.get("Resume_str", "") or "").strip()
        if not text and row.get("Resume_html"):
            text = html_to_text(row.get("Resume_html"))
        print(f"Extracting candidate {len(candidates) + 1} from row {row_number}: {candidate_id}")
        try:
            candidate = extractor.extract_resume(
                candidate_id=candidate_id or f"candidate-{row_number}",
                name=str(row.get("Name", "") or f"Candidate {candidate_id or row_number}"),
                email=str(row.get("Email", "") or f"candidate{candidate_id or row_number}@example.com"),
                text=text,
                category=str(row.get("Category", "") or ""),
                vocabulary=vocabulary,
            )
        except Exception as exc:
            print(f"  Skipping candidate {candidate_id or row_number}: extraction failed: {exc}")
            continue
        if not candidate.get("experience_years"):
            candidate["experience_years"] = int(row.get("Experience") or 0)
        candidates.append(normalizer.normalize_candidate(candidate))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Import LinkedIn jobs and resumes into Neo4j.")
    parser.add_argument("--linkedin-dir", default="../LinkedInJobPostings (2023-2024)")
    parser.add_argument("--resume-csv", default="../ResumeDataset/Resume/Resume.csv")
    parser.add_argument("--job-limit", type=int, default=int(os.getenv("MA_RANK_JOB_LIMIT", "5000")))
    parser.add_argument("--resume-limit", type=int, default=int(os.getenv("MA_RANK_RESUME_LIMIT", "500")))
    parser.add_argument("--clear", action="store_true", help="Delete all existing graph data before import.")
    args = parser.parse_args()

    linkedin_dir = Path(args.linkedin_dir).resolve()
    resume_csv = Path(args.resume_csv).resolve()
    extractor = ExtractorAgent()
    print(f"Using LLM provider: {extractor.llm.provider}")
    normalizer = NormalizerAgent()
    jobs = load_jobs(linkedin_dir, args.job_limit, extractor, normalizer)
    vocabulary = {skill for job in jobs for skill in job["skills"]}
    vocabulary.update(normalize_skill(skill) for skill in vocabulary)
    candidates = load_resumes(resume_csv, args.resume_limit, vocabulary, extractor, normalizer)
    write_augmented(
        augment_resumes(pd.read_csv(resume_csv, nrows=args.resume_limit)),
        augment_jobs(pd.read_csv(linkedin_dir / "postings.csv", nrows=args.job_limit)),
        Path("outputs"),
    )
    write_job_domain_dataset(
        build_job_domain_dataset(pd.read_csv(linkedin_dir / "postings.csv", nrows=args.job_limit), linkedin_dir),
        Path("outputs"),
    )

    with GraphStore(get_neo4j_config()) as graph:
        graph.verify()
        graph.init_schema()
        if args.clear:
            graph.clear()
            graph.init_schema()
        writer = GraphWriterAgent(graph)
        job_count = writer.write_jobs(jobs)["jobs_written"]
        candidate_count = writer.write_candidates(candidates)["candidates_written"]
    print(f"Imported {job_count} jobs and {candidate_count} candidates.")


if __name__ == "__main__":
    main()
