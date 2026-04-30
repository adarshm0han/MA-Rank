from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ma_rank.preprocessing import (
    augment_jobs,
    augment_resumes,
    build_job_domain_dataset,
    write_augmented,
    write_job_domain_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MA-Rank augmented recruiter-side datasets.")
    parser.add_argument("--linkedin-dir", default="../LinkedInJobPostings (2023-2024)")
    parser.add_argument("--resume-csv", default="../ResumeDataset/Resume/Resume.csv")
    parser.add_argument("--job-limit", type=int, default=5000)
    parser.add_argument("--resume-limit", type=int, default=500)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    linkedin_dir = Path(args.linkedin_dir).resolve()
    resume_csv = Path(args.resume_csv).resolve()
    out_dir = Path(args.outdir).resolve()

    resumes = pd.read_csv(resume_csv, nrows=args.resume_limit)
    jobs = pd.read_csv(linkedin_dir / "postings.csv", nrows=args.job_limit)

    resumes_augmented = augment_resumes(resumes)
    jobs_augmented = augment_jobs(jobs)
    jobs_sales_technology = build_job_domain_dataset(jobs, linkedin_dir)
    write_augmented(resumes_augmented, jobs_augmented, out_dir)
    write_job_domain_dataset(jobs_sales_technology, out_dir)

    print(f"Wrote {len(resumes_augmented)} resumes and {len(jobs_augmented)} jobs to {out_dir}")
    print(f"Resumes: {out_dir / 'resumes_augmented.csv'}")
    print(f"Jobs: {out_dir / 'jobs_augmented.csv'}")
    print(f"Sales/technology jobs: {out_dir / 'jobs_sales_technology.csv'} ({len(jobs_sales_technology)} rows)")


if __name__ == "__main__":
    main()
