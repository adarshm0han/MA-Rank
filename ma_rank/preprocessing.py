from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .normalizer import classify_domain, infer_years


SALES_JOB_SKILL_ABRS = {"SALE", "BD"}
TECH_JOB_SKILL_ABRS = {"IT"}
TECH_CONTEXT_JOB_SKILL_ABRS = {"ENG", "QA"}


def synthetic_names_emails(n: int) -> tuple[list[str], list[str]]:
    try:
        from faker import Faker
        fake = Faker()
        names = [fake.name() for _ in range(n)]
        emails = [f"{_slug(name)}@{fake.free_email_domain()}" for name in names]
        return names, emails
    except Exception:
        firsts = ["Michael", "David", "James", "Mary", "Jennifer", "Alex", "Taylor", "Jordan", "Sara", "Daniel"]
        lasts = ["Smith", "Johnson", "Williams", "Brown", "Davis", "Miller", "Wilson", "Garcia", "Martinez", "Robinson"]
        names, emails = [], []
        for _ in range(n):
            name = f"{random.choice(firsts)} {random.choice(lasts)}"
            names.append(name)
            emails.append(f"{_slug(name)}{random.randint(1, 99)}@example.com")
        return names, emails


def augment_resumes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_col = _find_col(df, ["Resume_str", "Resume", "resume_text", "text", "Resume_html"])
    if "Experience" not in df.columns:
        df["Experience"] = None
    if text_col:
        inferred = df[text_col].fillna("").astype(str).apply(infer_years)
        df["Experience"] = df["Experience"].where(pd.notna(df["Experience"]), inferred)
        median = _median_or_none(df["Experience"])
        df["Experience"] = df.apply(
            lambda row: _resume_experience(row.get(text_col, ""), median)
            if _missing(row.get("Experience")) else row.get("Experience"),
            axis=1,
        )
    if "Name" not in df.columns:
        df["Name"] = None
    if "Email" not in df.columns:
        df["Email"] = None
    missing_names = df["Name"].isna() | (df["Name"].astype(str).str.strip() == "")
    if missing_names.any():
        names, emails = synthetic_names_emails(int(missing_names.sum()))
        df.loc[missing_names, "Name"] = names
        df.loc[missing_names, "Email"] = emails
    missing_emails = df["Email"].isna() | (df["Email"].astype(str).str.strip() == "")
    if missing_emails.any():
        df.loc[missing_emails, "Email"] = df.loc[missing_emails, "Name"].fillna("").apply(lambda name: f"{_slug(name)}@example.com")
    return _sanitize_frame(df)


def augment_jobs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "posted_date" not in df.columns:
        today = pd.Timestamp.now().normalize()
        rand_days = np.random.randint(0, 365, size=len(df))
        df["posted_date"] = [(today - pd.Timedelta(int(days), "D")).date().isoformat() for days in rand_days]
    title_col = _find_col(df, ["title", "job_title", "jobtitle", "position"])
    desc_col = _find_col(df, ["description", "job_description", "details", "requirements"])
    company_col = _find_col(df, ["company_name", "company", "employer"])
    location_col = _find_col(df, ["location", "city", "job_location", "work_location"])
    if "job_id" not in df.columns:
        df.insert(0, "job_id", range(1, len(df) + 1))
    if "company_name" not in df.columns:
        df["company_name"] = df[company_col] if company_col else ""
    if "title" not in df.columns:
        df["title"] = df[title_col] if title_col else ""
    if "description" not in df.columns:
        df["description"] = df[desc_col] if desc_col else ""
    if "location" not in df.columns:
        df["location"] = df[location_col] if location_col else ""
    combined = df["description"].fillna("").astype(str) + " " + df["title"].fillna("").astype(str)
    if "Experience" not in df.columns:
        df["Experience"] = combined.apply(infer_years)
    median = _median_or_none(df["Experience"])
    df["Experience"] = df.apply(
        lambda row: _job_experience(row.get("title", ""), median)
        if _missing(row.get("Experience")) or int(row.get("Experience") or 0) == 0 else row.get("Experience"),
        axis=1,
    )
    df["company_name"] = df["company_name"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    df = df.dropna(subset=["company_name"]).reset_index(drop=True)
    return _sanitize_frame(df)


def build_job_domain_dataset(postings: pd.DataFrame, linkedin_dir: Path) -> pd.DataFrame:
    """Create MA-Rank's sales/technology job dataset from postings + job_skills."""
    jobs = augment_jobs(postings)
    abrs_by_job, names_by_job = _load_job_skill_tags(linkedin_dir)
    jobs["linkedin_skill_abrs"] = jobs["job_id"].apply(lambda job_id: abrs_by_job.get(_job_id_key(job_id), []))
    jobs["linkedin_skill_names"] = jobs["job_id"].apply(lambda job_id: names_by_job.get(_job_id_key(job_id), []))
    jobs["ma_rank_domain"] = jobs.apply(
        lambda row: classify_job_domain_from_skill_tags(
            row.get("linkedin_skill_abrs", []),
            row.get("title", ""),
            row.get("description", ""),
        ),
        axis=1,
    )
    return jobs[jobs["ma_rank_domain"].isin(["sales", "technology"])].reset_index(drop=True)


def classify_job_domain_from_skill_tags(skill_abrs: object, title: object = "", description: object = "") -> str:
    """Classify a job using LinkedIn job_skills.csv tags as the primary source."""
    if isinstance(skill_abrs, str):
        abrs = {part.strip().upper() for part in re.split(r"[,;|]", skill_abrs) if part.strip()}
    elif isinstance(skill_abrs, list):
        abrs = {str(part).strip().upper() for part in skill_abrs if str(part).strip()}
    else:
        abrs = set()

    inferred = classify_domain(description, job_title=title)
    strong_tech_context = _has_strong_technology_context(title, description)
    has_sales = bool(abrs & SALES_JOB_SKILL_ABRS)
    has_tech = bool(abrs & TECH_JOB_SKILL_ABRS) and strong_tech_context
    has_context_tech = bool(abrs & TECH_CONTEXT_JOB_SKILL_ABRS) and strong_tech_context

    if has_sales and (has_tech or has_context_tech):
        title_domain = classify_domain("", job_title=title)
        if title_domain in {"sales", "technology"}:
            return title_domain
        return inferred if inferred in {"sales", "technology"} else "technology"
    if has_tech or has_context_tech:
        return "technology"
    if has_sales:
        return "sales"
    return ""


def _has_strong_technology_context(title: object = "", description: object = "") -> bool:
    text = f"{title or ''} {description or ''}".lower()
    patterns = [
        r"\bsoftware\b", r"\bdeveloper\b", r"\bprogrammer\b", r"\bfrontend\b", r"\bbackend\b",
        r"\bfull stack\b", r"\bweb developer\b", r"\bmobile developer\b",
        r"\bdata scientist\b", r"\bdata engineer\b", r"\bmachine learning\b", r"\bml engineer\b",
        r"\bdevops\b", r"\bsre\b", r"\bcloud\b", r"\baws\b", r"\bazure\b", r"\bgcp\b",
        r"\bnetwork engineer\b", r"\bsystems administrator\b", r"\bsystem administrator\b",
        r"\bdatabase\b", r"\bsql\b", r"\bpython\b", r"\bjava\b", r"\bjavascript\b",
        r"\breact\b", r"\bnode\.?js\b", r"\bdocker\b", r"\bkubernetes\b",
        r"\bcybersecurity\b", r"\binformation security\b", r"\bit support\b",
        r"\binformation technology\b", r"\bhelp desk\b", r"\bqa engineer\b",
        r"\btest automation\b", r"\bsalesforce developer\b", r"\bsalesforce architect\b",
        r"\bapex\b", r"\blightning\b", r"\bsoql\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def write_augmented(resumes: pd.DataFrame, jobs: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    resumes.to_csv(out_dir / "resumes_augmented.csv", index=False)
    jobs.to_csv(out_dir / "jobs_augmented.csv", index=False)
    try:
        resumes.to_excel(out_dir / "resumes_augmented.xlsx", index=False)
        jobs.to_excel(out_dir / "jobs_augmented.xlsx", index=False)
    except Exception:
        pass


def write_job_domain_dataset(jobs: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = jobs.copy()
    for col in ["linkedin_skill_abrs", "linkedin_skill_names"]:
        if col in output.columns:
            output[col] = output[col].apply(lambda value: "|".join(value) if isinstance(value, list) else value)
    output.to_csv(out_dir / "jobs_sales_technology.csv", index=False)
    try:
        output.to_excel(out_dir / "jobs_sales_technology.xlsx", index=False)
    except Exception:
        pass


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    columns = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    return None


def _missing(value) -> bool:
    return value is None or pd.isna(value) or str(value).strip() in {"", "None", "nan"}


def _median_or_none(values) -> int | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return int(numeric.median()) if not numeric.empty else None


def _resume_experience(text: object, median: int | None) -> int:
    lowered = str(text or "").lower()
    if "intern" in lowered:
        return 0
    if any(term in lowered for term in ["jr", "junior", "assistant", "entry", "associate"]):
        return 1
    if any(term in lowered for term in ["senior", " sr ", "lead"]):
        return 6
    if any(term in lowered for term in ["manager", "director"]):
        return 4
    return int(median) if median is not None else 2


def _job_experience(title: object, median: int | None) -> int:
    lowered = str(title or "").lower()
    if "intern" in lowered:
        return 0
    if any(term in lowered for term in ["jr", "junior", "assistant", "associate", "entry"]):
        return 1
    if "senior" in lowered or " sr " in lowered:
        return 6
    if any(term in lowered for term in ["lead", "manager", "director"]):
        return 4
    return int(median) if median is not None else 2


def _sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    def clean(value):
        if value is None:
            return ""
        return re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", " ", str(value))
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(clean)
    return df


def _slug(value: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", ".", str(value or "candidate").lower()).strip(".")
    return slug or "candidate"


def _load_job_skill_tags(linkedin_dir: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    skill_path = linkedin_dir / "jobs" / "job_skills.csv"
    if not skill_path.exists():
        return {}, {}
    skill_map = _load_skill_mapping(linkedin_dir)
    rows = pd.read_csv(skill_path)
    rows["job_id_key"] = rows["job_id"].apply(_job_id_key)
    rows["skill_abr"] = rows["skill_abr"].fillna("").astype(str).str.strip().str.upper()
    rows = rows[rows["skill_abr"] != ""]
    rows["skill_name"] = rows["skill_abr"].map(skill_map).fillna(rows["skill_abr"])
    abrs = rows.groupby("job_id_key")["skill_abr"].apply(lambda values: sorted(set(values))).to_dict()
    names = rows.groupby("job_id_key")["skill_name"].apply(lambda values: sorted(set(values))).to_dict()
    return abrs, names


def _load_skill_mapping(linkedin_dir: Path) -> dict[str, str]:
    path = linkedin_dir / "mappings" / "skills.csv"
    if not path.exists():
        return {}
    rows = pd.read_csv(path)
    return dict(zip(rows["skill_abr"].astype(str).str.upper(), rows["skill_name"].astype(str)))


def _job_id_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text
