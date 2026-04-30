from __future__ import annotations

import re
from bs4 import BeautifulSoup

from .normalizer import classify_domain, dedupe_skills, infer_education, infer_years, normalize_skill


DEFAULT_SKILL_TERMS = {
    "python", "java", "javascript", "typescript", "sql", "nosql", "neo4j",
    "react", "node js", "excel", "microsoft excel", "tableau", "power bi",
    "machine learning", "deep learning", "nlp", "data mining", "pandas",
    "numpy", "scikit learn", "tensorflow", "pytorch", "aws", "azure", "gcp",
    "docker", "kubernetes", "linux", "git", "communication", "leadership",
    "customer service", "sales", "marketing", "crm", "salesforce",
    "project management", "data analysis", "statistics", "etl", "spark",
    "hadoop", "html", "css", "adobe creative cloud", "photoshop",
}


def html_to_text(html: object) -> str:
    if html is None:
        return ""
    soup = BeautifulSoup(str(html), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def extract_skills(text: object, vocabulary: set[str] | None = None) -> list[str]:
    body = f" {str(text or '').lower()} "
    terms = vocabulary or DEFAULT_SKILL_TERMS
    found = []
    for term in terms:
        norm = normalize_skill(term)
        pattern = r"(?<![a-z0-9+#.])" + re.escape(term.lower()) + r"(?![a-z0-9+#.])"
        if re.search(pattern, body):
            found.append(norm)
    return dedupe_skills(found)


def parse_job_description(
    title: str,
    company: str,
    description: str,
    location: str = "",
    vocabulary: set[str] | None = None,
) -> dict:
    combined = f"{title}\n{description}"
    domain = classify_domain(combined, job_title=title)
    return {
        "title": title.strip() or "Untitled Job",
        "company": company.strip() or "Unknown Company",
        "location": location.strip(),
        "description": description.strip(),
        "experience_years": infer_years(combined),
        "education": infer_education(combined),
        "domain": domain,
        "skills": extract_skills(combined, vocabulary),
    }


def parse_resume_text(
    candidate_id: str,
    name: str,
    email: str,
    text: str,
    category: str = "",
    vocabulary: set[str] | None = None,
) -> dict:
    domain = classify_domain(text, existing_domain=category)
    return {
        "candidate_id": str(candidate_id),
        "name": name.strip() or f"Candidate {candidate_id}",
        "email": email.strip(),
        "category": category.strip(),
        "domain": domain,
        "resume_text": text.strip(),
        "experience_years": infer_years(text),
        "education": infer_education(text),
        "skills": extract_skills(text, vocabulary),
    }
