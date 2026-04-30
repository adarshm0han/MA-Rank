from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .normalizer import normalize_skill


@dataclass
class RankedCandidate:
    candidate_id: str
    name: str
    email: str
    score: float
    skill_score: float
    skill_match_ratio: float
    jaccard_score: float
    weighted_skill_score: float
    semantic_score: float
    guttman_score: float
    experience_score: float
    education_score: float
    required_experience_years: int
    candidate_experience_years: int
    required_education: str
    candidate_education: str
    required_skills: list[str]
    candidate_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    score_breakdown: dict[str, float]
    explanation: str
    uncertainty: str


EDU_RANK = {"": 0, "bachelor's": 1, "master's": 2, "phd": 3}
EXACT_ONLY_SKILLS = {
    "c", "r", "go", "qa", "ui", "ux", "it", "hr", "bi", "ml", "ai", "js",
    "c#", "c++", "sql", "aws", "gcp", "crm", "erp", "api", "css", "html",
}

SKILL_LEVELS = {
    "technology": {
        "Level_1": ["c", "sql", "html", "css", "git", "bash"],
        "Level_2": ["python", "javascript", "react", "node.js", "java", "typescript"],
        "Level_3": ["django", "flask", "postgresql", "nosql", "mongodb", "mysql"],
        "Level_4": ["microservices", "kafka", "aws", "docker", "redis", "elasticsearch", "rest api", "api"],
        "Level_5": ["kubernetes", "distributed systems", "cloud architecture", "system design", "scalability"],
    },
    "sales": {
        "Level_1": ["crm", "lead generation", "email marketing"],
        "Level_2": ["salesforce", "hubspot", "customer outreach", "cold calling", "prospecting"],
        "Level_3": ["forecasting", "territory management", "pipeline management", "social media marketing"],
        "Level_4": ["pipeline strategy", "account management", "deal closing", "marketing automation"],
        "Level_5": ["enterprise sales", "business strategy", "strategic partnerships", "b2b sales"],
    },
}


def _clean_skill(value: Any) -> str:
    return " ".join(normalize_skill(value).lower().strip().split())


def _ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        skill = _clean_skill(value)
        if skill and skill not in seen:
            seen.add(skill)
            unique.append(skill)
    return unique


def _skill_name(skill: Any) -> str:
    if isinstance(skill, dict):
        return _clean_skill(skill.get("name") or skill.get("norm_name") or "")
    return _clean_skill(skill)


def _skill_aliases(skill: Any) -> list[str]:
    if not isinstance(skill, dict):
        return []
    aliases = skill.get("aliases") or []
    if not isinstance(aliases, list):
        return []
    return [_clean_skill(alias) for alias in aliases if _clean_skill(alias)]


def _skill_popularity(skill: Any) -> int:
    if not isinstance(skill, dict):
        return 0
    try:
        return max(int(skill.get("popularity") or 0), 0)
    except (TypeError, ValueError):
        return 0


def _token_overlap(left: str, right: str) -> bool:
    left_tokens = {token for token in left.split() if len(token) > 2}
    right_tokens = {token for token in right.split() if len(token) > 2}
    return bool(left_tokens & right_tokens)


def _allows_fuzzy_match(required: str, candidate: str) -> bool:
    if required in EXACT_ONLY_SKILLS or candidate in EXACT_ONLY_SKILLS:
        return False
    if len(required) < 4 or len(candidate) < 4:
        return False
    return True


def _match_weight(required: str, candidate_skill: Any) -> float:
    candidate = _skill_name(candidate_skill)
    if not required or not candidate:
        return 0.0
    aliases = _skill_aliases(candidate_skill)
    if candidate == required:
        base = 3.0
    elif required in aliases:
        base = 2.5
    elif _allows_fuzzy_match(required, candidate) and (required in candidate or candidate in required):
        base = 1.5
    elif _allows_fuzzy_match(required, candidate) and _token_overlap(required, candidate):
        base = 1.0
    else:
        return 0.0
    return base * (1.0 / (1.0 + log(1.0 + float(_skill_popularity(candidate_skill)))))


def _match_skills(required_skills: list[str], candidate_skill_details: list[Any]) -> tuple[list[str], list[str], float]:
    matched = []
    weighted_score = 0.0
    for required in required_skills:
        best_weight = 0.0
        for candidate_skill in candidate_skill_details:
            best_weight = max(best_weight, _match_weight(required, candidate_skill))
        if best_weight > 0:
            matched.append(required)
            weighted_score += best_weight
    missing = [skill for skill in required_skills if skill not in set(matched)]
    return matched, missing, weighted_score


def _coverage(matched: list[str], required: list[str]) -> float:
    required_set = set(required)
    if not required_set:
        return 0.0
    return len(set(matched) & required_set) / len(required_set)


def _jaccard(matched: list[str], required: list[str], candidate: list[str]) -> float:
    required_set = set(required)
    candidate_set = set(candidate)
    union = required_set | candidate_set
    if not union:
        return 0.0
    return len(set(matched)) / len(union)


def _semantic_scores(job_text: str, candidate_texts: list[str]) -> list[float]:
    if not candidate_texts:
        return []
    corpus = [job_text] + candidate_texts
    try:
        matrix = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(corpus)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return [float(score) for score in scores]
    except ValueError:
        return [0.0 for _ in candidate_texts]


def _experience_score(candidate_years: int, required_years: int) -> float:
    if required_years <= 0:
        return 1.0
    return min(max(candidate_years, 0) / required_years, 1.0)


def _education_score(candidate_education: str, required_education: str) -> float:
    required = EDU_RANK.get(required_education or "", 0)
    candidate = EDU_RANK.get(candidate_education or "", 0)
    if required == 0:
        return 1.0
    if candidate >= required:
        return 1.0
    if candidate == 0:
        return 0.0
    return 0.5 if required - candidate == 1 else 0.0


def _guttman_score(candidate_skills: list[str], job_skills: list[str], domain: str) -> float:
    levels = SKILL_LEVELS.get(domain or "", {})
    if not levels:
        return 0.0
    candidate_set = set(_ordered_unique(candidate_skills))
    job_set = set(_ordered_unique(job_skills))
    total_weight = 0
    achieved_weight = 0
    for index, (_, level_skills) in enumerate(levels.items(), start=1):
        skills_at_level = job_set & set(level_skills)
        if not skills_at_level:
            continue
        total_weight += index * len(skills_at_level)
        achieved_weight += index * len(skills_at_level & candidate_set)
    return achieved_weight / total_weight if total_weight else 0.0


def _uncertainty(skill_match_ratio: float, experience_score: float, education_score: float, missing_skills: list[str]) -> str:
    if skill_match_ratio < 0.25 or (missing_skills and len(missing_skills) > 2 * max(1, int(skill_match_ratio * 10))):
        return "high"
    if skill_match_ratio < 0.5 or experience_score < 0.75 or education_score < 1.0:
        return "medium"
    return "low"


def _explanation(
    matched_skills: list[str],
    required_skills: list[str],
    skill_match_ratio: float,
    candidate_years: int,
    required_years: int,
    experience_score: float,
    education_score: float,
) -> str:
    exp_text = "No explicit experience requirement." if required_years <= 0 else (
        f"Experience fit: {candidate_years}/{required_years} required years ({experience_score:.0%})."
    )
    edu_text = "Education requirement satisfied." if education_score >= 1.0 else "Education requirement is weaker or missing."
    return (
        f"Matched {len(matched_skills)} of {len(required_skills)} required skills "
        f"({skill_match_ratio:.0%} coverage). {exp_text} {edu_text}"
    )


def rank_candidates(
    job: dict,
    candidates: list[dict],
    limit: int = 10,
    include_zero_skill_matches: bool = False,
) -> list[RankedCandidate]:
    required_skills = _ordered_unique(job.get("skills", []))
    job_text = f"{job.get('title', '')}\n{job.get('description', '')}\n{' '.join(required_skills)}"
    candidate_texts = [
        f"{row.get('resume_text', '')}\n{' '.join(row.get('candidate_skills', []))}"
        for row in candidates
    ]
    semantic_scores = _semantic_scores(job_text, candidate_texts)
    ranked = []

    for row, semantic in zip(candidates, semantic_scores):
        candidate_skill_details = row.get("candidate_skill_details") or row.get("candidate_skills", [])
        candidate_skills = _ordered_unique([_skill_name(skill) for skill in candidate_skill_details])
        matched_skills, missing_skills, weighted_skill_score = _match_skills(required_skills, candidate_skill_details)
        if required_skills and not matched_skills and not include_zero_skill_matches:
            continue

        skill_match_ratio = _coverage(matched_skills, required_skills)
        jaccard_score = _jaccard(matched_skills, required_skills, candidate_skills)
        experience_score = _experience_score(
            int(row.get("experience_years") or 0),
            int(job.get("experience_years") or 0),
        )
        education_score = _education_score(row.get("education") or "", job.get("education") or "")
        guttman_score = _guttman_score(candidate_skills, required_skills, job.get("domain", ""))
        weighted_skill_ratio = min(weighted_skill_score / max(len(required_skills), 1), 1.0)

        score = (
            0.40 * skill_match_ratio
            + 0.15 * jaccard_score
            + 0.15 * weighted_skill_ratio
            + 0.10 * semantic
            + 0.10 * experience_score
            + 0.05 * education_score
            + 0.05 * guttman_score
        )
        uncertainty = _uncertainty(skill_match_ratio, experience_score, education_score, missing_skills)
        score_breakdown = {
            "skill_match_ratio": round(skill_match_ratio * 100, 2),
            "jaccard": round(jaccard_score * 100, 2),
            "weighted_skill": round(weighted_skill_ratio * 100, 2),
            "semantic": round(semantic * 100, 2),
            "experience_fit": round(experience_score * 100, 2),
            "education_fit": round(education_score * 100, 2),
            "guttman": round(guttman_score * 100, 2),
        }
        explanation = _explanation(
            matched_skills,
            required_skills,
            skill_match_ratio,
            int(row.get("experience_years") or 0),
            int(job.get("experience_years") or 0),
            experience_score,
            education_score,
        )
        ranked.append(
            RankedCandidate(
                candidate_id=row.get("candidate_id", ""),
                name=row.get("name", ""),
                email=row.get("email", ""),
                score=round(score * 100, 2),
                skill_score=round(skill_match_ratio * 100, 2),
                skill_match_ratio=round(skill_match_ratio * 100, 2),
                jaccard_score=round(jaccard_score * 100, 2),
                weighted_skill_score=round(weighted_skill_ratio * 100, 2),
                semantic_score=round(semantic * 100, 2),
                guttman_score=round(guttman_score * 100, 2),
                experience_score=round(experience_score * 100, 2),
                education_score=round(education_score * 100, 2),
                required_experience_years=int(job.get("experience_years") or 0),
                candidate_experience_years=int(row.get("experience_years") or 0),
                required_education=job.get("education") or "",
                candidate_education=row.get("education") or "",
                required_skills=required_skills,
                candidate_skills=candidate_skills,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                score_breakdown=score_breakdown,
                explanation=explanation,
                uncertainty=uncertainty,
            )
        )

    return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]


def consensus_review(ranked: list[RankedCandidate]) -> dict:
    if not ranked:
        return {"final_choice": None, "message": "No candidates matched the job."}
    top = ranked[0]
    disagreements = [item for item in ranked[:5] if item.uncertainty == "high"]
    if top.uncertainty == "high":
        message = "Top score is unstable because required skills are mostly missing. Human review recommended."
    elif disagreements:
        message = "Top recommendation is usable, but some shortlist candidates have high uncertainty."
    else:
        message = "Ranking and consensus checks agree on the top candidate."
    return {
        "final_choice": top,
        "message": message,
        "reviewed_count": min(len(ranked), 5),
    }
