from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any, TypedDict

from .extractors import extract_skills
from .graph import GraphStore
from .llm import LLMClient
from .normalizer import classify_domain, dedupe_skills, infer_education, infer_years
from .ranking import rank_candidates


class AgentState(TypedDict, total=False):
    job: dict[str, Any]
    candidates: list[dict[str, Any]]
    ranked: list[dict[str, Any]]
    consensus: dict[str, Any]
    graph_results: dict[str, Any]


class ExtractorAgent:
    """LLM-backed extractor for recruiter-side MA-Rank data."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def extract_job(
        self,
        title: str,
        company: str,
        description: str,
        location: str = "",
        domain_hint: str = "",
    ) -> dict[str, Any]:
        system = (
            "You are a JSON-only extractor for job posting data. "
            "Respond only with valid JSON, no markdown, no code blocks, no explanations."
        )
        domain = domain_hint or classify_domain(f"{title} {description}", job_title=title)
        prompt = _job_extraction_prompt(
            {
                "title": title,
                "company": company,
                "location": location,
                "description": description,
                "domain": domain,
            }
        )
        data = self.llm.extract_json(system, prompt)
        return _merge_extraction(
            data,
            source_text=f"{title} {description}",
            input_fields={
                "title": title,
                "company": company,
                "location": location,
                "description": description,
                "domain": domain,
            },
            require_domain=True,
        )

    def extract_resume(
        self,
        candidate_id: str,
        name: str,
        email: str,
        text: str,
        category: str = "",
        vocabulary: set[str] | None = None,
    ) -> dict[str, Any]:
        system = (
            "You are a JSON-only extractor for resume data. "
            "Respond only with valid JSON, no markdown, no code blocks, no explanations."
        )
        job_title = _extract_resume_job_title(text)
        domain = classify_domain(text, existing_domain=category, job_title=job_title)
        prompt = _resume_extraction_prompt(
            candidate_id=candidate_id,
            name=name,
            email=email,
            text=text,
            domain=domain,
            job_title=job_title,
        )
        data = self.llm.extract_json(system, prompt)
        merged = _merge_extraction(
            data,
            source_text=text,
            input_fields={"name": name, "email": email, "category": category, "resume_text": text},
            require_domain=False,
        )
        merged["candidate_id"] = candidate_id
        merged["resume_text"] = text
        if str(name or "").strip():
            merged["name"] = str(name).strip()
        if str(email or "").strip():
            merged["email"] = str(email).strip()
        merged["category"] = category
        return merged


class NormalizerAgent:
    def normalize_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return _normalize_record(job)

    def normalize_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return _normalize_record(candidate)


class GraphWriterAgent:
    def __init__(self, graph: GraphStore):
        self.graph = graph

    def write_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return {"jobs_written": self.graph.upsert_jobs([job])}

    def write_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return {"candidates_written": self.graph.upsert_candidates([candidate])}

    def write_jobs(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        return {"jobs_written": self.graph.upsert_jobs(jobs)}

    def write_candidates(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return {"candidates_written": self.graph.upsert_candidates(candidates)}


class MatcherRankerAgent:
    def __init__(self, graph: GraphStore):
        self.graph = graph

    def rank_for_job(
        self,
        job_id: str,
        limit: int = 10,
        candidate_pool_limit: int = 5000,
        include_zero_skill_matches: bool = False,
    ) -> list[dict[str, Any]]:
        job, candidates = self.graph.candidate_rank_pool_for_job(job_id, limit=candidate_pool_limit)
        if not job:
            return []
        return [
            asdict(item)
            for item in rank_candidates(
                job,
                candidates,
                limit=limit,
                include_zero_skill_matches=include_zero_skill_matches,
            )
        ]


class ConsensusAgent:
    def __init__(self, graph: GraphStore | None = None, llm: LLMClient | None = None):
        self.graph = graph
        self.llm = llm or LLMClient()

    def review(
        self,
        job: dict[str, Any],
        ranked: list[dict[str, Any]],
        recruiter_instructions: str = "",
        top_n: int = 5,
    ) -> dict[str, Any]:
        if not ranked:
            return {
                "final_candidate_id": None,
                "confidence": "low",
                "message": "No ranked candidates were available for consensus review.",
                "reason": "The matcher/ranker did not return any candidates.",
                "human_review_required": True,
                "risk_flags": ["No candidates returned"],
                "evidence_notes": [],
                "reviewed_candidate_ids": [],
            }

        top_candidates = ranked[:top_n]
        evidence_by_id = {}
        if self.graph is not None:
            evidence_rows = self.graph.get_candidates_by_ids([row.get("candidate_id", "") for row in top_candidates])
            evidence_by_id = {str(row.get("candidate_id")): row for row in evidence_rows}

        compact_candidates = [
            _compact_candidate_for_consensus(job, candidate, evidence_by_id.get(str(candidate.get("candidate_id")), {}))
            for candidate in top_candidates
        ]
        packet = {
            "job": {
                "job_id": job.get("job_id"),
                "title": job.get("title"),
                "company": job.get("company"),
                "domain": job.get("domain"),
                "description_excerpt": _job_description_excerpt(job.get("description", "")),
                "required_skills": job.get("skills", []),
                "required_experience_years": job.get("experience_years", 0),
                "required_education": job.get("education", ""),
            },
            "recruiter_instructions": recruiter_instructions,
            "top_candidates": compact_candidates,
        }
        system = (
            "You are the MA-Rank Consensus Agent, a second-pass recruiter evidence reviewer. "
            "Return only valid JSON. Do not use markdown. Do not invent skills, education, "
            "experience, or resume evidence."
        )
        prompt = (
            "Review the top ranked candidates for the job. Use the ranking scores, matched skills, "
            "missing skills, experience fit, education fit, recruiter instructions, and resume excerpts. "
            "The final reason must be resume-evidence-first: describe what the selected candidate's resume "
            "actually shows about their work, tools, and relevant experience. Keep the numeric MA-Rank score "
            "as secondary context only. Do not write a reason that only says the candidate has the highest "
            "score or the most matched skills. "
            "If recruiter instructions contain must-have constraints, prioritize candidates that satisfy them. "
            "If no candidate satisfies important requirements, choose the best available candidate but set "
            "human_review_required to true and list the risks.\n\n"
            "Return JSON with exactly these keys:\n"
            "{\n"
            '  "final_candidate_id": "candidate id string or null",\n'
            '  "confidence": "high, medium, or low",\n'
            '  "message": "short recruiter-facing recommendation",\n'
            '  "reason": "resume-evidence-first explanation of why this candidate is the best available option",\n'
            '  "human_review_required": true,\n'
            '  "risk_flags": ["missing must-have or evidence risk", "..."],\n'
            '  "evidence_notes": ["brief note grounded in resume excerpt or extracted fields", "..."],\n'
            '  "reviewed_candidate_ids": ["id1", "id2"]\n'
            "}\n\n"
            f"Evidence packet:\n{json.dumps(packet, ensure_ascii=True, indent=2)}"
        )
        try:
            response = self.llm.extract_json(system, prompt)
            return _normalize_consensus_response(response, compact_candidates, job, recruiter_instructions)
        except Exception as exc:
            return _deterministic_consensus(job, compact_candidates, recruiter_instructions, llm_error=str(exc))


def build_rank_workflow(graph: GraphStore):
    """Build the LangGraph workflow when installed; otherwise return a sequential callable."""
    matcher = MatcherRankerAgent(graph)
    consensus = ConsensusAgent(graph)

    def rank_node(state: AgentState) -> AgentState:
        job_id = state["job"]["job_id"]
        state["ranked"] = matcher.rank_for_job(job_id, limit=state.get("job", {}).get("limit", 10))
        return state

    def consensus_node(state: AgentState) -> AgentState:
        job_id = state["job"]["job_id"]
        job = graph.get_job(job_id) or {"job_id": job_id}
        state["consensus"] = consensus.review(
            job,
            state.get("ranked", []),
            state.get("job", {}).get("instructions", ""),
            top_n=state.get("job", {}).get("consensus_top_n", 5),
        )
        return state

    try:
        from langgraph.graph import END, START, StateGraph

        workflow = StateGraph(AgentState)
        workflow.add_node("matcher_ranker_agent", rank_node)
        workflow.add_node("consensus_agent", consensus_node)
        workflow.add_edge(START, "matcher_ranker_agent")
        workflow.add_edge("matcher_ranker_agent", "consensus_agent")
        workflow.add_edge("consensus_agent", END)
        return workflow.compile()
    except Exception:
        class SequentialWorkflow:
            def invoke(self, state: AgentState) -> AgentState:
                return consensus_node(rank_node(state))

        return SequentialWorkflow()


def _compact_candidate_for_consensus(
    job: dict[str, Any],
    candidate: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    resume_text = str(evidence.get("resume_text") or "")
    resume_excerpt = _resume_excerpt_for_consensus(resume_text, job, candidate) or str(candidate.get("resume_excerpt") or "")
    resume_overview = _resume_overview_for_consensus(resume_text) or str(candidate.get("resume_overview") or "")
    resume_evidence_summary = (
        str(candidate.get("resume_evidence_summary") or "").strip()
        or _resume_evidence_summary(candidate, resume_overview, resume_excerpt)
    )
    return {
        "candidate_id": candidate.get("candidate_id"),
        "name": candidate.get("name"),
        "email": candidate.get("email"),
        "rank_score": candidate.get("score"),
        "uncertainty": candidate.get("uncertainty"),
        "matched_skills": candidate.get("matched_skills", []),
        "missing_skills": candidate.get("missing_skills", []),
        "candidate_skills": evidence.get("candidate_skills") or candidate.get("candidate_skills", []),
        "required_experience_years": candidate.get("required_experience_years"),
        "candidate_experience_years": evidence.get("experience_years", candidate.get("candidate_experience_years")),
        "required_education": candidate.get("required_education"),
        "candidate_education": evidence.get("education", candidate.get("candidate_education", "")),
        "score_breakdown": candidate.get("score_breakdown", {}),
        "ranking_explanation": candidate.get("explanation", ""),
        "resume_overview": resume_overview,
        "resume_excerpt": resume_excerpt,
        "resume_evidence_summary": resume_evidence_summary,
    }


def _job_description_excerpt(description: Any, limit: int = 3000) -> str:
    cleaned = re.sub(r"\s+", " ", str(description or "")).strip()
    return cleaned[:limit]


def _resume_excerpt_for_consensus(text: str, job: dict[str, Any], candidate: dict[str, Any], limit: int = 2400) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    terms = set()
    for skill in list(job.get("skills", [])) + list(candidate.get("matched_skills", [])) + list(candidate.get("missing_skills", [])):
        skill_text = str(skill or "").lower().strip()
        if len(skill_text) > 2:
            terms.add(skill_text)
    snippets = []
    total = 0
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        normalized = re.sub(r"\s+", " ", sentence).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if any(term in lowered for term in terms):
            snippets.append(normalized)
            total += len(normalized)
        if total >= limit:
            break
    excerpt = " ".join(snippets).strip() or cleaned
    return excerpt[:limit]


def _resume_overview_for_consensus(text: str, limit: int = 1000) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    return cleaned[:limit]


def _resume_evidence_summary(candidate: dict[str, Any], overview: str, excerpt: str) -> str:
    matched = [str(skill) for skill in candidate.get("matched_skills", [])[:8]]
    missing = [str(skill) for skill in candidate.get("missing_skills", [])[:6]]
    years = candidate.get("candidate_experience_years")
    education = candidate.get("candidate_education") or "not confirmed"
    evidence_text = excerpt or overview
    evidence_sentence = ""
    if evidence_text:
        first_sentence = re.split(r"(?<=[.!?])\s+", evidence_text.strip(), maxsplit=1)[0]
        evidence_sentence = first_sentence[:360]
    parts = [
        f"Resume evidence indicates {years or 0} years of extracted experience and education as {education}.",
    ]
    if matched:
        parts.append(f"Relevant extracted skills include {', '.join(matched)}.")
    if evidence_sentence:
        parts.append(f"Resume excerpt evidence: {evidence_sentence}")
    if missing:
        parts.append(f"Important gaps remain around {', '.join(missing)}.")
    return " ".join(parts)


def _normalize_consensus_response(
    response: dict[str, Any],
    ranked: list[dict[str, Any]],
    job: dict[str, Any],
    recruiter_instructions: str = "",
) -> dict[str, Any]:
    candidate_ids = [str(row.get("candidate_id")) for row in ranked if row.get("candidate_id")]
    final_candidate_id = str(response.get("final_candidate_id") or "").strip()
    if final_candidate_id not in candidate_ids:
        final_candidate_id = candidate_ids[0] if candidate_ids else ""
    selected = next((row for row in ranked if str(row.get("candidate_id")) == final_candidate_id), ranked[0] if ranked else {})
    confidence = str(response.get("confidence") or "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    risk_flags = _string_list(response.get("risk_flags", []))
    if not risk_flags:
        risk_flags = _candidate_risk_flags(selected, job, recruiter_instructions)
    evidence_notes = _string_list(response.get("evidence_notes", []))
    if not evidence_notes:
        evidence_notes = [note for note in [
            selected.get("resume_evidence_summary"),
            selected.get("ranking_explanation"),
        ] if note]
    human_review_required = response.get("human_review_required")
    if not isinstance(human_review_required, bool):
        human_review_required = confidence != "high" or bool(risk_flags)
    reason = str(response.get("reason") or response.get("message") or "").strip()
    if not reason or reason.lower() in {"consensus review completed.", "consensus review completed"}:
        reason = _consensus_reason_from_candidate(selected)
    message = str(response.get("message") or "").strip()
    if not message or message.lower() in {"consensus review completed.", "consensus review completed"}:
        selected_name = selected.get("name") or final_candidate_id or "the selected candidate"
        message = f"Selected {selected_name} after reviewing ranking signals and resume evidence."
    reviewed = _string_list(response.get("reviewed_candidate_ids", [])) or candidate_ids
    return {
        "final_candidate_id": final_candidate_id or None,
        "confidence": confidence,
        "message": message,
        "reason": reason,
        "human_review_required": human_review_required,
        "risk_flags": risk_flags,
        "evidence_notes": evidence_notes,
        "reviewed_candidate_ids": [candidate_id for candidate_id in reviewed if candidate_id in candidate_ids],
    }


def _consensus_reason_from_candidate(candidate: dict[str, Any]) -> str:
    name = candidate.get("name") or candidate.get("candidate_id") or "This candidate"
    evidence_summary = str(candidate.get("resume_evidence_summary") or "").strip()
    if evidence_summary:
        base = f"Based on the resume evidence, {name} is the best available option because {evidence_summary}"
    else:
        base = f"Based on the extracted resume profile, {name} is the best available option because {_fallback_resume_reason(candidate)}"
    missing = [str(skill) for skill in candidate.get("missing_skills", [])[:6]]
    if missing:
        base += f" The main unresolved gaps are {', '.join(missing)}."
    return base


def _candidate_risk_flags(candidate: dict[str, Any], job: dict[str, Any], recruiter_instructions: str = "") -> list[str]:
    flags = []
    uncertainty = candidate.get("uncertainty")
    if uncertainty in {"medium", "high"}:
        flags.append(f"Ranker uncertainty is {uncertainty}.")
    missing = [str(skill) for skill in candidate.get("missing_skills", []) if str(skill).strip()]
    required_count = len(job.get("skills") or candidate.get("required_skills") or [])
    if missing and required_count:
        flags.append(f"Missing {len(missing)} of {required_count} required skills.")
    required_years = int(candidate.get("required_experience_years") or job.get("experience_years") or 0)
    candidate_years = int(candidate.get("candidate_experience_years") or 0)
    if required_years and candidate_years < required_years:
        flags.append(f"Experience evidence is {candidate_years}/{required_years} required years.")
    if candidate.get("required_education") and not candidate.get("candidate_education"):
        flags.append("Required education is not confirmed in extracted candidate profile.")
    flags.extend(_must_have_risks(recruiter_instructions, missing))
    return flags


def _deterministic_consensus(
    job: dict[str, Any],
    ranked: list[dict[str, Any]],
    recruiter_instructions: str = "",
    llm_error: str = "",
) -> dict[str, Any]:
    top = ranked[0] if ranked else {}
    missing = list(top.get("missing_skills", []))
    matched = list(top.get("matched_skills", []))
    risk_flags = []
    if top.get("uncertainty") in {"medium", "high"}:
        risk_flags.append(f"Ranker uncertainty is {top.get('uncertainty')}.")
    if missing:
        risk_flags.append(f"Missing {len(missing)} of {len(job.get('skills', top.get('required_skills', [])))} required skills.")
    required_years = int(top.get("required_experience_years") or job.get("experience_years") or 0)
    candidate_years = int(top.get("candidate_experience_years") or 0)
    if required_years and candidate_years < required_years:
        risk_flags.append(f"Experience evidence is {candidate_years}/{required_years} required years.")
    if top.get("required_education") and not top.get("candidate_education"):
        risk_flags.append("Required education is not confirmed in extracted candidate profile.")
    must_have_flags = _must_have_risks(recruiter_instructions, missing)
    risk_flags.extend(must_have_flags)
    if llm_error:
        risk_flags.append(f"LLM consensus unavailable: {llm_error[:180]}")
    confidence = "medium" if top.get("uncertainty") == "medium" and not must_have_flags else "low" if risk_flags else "high"
    evidence_summary = str(top.get("resume_evidence_summary") or "").strip()
    if not evidence_summary:
        evidence_summary = _fallback_resume_reason(top)
    gap_sentence = ""
    if missing:
        gap_sentence = f" The main unresolved gaps are {', '.join(str(skill) for skill in missing[:6])}."
    reason = (
        f"Based on the resume evidence, {top.get('name', top.get('candidate_id'))} is the best available option because "
        f"{evidence_summary}{gap_sentence}"
    )
    return {
        "final_candidate_id": top.get("candidate_id"),
        "confidence": confidence,
        "message": f"Selected {top.get('name', top.get('candidate_id'))} as the strongest available candidate from the ranked shortlist.",
        "reason": reason,
        "human_review_required": bool(risk_flags),
        "risk_flags": risk_flags,
        "evidence_notes": [note for note in [evidence_summary, top.get("ranking_explanation") or top.get("explanation", "")] if note],
        "reviewed_candidate_ids": [str(row.get("candidate_id")) for row in ranked if row.get("candidate_id")],
    }


def _fallback_resume_reason(candidate: dict[str, Any]) -> str:
    matched = [str(skill) for skill in candidate.get("matched_skills", [])[:8]]
    missing = [str(skill) for skill in candidate.get("missing_skills", [])[:6]]
    years = candidate.get("candidate_experience_years")
    education = candidate.get("candidate_education") or "not confirmed"
    pieces = [f"their extracted profile shows {years or 0} years of experience and education as {education}"]
    if matched:
        pieces.append(f"with relevant skills in {', '.join(matched)}")
    if missing:
        pieces.append(f"while still missing evidence for {', '.join(missing)}")
    return ", ".join(pieces) + "."


def _must_have_risks(instructions: str, missing_skills: list[str]) -> list[str]:
    lowered = str(instructions or "").lower()
    if not lowered:
        return []
    risks = []
    for skill in missing_skills:
        skill_text = str(skill or "").lower()
        if skill_text and skill_text in lowered:
            risks.append(f"Recruiter instruction mentions missing skill: {skill}.")
    return risks


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()]


def _resume_extraction_prompt(
    candidate_id: str,
    name: str,
    email: str,
    text: str,
    domain: str,
    job_title: str = "",
) -> str:
    skill_instructions = _domain_skill_instructions(domain, source="resume")
    return (
        "You are a JSON-only extractor for resume data. Respond ONLY with valid JSON, no markdown, no code blocks, no explanations.\n\n"
        "CRITICAL RULES - ABSOLUTELY NO HALLUCINATION:\n"
        "1. Extract ONLY information that is EXPLICITLY stated in the resume text below\n"
        "2. Do NOT infer, assume, guess, or fabricate any information\n"
        "3. Do NOT add skills based on job titles, education degrees, or responsibilities\n"
        "4. If a skill is not explicitly listed in the resume, DO NOT include it\n"
        "5. Carefully read through the entire resume text to find skills mentioned in:\n"
        "   - A dedicated 'Skills' section (if present)\n"
        "   - Job descriptions/experience sections that mention specific tools or technologies\n"
        "   - Project descriptions that list technologies used\n"
        "   - Any bullet points or lists that mention technical or business tools\n\n"
        "Extract the following information from the resume:\n"
        'Schema: {\n'
        '  "name": "string or null",\n'
        '  "email": "string or null",\n'
        '  "experience": "string summary of work experience or empty string",\n'
        '  "education": "string summary of education or empty string",\n'
        '  "domain": "sales, technology, or empty string",\n'
        '  "skills": ["skill1", "skill2", ...]\n'
        '}\n\n'
        f"{skill_instructions}"
        "SKILLS EXTRACTION - WHERE TO LOOK IN RESUME:\n"
        "1. Look for a dedicated 'Skills', 'Technical Skills', 'Tools', or 'Technologies' section\n"
        "2. Look in work experience/job descriptions that mention specific tools (e.g., 'Developed applications using Python and React')\n"
        "3. Look in project descriptions that list technologies used\n"
        "4. Extract skills that are explicitly mentioned, even if embedded in sentences (e.g., 'Worked with Django and PostgreSQL' -> extract 'django' and 'postgresql')\n"
        "5. If the resume says 'experience with X' or 'proficient in Y', extract X and Y as skills\n"
        "6. Extract all skills mentioned throughout the resume\n\n"
        "WHAT TO EXCLUDE (NEVER extract these as skills):\n"
        "- Soft skills: communication, leadership, teamwork, collaboration, problem-solving, time management, public speaking, presentation, negotiation, etc.\n"
        "- Generic phrases without tool names: 'experience with', 'familiarity with', 'knowledge of', 'proficiency in' (only extract if followed by a specific tool/technology)\n"
        "- Years of experience: '5 years', '3+ years', '10 years experience'\n"
        "- Education levels: 'Bachelor's degree', 'Master's', 'PhD' (extract in education field instead)\n"
        "- Company names, job titles, department names, or location names\n"
        "- Generic terms without context: experience, knowledge, familiarity, expertise, proficiency (without specific tool/technology name)\n\n"
        "Normalize skill names to lowercase and remove extra spaces (e.g., 'PYTHON' -> 'python', 'ML Ops' -> 'mlops', 'Salesforce CRM' -> 'salesforce crm').\n"
        "Return distinct skills only (no duplicates).\n"
        "If no explicit skills are mentioned in the resume, return an empty array [].\n\n"
        f"Candidate id: {candidate_id}\n"
        f"Known name: {name}\n"
        f"Known email: {email}\n"
        f"Detected job title hint: {job_title}\n"
        f"Detected domain hint: {domain}\n"
        f"Resume text:\n{text[:6000]}\n\n"
        "Return valid JSON only. Use null for missing name/email, empty string for missing experience/education/domain, empty array [] if no skills are explicitly mentioned."
    )


def _job_extraction_prompt(job: dict[str, Any]) -> str:
    title = str(job.get("title", "") or "")
    company = str(job.get("company", "") or "")
    location = str(job.get("location", "") or "")
    description = str(job.get("description", "") or "")
    domain = classify_domain(f"{title} {description}", existing_domain=job.get("domain", ""), job_title=title)
    skill_instructions = _domain_skill_instructions(domain, source="job description")
    return (
        "You are a JSON-only extractor for job posting data. Respond ONLY with valid JSON, no markdown, no code blocks, no explanations.\n\n"
        "CRITICAL RULES - ABSOLUTELY NO HALLUCINATION:\n"
        "1. Extract ONLY information that is EXPLICITLY stated in the job description text below\n"
        "2. Do NOT infer, assume, guess, or fabricate any information\n"
        "3. Do NOT add skills based on job titles, company names, or responsibilities\n"
        "4. If a skill is not explicitly listed in the job description, DO NOT include it\n"
        "5. Carefully read through the entire job description text to find skills mentioned in:\n"
        "   - Requirements/Qualifications sections\n"
        "   - 'Required Skills', 'Preferred Skills', 'Technical Skills', 'Must Have' sections\n"
        "   - Job responsibilities that mention specific tools, technologies, or platforms\n"
        "   - Any bullet points or lists that mention technical or business tools\n\n"
        "Extract the following information from the job posting:\n"
        'Schema: {\n'
        '  "skills": ["skill1", "skill2", ...],\n'
        '  "location": "string or empty string",\n'
        '  "experience": "REQUIRED - Extract years of experience requirement as a string (e.g., 3+ years, 5 years experience, minimum 2 years, 2-5 years, at least 4 years). If no experience requirement is mentioned, return empty string.",\n'
        '  "education": "string requirements or empty string (e.g., Bachelor\'s degree, Master\'s in Computer Science)",\n'
        '  "posting_date": "YYYY-MM-DD format or empty string",\n'
        '  "domain": "MUST be either sales or technology based on the job description. Classify the job as sales-related (sales, account management, business development) or technology-related (software, engineering, IT, data science). Return empty string if unclear."\n'
        '}\n\n'
        "EXPERIENCE EXTRACTION - IMPORTANT:\n"
        "- Look for phrases like: 'X years', 'X+ years', 'minimum X years', 'at least X years', 'X-Y years', 'X to Y years'\n"
        "- Extract the full experience requirement text exactly as written (e.g., '3+ years', '5 years of experience', 'minimum 2 years')\n"
        "- If the job mentions experience in ranges (e.g., '2-5 years'), extract the minimum value or the full range\n"
        "- If no experience requirement is specified, return empty string\n\n"
        f"{skill_instructions}"
        "SKILLS EXTRACTION - WHERE TO LOOK IN JOB DESCRIPTION:\n"
        "1. Look for sections titled: 'Requirements', 'Qualifications', 'Required Skills', 'Preferred Skills', 'Technical Skills', 'Must Have', 'Nice to Have'\n"
        "2. Look in bullet points that list skills, tools, or technologies\n"
        "3. Look in job responsibilities that mention specific tools (e.g., 'Experience with Python', 'Proficiency in Salesforce')\n"
        "4. Extract skills that are explicitly mentioned, even if embedded in sentences (e.g., 'Experience with React and Node.js' -> extract 'react' and 'node.js')\n"
        "5. If the description says 'experience with X' or 'knowledge of Y', extract X and Y as skills\n"
        "6. Extract all skills mentioned, whether required or preferred\n\n"
        "WHAT TO EXCLUDE (NEVER extract these as skills):\n"
        "- Soft skills: communication, leadership, teamwork, collaboration, problem-solving, time management, public speaking, presentation, negotiation, etc.\n"
        "- Generic phrases without tool names: 'experience with', 'familiarity with', 'knowledge of', 'proficiency in' (only extract if followed by a specific tool/technology)\n"
        "- Years of experience: '5 years', '3+ years', '10 years experience', 'minimum 2 years'\n"
        "- Education requirements: 'Bachelor's degree', 'Master's', 'PhD' (extract in education field instead)\n"
        "- Company names, job titles, department names, or location names\n"
        "- Generic terms without context: experience, knowledge, familiarity, expertise, proficiency (without specific tool/technology name)\n"
        "- Requirements that are not skills: 'remote work', 'full-time', 'contract', salary ranges, etc.\n\n"
        "DOMAIN CLASSIFICATION:\n"
        "- Return 'technology' for: software engineering, IT, data science, programming, technical roles, engineering positions\n"
        "- Return 'sales' for: sales roles, account management, business development, revenue generation, client relationship roles\n"
        "- Return empty string if the job doesn't clearly fit into either category\n\n"
        "Normalize skill names to lowercase and remove extra spaces (e.g., 'PYTHON' -> 'python', 'ML Ops' -> 'mlops', 'Salesforce CRM' -> 'salesforce crm').\n"
        "Return distinct skills only (no duplicates).\n"
        "If no explicit skills are mentioned in the job description, return an empty array [].\n\n"
        f"Job Title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location}\n"
        f"Detected domain hint: {domain}\n"
        f"Job Description:\n{description[:6000]}\n\n"
        "Return valid JSON only. Use empty string for missing fields, empty array [] if no skills are explicitly mentioned."
    )


def _domain_skill_instructions(domain: str, source: str) -> str:
    if domain == "sales":
        return (
            "SKILLS EXTRACTION FOR SALES DOMAIN:\n"
            f"Extract ONLY the following types of skills IF EXPLICITLY mentioned in the {source}:\n"
            "- CRM systems: Salesforce, HubSpot, Zoho, Microsoft Dynamics, Pipedrive, etc.\n"
            "- Sales tools: Outreach.io, Gong, ZoomInfo, LinkedIn Sales Navigator, etc.\n"
            "- Marketing platforms: Marketo, HubSpot Marketing, Mailchimp, etc.\n"
            "- Social media and ad platforms: Hootsuite, Buffer, Sprout Social, Meta Ads, Facebook Ads, Instagram Ads, LinkedIn Ads, Google Ads, X/Twitter Ads, etc.\n"
            "- Digital sales and marketing skills: social media marketing, social media management, email marketing, marketing automation, campaign management, lead generation, prospecting, cold calling, pipeline management, forecasting, territory management, account management, B2B sales, enterprise sales.\n"
            "- Sales methodologies: SPIN, MEDDIC, Challenger Sale, etc.\n"
            "- Business intelligence tools: Tableau, Power BI, etc. (if mentioned)\n"
            "- Payment/transaction systems: Stripe, PayPal, etc. (if mentioned)\n"
            "- DO NOT extract programming languages or technical IT skills unless explicitly listed\n"
            "- DO NOT infer technical skills based on job titles or responsibilities\n\n"
        )
    if domain == "technology":
        return (
            "SKILLS EXTRACTION FOR TECHNOLOGY DOMAIN:\n"
            f"Extract ONLY the following types of skills IF EXPLICITLY mentioned in the {source}:\n"
            "- Programming languages: Python, Java, JavaScript, TypeScript, C++, C#, Go, Rust, Ruby, PHP, Swift, Kotlin, etc.\n"
            "- Frameworks & Libraries: React, Angular, Vue, Node.js, Django, Flask, Spring, TensorFlow, PyTorch, Scikit-learn, etc.\n"
            "- APIs and integration: REST API, RESTful APIs, SOAP, GraphQL, OpenAPI/Swagger, Postman, API integration, microservices.\n"
            "- Tools & Technologies: Docker, Kubernetes, Git, Jenkins, GitHub Actions, GitLab CI, CircleCI, Terraform, Ansible, CI/CD tools, etc.\n"
            "- Platforms & Cloud: AWS, EC2, S3, Lambda, IAM, Azure, Azure DevOps, GCP, BigQuery, Heroku, DigitalOcean, Snowflake, Databricks, etc.\n"
            "- Databases and data systems: SQL, PostgreSQL, MySQL, SQL Server, MongoDB, Redis, Cassandra, Elasticsearch, Neo4j, Oracle, PL/SQL, etc.\n"
            "- Operating Systems: Linux, Unix, Windows Server, macOS, etc.\n"
            "- Distributed systems and architecture: distributed systems, system design, scalability, cloud architecture, message queues, RabbitMQ, Kafka.\n"
            "- Testing and verification: software testing, QA testing, test automation, Selenium, Cypress, JUnit, PyTest, unit testing, integration testing, regression testing, verification and validation.\n"
            "- Methods and practices: Agile, Scrum, SDLC, Machine Learning, Deep Learning, NLP, Computer Vision, Data Science, MLOps, DevOps, ETL, data engineering.\n"
            "- Specific technologies: Spark, Hadoop, Kafka, RabbitMQ, GraphQL, REST API, Microservices, Airflow, Tableau, Power BI, etc.\n"
            "- DO NOT extract sales tools or CRM systems unless explicitly mentioned\n\n"
        )
    return (
        "SKILLS EXTRACTION:\n"
        "Extract ONLY skills that are EXPLICITLY mentioned, whether technical or business-related.\n"
        "Do NOT infer skills based on job titles, education, or responsibilities.\n\n"
    )


def _merge_extraction(
    data: dict[str, Any],
    source_text: str,
    input_fields: dict[str, Any],
    require_domain: bool = True,
) -> dict[str, Any]:
    merged = dict(input_fields)
    for key, value in data.items():
        if value is None:
            continue
        if _has_content(merged.get(key)) and not _has_content(value):
            continue
        merged[key] = value
    merged["experience_years"] = _coerce_experience_years(
        merged.get("experience_years"),
        merged.get("experience") or source_text,
    )
    merged["education"] = _normalize_llm_education(merged.get("education", ""))
    domain = str(merged.get("domain", "")).strip().lower()
    if domain not in {"sales", "technology"}:
        domain = classify_domain(
            source_text,
            input_fields.get("category", ""),
            input_fields.get("title", ""),
        )
    if domain not in {"sales", "technology"}:
        if require_domain:
            raise ValueError(f"Could not classify record as sales or technology. LLM returned: {merged.get('domain')!r}")
        domain = ""
    merged["domain"] = domain
    skills = _coerce_skill_list(merged.get("skills", []))
    if source_text:
        skills = [skill for skill in skills if _skill_appears_in_text(str(skill), source_text)]
    merged["skills"] = dedupe_skills(skills)
    return merged


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized["skills"] = dedupe_skills(record.get("skills", []))
    normalized["education"] = infer_education(record.get("education", "")) or record.get("education", "")
    normalized["experience_years"] = int(record.get("experience_years") or 0)
    normalized["domain"] = classify_domain(
        record.get("description") or record.get("resume_text", ""),
        record.get("domain") or record.get("category", ""),
        record.get("title", ""),
    )
    return normalized


def _object_view(row: dict[str, Any]):
    class View:
        def __init__(self, data):
            self.__dict__.update(data)
    return View(row)


def _coerce_experience_years(value: Any, fallback_text: Any = "") -> int:
    if isinstance(value, bool):
        raise ValueError("experience_years must be an integer, not boolean.")
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str) and value.strip():
        parsed = infer_years(value)
        if parsed:
            return parsed
        if value.strip().isdigit():
            return int(value.strip())
    parsed = infer_years(fallback_text)
    return max(parsed, 0)


def _normalize_llm_education(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"empty string", "empty", "none", "null", "n/a", "na", "not specified", "not mentioned"}:
        return ""
    normalized = infer_education(text)
    if normalized:
        return normalized
    lowered = text.lower()
    if lowered in {"bachelor's", "master's", "phd"}:
        return lowered
    if any(term in lowered for term in ["certificate", "certification", "diploma", "completion", "associate"]):
        return ""
    return ""


def _skill_appears_in_text(skill: str, text: str) -> bool:
    text_clean = re.sub(r"<[^>]+>", " ", text).lower()
    text_clean = re.sub(r"\s+", " ", text_clean)
    skill_lower = skill.lower().strip()
    if skill_lower in text_clean:
        return True
    skill_clean = re.sub(r"[^\w\s-]", "", skill_lower)
    words = [word for word in skill_clean.split() if len(word) > 1]
    if len(words) > 1 and all(word in text_clean for word in words):
        return True
    if len(words) == 1 and words[0] in text_clean:
        return True
    variations = {
        skill_clean,
        skill_clean.replace(" ", ""),
        skill_clean.replace("-", ""),
        skill_clean.replace("_", ""),
        skill_clean.replace(" ", "-"),
        skill_clean.replace("-", " "),
    }
    return any(variation and len(variation) > 2 and variation in text_clean for variation in variations)


def _coerce_skill_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none", "empty", "empty array"}:
            return []
        raw = re.split(r"[,;|]", text)
    else:
        raw = [value]
    skills = []
    for item in raw:
        if isinstance(item, dict):
            item = item.get("name") or item.get("skill") or item.get("text") or ""
        text = str(item).strip()
        if text:
            skills.append(text)
    return skills


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {
            "",
            "null",
            "none",
            "empty",
            "empty string",
            "n/a",
            "na",
            "not specified",
            "not mentioned",
        }
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _extract_resume_job_title(text: str) -> str:
    lines = text.splitlines()[:10]
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if not lowered or "@" in lowered or "email" in lowered or "phone" in lowered:
            continue
        if any(keyword in lowered for keyword in ["title", "position", "role", "current role"]):
            parts = re.split(r"[:|*]", stripped, maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                return parts[1].strip()
        if len(stripped) < 60 and not re.search(r"\d{3}[-.]?\d{3}[-.]?\d{4}", stripped):
            indicators = [
                "developer", "engineer", "manager", "sales", "executive", "analyst",
                "specialist", "consultant", "architect", "administrator", "director",
            ]
            if any(indicator in lowered for indicator in indicators):
                return stripped
    return ""
