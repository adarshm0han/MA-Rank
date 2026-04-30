from __future__ import annotations

from contextlib import AbstractContextManager

from neo4j import GraphDatabase

from .config import Neo4jConfig
from .normalizer import EmbeddingSkillNormalizer, classify_domain, dedupe_skills, infer_education


class GraphStore(AbstractContextManager):
    def __init__(self, config: Neo4jConfig):
        self.driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
        self.normalizer = EmbeddingSkillNormalizer(self.driver)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        self.driver.close()

    def verify(self) -> None:
        self.driver.verify_connectivity()

    def init_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (c:Candidate) REQUIRE c.candidate_id IS UNIQUE",
            "CREATE CONSTRAINT job_id IF NOT EXISTS FOR (j:Job) REQUIRE j.job_id IS UNIQUE",
            "CREATE CONSTRAINT skill_norm_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.norm_name IS UNIQUE",
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (cat:Category) REQUIRE cat.name IS UNIQUE",
            "CREATE INDEX job_title IF NOT EXISTS FOR (j:Job) ON (j.title)",
            "CREATE INDEX candidate_name IF NOT EXISTS FOR (c:Candidate) ON (c.name)",
        ]
        with self.driver.session() as session:
            for statement in statements:
                session.run(statement)

    def clear(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def upsert_candidates(self, candidates: list[dict]) -> int:
        rows = []
        for candidate in candidates:
            domain = candidate.get("domain") or classify_domain(
                candidate.get("resume_text", ""),
                candidate.get("domain") or candidate.get("category", ""),
            )
            if domain not in {"sales", "technology"}:
                domain = ""
            skills = [self._skill_payload(skill) for skill in dedupe_skills(candidate.get("skills", []))]
            rows.append({
                **candidate,
                "domain": domain,
                "education": infer_education(candidate.get("education", "")) or candidate.get("education", ""),
                "skills": skills,
            })
        query = """
        UNWIND $rows AS row
        MERGE (c:Candidate {candidate_id: row.candidate_id})
        SET c.name = row.name,
            c.email = row.email,
            c.category = row.category,
            c.domain = row.domain,
            c.resume_text = row.resume_text,
            c.experience_years = coalesce(row.experience_years, 0),
            c.experience = coalesce(row.experience_years, 0),
            c.education = coalesce(row.education, "")
        WITH c, row
        FOREACH (skill IN row.skills |
            MERGE (s:Skill {norm_name: skill.norm_name})
            ON CREATE SET s.id = randomUUID(), s.name = skill.name,
                s.aliases = skill.aliases, s.embedding = skill.embedding, s.created_at = datetime()
            ON MATCH SET s.name = coalesce(s.name, skill.name),
                s.embedding = coalesce(s.embedding, skill.embedding)
            MERGE (c)-[:HAS_SKILL]->(s)
        )
        RETURN count(DISTINCT c) AS count
        """
        with self.driver.session() as session:
            record = session.run(query, rows=rows).single()
            return int(record["count"] if record else 0)

    def upsert_jobs(self, jobs: list[dict]) -> int:
        rows = []
        for job in jobs:
            domain = str(job.get("domain", "") or "").strip().lower()
            if domain not in {"sales", "technology"}:
                domain = classify_domain(job.get("description", ""), job.get("domain", ""), job.get("title", ""))
            if domain not in {"sales", "technology"}:
                continue
            skills = [self._skill_payload(skill) for skill in dedupe_skills(job.get("skills", []))]
            rows.append({
                **job,
                "domain": domain,
                "education": infer_education(job.get("education", "")) or job.get("education", ""),
                "skills": skills,
            })
        query = """
        UNWIND $rows AS row
        MERGE (j:Job {job_id: row.job_id})
        SET j.title = row.title,
            j.company = row.company,
            j.location = row.location,
            j.description = row.description,
            j.experience_years = coalesce(row.experience_years, 0),
            j.experience = coalesce(row.experience_years, 0),
            j.education = coalesce(row.education, ""),
            j.posting_date = coalesce(row.posting_date, ""),
            j.domain = row.domain,
            j.linkedin_skill_abrs = coalesce(row.linkedin_skill_abrs, []),
            j.linkedin_skill_names = coalesce(row.linkedin_skill_names, []),
            j.source = coalesce(row.source, "manual")
        WITH j, row
        FOREACH (_ IN CASE WHEN row.domain IS NULL OR row.domain = "" THEN [] ELSE [1] END |
            MERGE (cat:Category {name: row.domain})
            ON CREATE SET cat.id = randomUUID()
            MERGE (cat)-[:HAS_JOB]->(j)
        )
        WITH j, row
        FOREACH (skill IN row.skills |
            MERGE (s:Skill {norm_name: skill.norm_name})
            ON CREATE SET s.id = randomUUID(), s.name = skill.name,
                s.aliases = skill.aliases, s.embedding = skill.embedding, s.created_at = datetime()
            ON MATCH SET s.name = coalesce(s.name, skill.name),
                s.embedding = coalesce(s.embedding, skill.embedding)
            MERGE (j)-[:REQUIRES]->(s)
        )
        RETURN count(DISTINCT j) AS count
        """
        with self.driver.session() as session:
            record = session.run(query, rows=rows).single()
            return int(record["count"] if record else 0)

    def get_job(self, job_id: str) -> dict | None:
        query = """
        MATCH (j:Job {job_id: $job_id})
        OPTIONAL MATCH (j)-[:REQUIRES|REQUIRES_SKILL]->(s:Skill)
        WITH j, [skill IN collect(DISTINCT coalesce(s.norm_name, s.name)) WHERE skill IS NOT NULL] AS skills
        RETURN j {.*, skills: skills} AS job
        """
        with self.driver.session() as session:
            record = session.run(query, job_id=str(job_id)).single()
            return record["job"] if record else None

    def list_jobs(self, limit: int = 200) -> list[dict]:
        query = """
        MATCH (j:Job)
        OPTIONAL MATCH (j)-[:REQUIRES|REQUIRES_SKILL]->(s:Skill)
        RETURN j.job_id AS job_id, j.title AS title, j.company AS company,
               j.location AS location, j.domain AS domain, collect(DISTINCT coalesce(s.norm_name, s.name)) AS skills
        ORDER BY j.title
        LIMIT $limit
        """
        with self.driver.session() as session:
            return [record.data() for record in session.run(query, limit=limit)]

    def candidates_for_job(self, job_id: str, limit: int = 25) -> list[dict]:
        query = """
        MATCH (j:Job {job_id: $job_id})
        OPTIONAL MATCH (j)-[:REQUIRES|REQUIRES_SKILL]->(required:Skill)
        WITH j, collect(DISTINCT coalesce(required.norm_name, required.name)) AS required_skills
        MATCH (c:Candidate)
        OPTIONAL MATCH (c)-[:HAS_SKILL]->(candidate_skill:Skill)
        WITH j, c, required_skills, collect(DISTINCT coalesce(candidate_skill.norm_name, candidate_skill.name)) AS candidate_skills
        WITH j, c, required_skills, candidate_skills,
             [skill IN candidate_skills WHERE skill IN required_skills] AS matched_skills
        WITH j, c, required_skills, candidate_skills, matched_skills,
             [skill IN required_skills WHERE NOT skill IN matched_skills] AS missing_skills
        WHERE size(matched_skills) > 0 OR size(required_skills) = 0
        RETURN c.candidate_id AS candidate_id,
               c.name AS name,
               c.email AS email,
               c.category AS category,
               c.experience_years AS experience_years,
               c.education AS education,
               c.resume_text AS resume_text,
               candidate_skills,
               required_skills,
               matched_skills,
               missing_skills
        LIMIT $limit
        """
        with self.driver.session() as session:
            return [record.data() for record in session.run(query, job_id=str(job_id), limit=limit)]

    def candidate_rank_pool_for_job(self, job_id: str, limit: int = 5000) -> tuple[dict | None, list[dict]]:
        job = self.get_job(job_id)
        if not job:
            return None, []
        query = """
        MATCH (c:Candidate)
        OPTIONAL MATCH (c)-[:HAS_SKILL]->(candidate_skill:Skill)
        WITH c, candidate_skill,
             COUNT { (candidate_skill)<-[:HAS_SKILL]-(:Candidate) } AS popularity
        WITH c,
             collect(DISTINCT CASE WHEN candidate_skill IS NULL THEN null ELSE {
                name: coalesce(candidate_skill.norm_name, candidate_skill.name),
                aliases: coalesce(candidate_skill.aliases, []),
                popularity: popularity
             } END) AS skill_details
        WITH c, [skill IN skill_details WHERE skill IS NOT NULL AND skill.name IS NOT NULL] AS candidate_skill_details
        RETURN c.candidate_id AS candidate_id,
               c.name AS name,
               c.email AS email,
               c.category AS category,
               c.domain AS domain,
               c.experience_years AS experience_years,
               c.education AS education,
               c.resume_text AS resume_text,
               [skill IN candidate_skill_details | skill.name] AS candidate_skills,
               candidate_skill_details
        ORDER BY size(candidate_skill_details) DESC, c.candidate_id
        LIMIT $limit
        """
        with self.driver.session() as session:
            candidates = [record.data() for record in session.run(query, limit=limit)]
        return job, candidates

    def get_candidates_by_ids(self, candidate_ids: list[str]) -> list[dict]:
        ids = [str(candidate_id) for candidate_id in candidate_ids if str(candidate_id or "").strip()]
        if not ids:
            return []
        query = """
        MATCH (c:Candidate)
        WHERE c.candidate_id IN $candidate_ids
        OPTIONAL MATCH (c)-[:HAS_SKILL]->(s:Skill)
        WITH c, [skill IN collect(DISTINCT coalesce(s.norm_name, s.name)) WHERE skill IS NOT NULL] AS skills
        RETURN c.candidate_id AS candidate_id,
               c.name AS name,
               c.email AS email,
               c.category AS category,
               c.domain AS domain,
               c.experience_years AS experience_years,
               c.education AS education,
               c.resume_text AS resume_text,
               skills AS candidate_skills
        """
        with self.driver.session() as session:
            rows = [record.data() for record in session.run(query, candidate_ids=ids)]
        by_id = {str(row.get("candidate_id")): row for row in rows}
        return [by_id[candidate_id] for candidate_id in ids if candidate_id in by_id]

    def rank_candidates_weighted(self, required_skills: list[str], limit: int = 50, domain: str = "") -> list[dict]:
        required = dedupe_skills(required_skills)
        query = """
        WITH [s IN $skills WHERE s IS NOT NULL | toLower(trim(s))] AS reqs
        UNWIND reqs AS _
        WITH collect(distinct _) AS reqs
        MATCH (c:Candidate)-[:HAS_SKILL]->(sk:Skill)
        WITH c, sk, reqs,
             COUNT { (sk)<-[:HAS_SKILL]-(:Candidate) } AS popularity
        WITH c, collect(DISTINCT {
                name: coalesce(sk.norm_name, sk.name),
                aliases: coalesce(sk.aliases, []),
                popularity: popularity
             }) AS skillObjs, reqs
        UNWIND skillObjs AS so
        WITH c, so, reqs,
             CASE
               WHEN any(r IN reqs WHERE so.name = r) THEN 3.0
               WHEN any(r IN reqs WHERE any(a IN so.aliases WHERE toLower(trim(a)) = r)) THEN 2.5
               WHEN any(r IN reqs WHERE r IN so.name OR so.name IN r) THEN 1.5
               WHEN any(r IN reqs WHERE size([t IN split(so.name,' ') WHERE t IN split(r,' ')]) > 0) THEN 1.0
               ELSE 0.0
             END AS base_weight
        WITH c, so, base_weight,
             CASE WHEN base_weight > 0 THEN base_weight * (1.0 / (1.0 + log(1.0 + toFloat(coalesce(so.popularity,0))))) ELSE 0.0 END AS adjusted_weight,
             CASE WHEN base_weight > 0 THEN so.name ELSE null END AS matched_skill,
             so.name AS all_skill
        WITH c, collect(adjusted_weight) AS weights,
             [x IN collect(matched_skill) WHERE x IS NOT NULL] AS matched_skills,
             collect(all_skill) AS candidate_skills
        WHERE size(matched_skills) > 0
        RETURN c.candidate_id AS candidate_id, c.name AS name, c.email AS email,
               c.category AS category, c.domain AS domain, c.experience_years AS experience_years,
               c.education AS education, c.resume_text AS resume_text,
               candidate_skills, matched_skills,
               reduce(acc = 0.0, w IN weights | acc + w) AS weighted_skill_score
        ORDER BY weighted_skill_score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            return [record.data() for record in session.run(query, skills=required, limit=limit, domain=domain or "")]

    def _skill_payload(self, skill: str) -> dict:
        canonical, cleaned, embedding = self.normalizer.normalize(skill)
        return {
            "name": canonical or cleaned,
            "norm_name": canonical or cleaned,
            "aliases": [] if canonical == cleaned else [cleaned],
            "embedding": embedding,
        }
