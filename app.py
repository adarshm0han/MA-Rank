from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from ma_rank.agents import ExtractorAgent, GraphWriterAgent, NormalizerAgent, build_rank_workflow
from ma_rank.config import get_neo4j_config
from ma_rank.graph import GraphStore


st.set_page_config(page_title="MA-Rank Recruiter Console", layout="wide")


@st.cache_resource
def get_graph() -> GraphStore:
    graph = GraphStore(get_neo4j_config())
    graph.verify()
    graph.init_schema()
    return graph


@st.cache_resource
def get_agents():
    return ExtractorAgent(), NormalizerAgent()


def format_skills(skills: list[str]) -> str:
    return ", ".join(skills) if skills else "None found"


def main() -> None:
    st.title("MA-Rank Recruiter Console")
    st.caption("Multi-agent style resume screening with Neo4j-backed skill matching and explainable rankings.")

    graph = get_graph()
    extractor_agent, normalizer_agent = get_agents()
    graph_writer_agent = GraphWriterAgent(graph)
    tab_search, tab_add_job, tab_add_resume, tab_jobs = st.tabs(["Rank Candidates", "Add Job", "Add Resume", "Jobs"])

    with tab_add_job:
        st.subheader("Job Description Uploader Agent")
        st.caption("Runs Extractor Agent -> Normalizer Agent -> Neo4j Graph Writer Agent.")
        company = st.text_input("Company", value="")
        title = st.text_input("Job title", value="")
        location = st.text_input("Location", value="")
        description = st.text_area("Job description", height=260)
        if st.button("Parse and save job", type="primary"):
            if not title.strip() or not description.strip():
                st.warning("Job title and description are required.")
            else:
                parsed = extractor_agent.extract_job(title, company, description, location)
                parsed = normalizer_agent.normalize_job(parsed)
                parsed["job_id"] = f"manual-{uuid.uuid4()}"
                parsed["source"] = "manual"
                graph_writer_agent.write_job(parsed)
                st.success("Job saved to Neo4j.")
                st.write("Extracted skills:", format_skills(parsed["skills"]))

    with tab_add_resume:
        st.subheader("Resume Uploader Agent")
        st.caption("Runs Resume Parser Agent -> Normalizer Agent -> Neo4j Graph Writer Agent.")
        candidate_id = st.text_input("Candidate ID", value="", key="candidate_id")
        candidate_name = st.text_input("Candidate name", value="", key="candidate_name")
        candidate_email = st.text_input("Candidate email", value="", key="candidate_email")
        category = st.text_input("Original resume category", value="", key="candidate_category")
        uploaded_resume = st.file_uploader("Upload resume text file", type=["txt"], key="resume_upload")
        resume_text = st.text_area("Resume text", height=260, key="resume_text")
        if uploaded_resume is not None and not resume_text.strip():
            resume_text = uploaded_resume.read().decode("utf-8", errors="ignore")
            st.text_area("Loaded resume text", value=resume_text, height=160, disabled=True)

        if st.button("Parse and save resume", type="primary"):
            if not resume_text.strip():
                st.warning("Resume text is required.")
            else:
                parsed = extractor_agent.extract_resume(
                    candidate_id=candidate_id.strip() or f"manual-candidate-{uuid.uuid4()}",
                    name=candidate_name.strip() or "Unknown Candidate",
                    email=candidate_email.strip() or f"candidate-{uuid.uuid4()}@example.com",
                    text=resume_text,
                    category=category,
                    vocabulary=None,
                )
                parsed = normalizer_agent.normalize_candidate(parsed)
                graph_writer_agent.write_candidate(parsed)
                st.success("Candidate saved to Neo4j.")
                st.write("Extracted skills:", format_skills(parsed["skills"]))

    with tab_jobs:
        st.subheader("Known Jobs")
        jobs = graph.list_jobs(limit=300)
        if jobs:
            st.dataframe(pd.DataFrame(jobs), use_container_width=True, hide_index=True)
        else:
            st.info("No jobs found. Import data or add a job first.")

    with tab_search:
        st.subheader("Candidate Search Agent")
        st.caption("Runs Matcher/Ranker Agent -> Consensus Agent. Uses LangGraph when installed.")
        jobs = graph.list_jobs(limit=500)
        if not jobs:
            st.info("No jobs are available yet. Run the import script or add a job.")
            return

        labels = {
            f"{job['title']} at {job['company']} ({job['job_id']})": job["job_id"]
            for job in jobs
        }
        selected = st.selectbox("Select a job", list(labels.keys()))
        shortlist_size = st.slider("Shortlist size", min_value=5, max_value=25, value=10)
        instructions = st.text_area("Recruiter must-have instructions for Consensus Agent", height=100)

        if st.button("Rank candidates", type="primary"):
            job_id = labels[selected]
            workflow = build_rank_workflow(graph)
            result = workflow.invoke({
                "job": {
                    "job_id": job_id,
                    "limit": shortlist_size,
                    "instructions": instructions,
                    "consensus_top_n": min(5, shortlist_size),
                }
            })
            ranked = result.get("ranked", [])
            review = result.get("consensus", {})

            st.subheader("Consensus Agent")
            st.write(review.get("message", "No consensus output returned."))
            cols = st.columns(3)
            cols[0].metric("Confidence", review.get("confidence", "n/a"))
            cols[1].metric("Human review", "Required" if review.get("human_review_required") else "Not required")
            cols[2].metric("Reviewed", len(review.get("reviewed_candidate_ids", [])))
            if review.get("reason"):
                st.write(review["reason"])
            if review.get("risk_flags"):
                st.warning("Risk flags: " + "; ".join(review["risk_flags"]))
            if review.get("evidence_notes"):
                st.info("Evidence notes: " + "; ".join(review["evidence_notes"]))
            final_id = review.get("final_candidate_id")
            if final_id:
                final = next((item for item in ranked if item.get("candidate_id") == final_id), None)
                if final:
                    st.metric("Final recommendation", final.get("name", final_id), f"{final.get('score', 0)}%")

            st.subheader("Ranked Candidates")
            rows = [
                {
                    "rank": idx + 1,
                    "name": item.get("name"),
                    "email": item.get("email"),
                    "score": item.get("score"),
                    "skill_score": item.get("skill_score"),
                    "jaccard_score": item.get("jaccard_score"),
                    "weighted_skill_score": item.get("weighted_skill_score"),
                    "semantic_score": item.get("semantic_score"),
                    "guttman_score": item.get("guttman_score"),
                    "experience_score": item.get("experience_score"),
                    "education_score": item.get("education_score"),
                    "candidate_experience": item.get("candidate_experience_years"),
                    "required_experience": item.get("required_experience_years"),
                    "candidate_education": item.get("candidate_education"),
                    "required_education": item.get("required_education"),
                    "uncertainty": item.get("uncertainty"),
                    "matched_skills": format_skills(item.get("matched_skills", [])),
                    "missing_skills": format_skills(item.get("missing_skills", [])),
                    "candidate_skills": format_skills(item.get("candidate_skills", [])),
                    "explanation": item.get("explanation"),
                }
                for idx, item in enumerate(ranked)
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
