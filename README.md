# MA-Rank: Multi-Agent Resume Screening and Skill Mining

This is the recruiter-facing project MA-Rank

- job description intake
- resume/job parsing into structured fields
- skill normalization
- Neo4j graph writing
- candidate search and explainable ranking
- a lightweight consensus check that flags uncertain top results

## 1. Create a Python environment

```powershell
cd "C:\Adarsh Home\Projects\Data Mining Final Project\MA-Rank"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` after Neo4j is running.

## 2. Set up Neo4j

### Option A: Local Neo4j with Docker

Install Docker Desktop, then run:

```powershell
cd "C:\Adarsh Home\Projects\Data Mining Final Project\MA-Rank"
docker compose up -d
```

Open the Neo4j Browser:

```text
http://localhost:7474
```

Use:

```text
Username: neo4j
Password: change-this-password
```

Keep `.env` as:

```text
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change-this-password
```

### Option B: Neo4j AuraDB

Create a free AuraDB instance at Neo4j Aura, then copy the connection details into `.env`:

```text
NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-generated-password>
```

Do not hardcode credentials in Python files.

## 3. Load the graph

First prepare the cleaned/augmented files so you can inspect the dataset before LLM import:

```powershell
python scripts\prepare_data.py --job-limit 5000 --resume-limit 500
```

This writes:

```text
outputs\resumes_augmented.csv
outputs\resumes_augmented.xlsx
outputs\jobs_augmented.csv
outputs\jobs_augmented.xlsx
outputs\jobs_sales_technology.csv
outputs\jobs_sales_technology.xlsx
```

`jobs_sales_technology.csv` is the MA-Rank job dataset. It is built by joining
`postings.csv` with `jobs\job_skills.csv` and `mappings\skills.csv`. LinkedIn
skill tags such as `SALE`, `BD`, `IT`, `ENG`, and `QA` are used to assign the
job domain, with extra title/description checks for technology jobs.

Then start with a smaller import so the pipeline is fast:

```powershell
python scripts\import_data.py --job-limit 5000 --resume-limit 500 --clear
```

The script reads:

```text
..\LinkedInJobPostings (2023-2024)\postings.csv
..\LinkedInJobPostings (2023-2024)\jobs\job_skills.csv
..\LinkedInJobPostings (2023-2024)\mappings\skills.csv
..\ResumeDataset\Resume\Resume.csv
```

Jobs are filtered to sales/technology from the LinkedIn job-skill tags before
LLM extraction. Candidates are not hard-filtered by resume domain; they are
written as a broader pool and ranked by extracted skill overlap, experience,
and education.

It creates this graph shape:

```text
(Category)-[:HAS_JOB]->(Job)-[:REQUIRES]->(Skill)<-[:HAS_SKILL]-(Candidate)
```

The code also stores candidate/job metadata such as title, company, location, experience years, education, and resume text.

## 4. Run the recruiter app

```powershell
streamlit run app.py
```

The app opens a recruiter console where you can:

- add a new job description
- add a new resume/candidate
- inspect jobs in Neo4j
- rank candidates for a selected job
- view required skills, candidate skills, matched skills, missing skills, skill match ratio, experience fit, education fit, scoring components, explanations, and uncertainty flags

You can also rank candidates from the terminal once the graph is loaded:

```powershell
python scripts\rank_candidates.py <job_id> --limit 10
```

Run the evidence-aware consensus review over the ranked shortlist:

```powershell
python scripts\rank_candidates.py <job_id> --limit 10 --consensus --instructions "Must have PHP and WordPress"
```

This prints JSON with the fields used by the recruiter-facing agents:

```text
required_skills
candidate_skills
matched_skills
missing_skills
skill_match_ratio
jaccard_score
weighted_skill_score
semantic_score
guttman_score
experience_score
education_score
score_breakdown
uncertainty
explanation
```

With `--consensus`, the output also includes:

```text
final_candidate_id
confidence
message
reason
human_review_required
risk_flags
evidence_notes
reviewed_candidate_ids
```

## Agent Architecture

The project now follows the agents described in `Group16_ProgessReport.docx`:

- `ExtractorAgent`: parses resumes and job descriptions. It uses Ollama or Gemini when configured, with deterministic fallback.
- `NormalizerAgent`: canonicalizes skills, education, and experience fields.
- `GraphWriterAgent`: writes structured candidates, jobs, and skills to Neo4j.
- `MatcherRankerAgent`: retrieves candidates from Neo4j and ranks them with hybrid scoring.
- `ConsensusAgent`: re-checks the top 5 shortlist, pulls their resume text from Neo4j, reviews recruiter instructions and evidence, then selects or flags the final candidate.

The ranking workflow uses LangGraph when `langgraph` is installed. If it is not installed, the same agents run sequentially so the app still works.

Ported recruiter-side behavior from `Agentic-Career-Assistant-main`:

- soft-skill filtering before KG writes
- no-hallucination skill validation for LLM extraction
- richer experience parsing for ranges and minimum-year phrases
- education normalization to bachelor's, master's, and phd
- sales/technology category classification, including Salesforce technical-context handling
- `Category -> HAS_JOB -> Job` graph structure
- `Skill.norm_name`, `Skill.aliases`, `Skill.embedding`, and `Skill.created_at`
- embedding-backed skill canonicalization with Neo4j alias persistence
- candidate/job data augmentation for names, emails, posted dates, experience defaults, and Excel-safe text cleanup
- recruiter-side weighted candidate matching with exact, alias, substring, token-overlap, and rarity/popularity adjustments
- hybrid ranking with required-skill coverage, Jaccard overlap, weighted skill evidence, semantic similarity, Guttman hierarchy, experience fit, and education fit

To use local LLM agents with Ollama:

```powershell
ollama pull llama3.1:8b
ollama serve
```

Set `.env`:

```text
MA_RANK_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

To use OpenAI as the primary LLM provider:

```text
MA_RANK_LLM_PROVIDER=openai
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=gpt-4.1-mini
```

To use Gemini instead:

```text
MA_RANK_LLM_PROVIDER=gemini
GEMINI_API_KEY=<your-key>
GEMINI_MODEL=gemini-2.5-flash
```

## Useful Neo4j checks

Run these in Neo4j Browser:

```cypher
MATCH (n) RETURN labels(n), count(*) ORDER BY count(*) DESC;
```

```cypher
MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)
RETURN c.name, collect(s.name)[0..10] AS skills
LIMIT 10;
```

```cypher
MATCH (j:Job)-[:REQUIRES]->(s:Skill)
RETURN j.title, j.company, collect(s.name)[0..10] AS required_skills
LIMIT 10;
```

## Notes

This first version uses deterministic parsing and TF-IDF semantic scoring so it works without paid LLM keys. The old project had Gemini/Ollama pieces; those can be added later as stronger extractor and consensus agents once the graph-backed recruiter workflow is stable.
