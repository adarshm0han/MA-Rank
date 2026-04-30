from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterable
from typing import Optional


ALIASES = {
    "a p": "accounts payable",
    "ap": "accounts payable",
    "ar": "accounts receivable",
    "ads": "advertising",
    "adp": "adp payroll",
    "10 key by touch": "10-key typing",
    "ten key by touch": "10-key typing",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "js": "javascript",
    "node.js": "node.js",
    "react.js": "react",
    "ms word": "microsoft word",
    "excel": "microsoft excel",
    "ms excel": "microsoft excel",
    "microsoft excel spreadsheets": "microsoft excel",
    "excel spreadsheets": "microsoft excel",
    "powerpoint": "microsoft powerpoint",
    "power point": "microsoft powerpoint",
    "ms powerpoint": "microsoft powerpoint",
    "microsoft power point": "microsoft powerpoint",
    "word": "microsoft word",
    "ms office": "microsoft office suite",
    "microsoft office": "microsoft office suite",
    "office suite": "microsoft office suite",
    "microsoft outlook": "outlook",
    "ms outlook": "outlook",
    "microsoft access": "access",
    "sklearn": "scikit learn",
    "scikit-learn": "scikit learn",
    "adobe in design": "adobe indesign",
    "in design": "adobe indesign",
    "indesign": "adobe indesign",
    "adobe illustrator cc": "adobe illustrator",
    "illustrator": "adobe illustrator",
    "photoshop": "adobe photoshop",
    "adobe photoshop cc": "adobe photoshop",
    "adobe creative cloud": "adobe creative suite",
    "creative cloud": "adobe creative suite",
    "html5": "html",
    "html 5": "html",
    "css3": "css",
    "css 3": "css",
    "xhtml css": "html css",
    "html css coding": "html css",
    "intermediate html coding": "html",
    "javascript programming": "javascript",
    "java script": "javascript",
    "angularjs": "angular.js",
    "angular js": "angular.js",
    "node js": "node.js",
    "nodejs": "node.js",
    "structured query language": "sql",
    "sql databases": "sql",
    "sql database": "sql",
    "sql programming": "sql",
    "sql queries": "sql",
    "experience with sql": "sql",
    "knowledge of sql": "sql",
    "proficiency in sql": "sql",
    "postgres": "postgresql",
    "postgre sql": "postgresql",
    "postgre": "postgresql",
    "postgressql": "postgresql",
    "postgreSQL": "postgresql",
    "postgres sql": "postgresql",
    "postgresql database": "postgresql",
    "experience with postgresql": "postgresql",
    "knowledge of postgresql": "postgresql",
    "mysql database": "mysql",
    "my sql": "mysql",
    "mssql": "sql server",
    "ms sql": "sql server",
    "microsoft sql": "sql server",
    "microsoft sql server": "sql server",
    "sqlserver": "sql server",
    "sql server database": "sql server",
    "pl sql": "pl/sql",
    "plsql": "pl/sql",
    "apis": "api",
    "api development": "api",
    "api design": "api",
    "api integration": "api integration",
    "rest": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "rest apis": "rest api",
    "web api": "api",
    "web apis": "api",
    "soap api": "soap",
    "soap web services": "soap",
    "simple object access protocol": "soap",
    "open api": "openapi",
    "open api specification": "openapi",
    "swagger": "openapi",
    "amazon web services": "aws",
    "aws cloud": "aws",
    "aws ec2": "ec2",
    "amazon ec2": "ec2",
    "aws s3": "s3",
    "amazon s3": "s3",
    "aws lambda": "lambda",
    "amazon lambda": "lambda",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "g cloud": "gcp",
    "microsoft azure": "azure",
    "azure cloud": "azure",
    "azure dev ops": "azure devops",
    "azure pipelines": "azure devops",
    "cloud computing": "cloud",
    "cloud architecture": "cloud architecture",
    "distributed system": "distributed systems",
    "distributed computing": "distributed systems",
    "systems design": "system design",
    "system architecture": "system design",
    "scalable systems": "scalability",
    "scalable architecture": "scalability",
    "micro service": "microservices",
    "micro services": "microservices",
    "message queue": "message queues",
    "message queues": "message queues",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "continuous integration": "ci/cd",
    "continuous deployment": "ci/cd",
    "continuous delivery": "ci/cd",
    "github actions": "github actions",
    "gitlab ci": "gitlab ci",
    "circle ci": "circleci",
    "circleci": "circleci",
    "test automation": "test automation",
    "automated testing": "test automation",
    "automation testing": "test automation",
    "software testing": "software testing",
    "qa testing": "software testing",
    "quality assurance testing": "software testing",
    "tdd": "test driven development",
    "test-driven development": "test driven development",
    "test driven dev": "test driven development",
    "test driven development tdd": "test driven development",
    "unit tests": "unit testing",
    "unit test": "unit testing",
    "verification validation": "verification and validation",
    "verification and validation": "verification and validation",
    "validation verification": "verification and validation",
    "v v": "verification and validation",
    "v&v": "verification and validation",
    "unit testing": "unit testing",
    "integration testing": "integration testing",
    "regression testing": "regression testing",
    "selenium webdriver": "selenium",
    "agile software development methodology": "agile",
    "agile software development": "agile",
    "agile methodology": "agile",
    "agile methodologies": "agile",
    "scrum methodology": "scrum",
    "scrum framework": "scrum",
    "software development life cycle": "sdlc",
    "systems development life cycle": "sdlc",
    "etl development": "etl",
    "extract transform load": "etl",
    "apache spark": "spark",
    "apache kafka": "kafka",
    "apache airflow": "airflow",
    "power bi": "power bi",
    "microsoft power bi": "power bi",
    "tableau desktop": "tableau",
    "wordpress development": "wordpress",
    "wordpress developer": "wordpress",
    "salesforce crm": "salesforce",
    "hubspot crm": "hubspot",
    "zoho crm": "zoho",
    "microsoft dynamics crm": "microsoft dynamics",
    "dynamics crm": "microsoft dynamics",
    "linkedin sales navigator": "linkedin sales navigator",
    "sales navigator": "linkedin sales navigator",
    "outreach io": "outreach.io",
    "outreach.io": "outreach.io",
    "zoom info": "zoominfo",
    "zoominfo": "zoominfo",
    "hootsuite": "hootsuite",
    "buffer": "buffer",
    "sprout social": "sprout social",
    "social media marketing": "social media marketing",
    "social media management": "social media management",
    "facebook ads": "meta ads",
    "facebook advertising": "meta ads",
    "meta advertising": "meta ads",
    "meta ads manager": "meta ads",
    "instagram ads": "meta ads",
    "instagram advertising": "meta ads",
    "google adwords": "google ads",
    "google advertising": "google ads",
    "linkedin ads": "linkedin ads",
    "linkedin advertising": "linkedin ads",
    "twitter ads": "x ads",
    "twitter advertising": "x ads",
    "x advertising": "x ads",
    "email marketing": "email marketing",
    "mail chimp": "mailchimp",
    "hubspot marketing": "hubspot",
    "marketing automation": "marketing automation",
    "lead gen": "lead generation",
    "lead-generation": "lead generation",
    "prospecting": "prospecting",
    "cold calls": "cold calling",
    "cold-calling": "cold calling",
    "sales pipeline": "pipeline management",
    "pipeline": "pipeline management",
    "sales forecasting": "forecasting",
    "forecasting": "forecasting",
    "territory management": "territory management",
    "account management": "account management",
    "enterprise sales": "enterprise sales",
    "b2b sales": "b2b sales",
    "b2c sales": "b2c sales",
}

SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "collaboration", "problem solving",
    "problem-solving", "time management", "organization", "analytical thinking",
    "critical thinking", "creativity", "adaptability", "flexibility", "work ethic",
    "interpersonal", "presentation", "negotiation", "mentoring", "coaching",
    "detail oriented", "attention to detail", "multitasking", "self motivated",
    "self-motivated", "proactive", "initiative", "decision making", "decision-making",
    "strategic thinking", "business acumen", "customer service", "client facing",
    "written communication", "verbal communication", "public speaking", "training",
    "teaching", "supervision", "people management", "emotional intelligence",
    "conflict resolution", "stress management", "work-life balance",
}

BARE_CATEGORY_SKILLS = {
    "sales",
    "technology",
    "tech",
    "information technology",
    "it",
    "business",
    "operations",
    "marketing",
}

SHORT_SKILL_ALLOWLIST = {
    "c", "r", "go", "qa", "ui", "ux", "it", "hr", "bi", "ml", "ai", "js",
    "c#", "c++", "sql", "aws", "gcp", "crm", "erp", "api", "css", "html",
}

GENERIC_SKILL_TERMS = {
    "basic", "competitive", "clients", "client", "content", "clarify", "autonomy",
    "backup", "administrative", "administrative skills", "business", "work",
    "professional", "responsible", "responsibilities", "required", "preferred",
    "plus", "must have", "nice to have", "tools", "systems", "software",
}

GENERIC_SKILL_SUFFIXES = {
    " is", " are", " and", " or", " with", " using", " including",
}

TECHNICAL_DOMAIN_TERMS = [
    "apex", "lightning", "lwc", "visualforce", "vlocity", "copado", "soql",
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "react", "angular", "vue", "node.js", "django", "flask", "spring",
    "docker", "kubernetes", "jenkins", "bitbucket", "github", "git", "jira",
    "aws", "azure", "gcp", "sql", "postgresql", "mysql", "mongodb",
    "terraform", "ansible", "ci/cd", "cicd", "devops", "mlops",
]

TECH_TITLE_KEYWORDS = [
    "developer", "engineer", "programmer", "architect", "administrator", "admin",
    "scientist", "analyst", "specialist", "consultant", "technician", "technologist",
    "software", "hardware", "systems", "network", "database", "devops", "sre",
    "security", "cybersecurity", "qa", "quality assurance", "test", "testing",
    "data engineer", "data scientist", "machine learning", "ai", "ml engineer",
    "frontend", "backend", "full stack", "mobile", "web developer",
    "salesforce developer", "salesforce architect", "salesforce admin",
]

SALES_TITLE_KEYWORDS = [
    "sales", "account executive", "account manager", "account representative",
    "business development", "bd", "inside sales", "outside sales",
    "sales manager", "sales director", "sales representative", "sales rep",
    "sales specialist", "sales consultant", "sales associate",
    "territory manager", "regional sales", "national sales",
    "client acquisition", "customer success", "account management",
]


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s/-]", " ", text)
    text = re.sub(r"[\s/_-]+", " ", text)
    return text.strip()


def normalize_skill(value: object) -> str:
    cleaned = clean_text(value)
    cleaned = strip_skill_wrapper(cleaned)
    return ALIASES.get(cleaned, cleaned)


def strip_skill_wrapper(value: str) -> str:
    """Remove non-skill wrappers that LLMs often include around real skills."""
    text = value.strip()
    wrappers = [
        r"^(?:hands on\s+)?experience\s+(?:with|in|using)\s+",
        r"^(?:working\s+)?knowledge\s+of\s+",
        r"^proficiency\s+(?:with|in)\s+",
        r"^proficient\s+(?:with|in)\s+",
        r"^familiarity\s+(?:with|in)\s+",
        r"^familiar\s+with\s+",
        r"^expertise\s+(?:with|in)\s+",
        r"^skilled\s+(?:with|in)\s+",
        r"^ability\s+to\s+use\s+",
        r"^using\s+",
    ]
    changed = True
    while changed:
        changed = False
        for wrapper in wrappers:
            stripped = re.sub(wrapper, "", text).strip()
            if stripped != text:
                text = stripped
                changed = True
    return text


def is_soft_skill(skill: object) -> bool:
    skill_lower = normalize_skill(skill)
    if not skill_lower:
        return True
    if skill_lower in SOFT_SKILLS:
        return True
    soft_keywords = [
        "communication", "leadership", "teamwork", "people management",
        "problem solving", "interpersonal", "presentation", "negotiation",
        "time management", "organization", "multitasking", "self motivated",
        "customer service", "client facing", "public speaking",
    ]
    for keyword in soft_keywords:
        if keyword == skill_lower or (len(skill_lower) < 30 and keyword in skill_lower):
            if any(tech in skill_lower for tech in ["api", "sdk", "framework", "library", "tool", "platform"]):
                continue
            return True
    return skill_lower == "management"


def filter_skills(skills: Iterable[object]) -> list[str]:
    filtered = []
    seen = set()
    for skill in skills:
        norm = normalize_skill(skill)
        if not norm or norm == "null":
            continue
        if len(norm) < 3 and norm not in SHORT_SKILL_ALLOWLIST:
            continue
        if norm in GENERIC_SKILL_TERMS:
            continue
        if any(norm.endswith(suffix) for suffix in GENERIC_SKILL_SUFFIXES):
            continue
        compact = norm.replace("+", "").replace("-", "").replace(" ", "")
        if compact.isdigit():
            continue
        if is_soft_skill(norm):
            continue
        if norm in BARE_CATEGORY_SKILLS:
            continue
        if any(phrase in norm for phrase in [
            "years of", "years experience", "year of experience", "years' experience",
            "years of experience", "year experience", "plus years",
        ]):
            continue
        if norm in {"bachelor", "master", "phd", "doctorate", "degree", "diploma", "certification"}:
            continue
        if norm in {"experience", "knowledge", "familiarity", "proficiency", "expertise"}:
            continue
        if norm not in seen:
            seen.add(norm)
            filtered.append(norm)
    return filtered


def dedupe_skills(skills: Iterable[object]) -> list[str]:
    return filter_skills(skills)


def split_skills(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return dedupe_skills(value)
    text = str(value)
    parts = re.split(r"[,;|]", text)
    return dedupe_skills(parts)


def infer_years(text: object) -> int:
    if text is None:
        return 0
    lowered = str(text).lower()
    patterns = [
        r"(\d+)\s*\+\s*(?:years?|yrs?)",
        r"(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?experience",
        r"minimum\s*(\d+)\s*(?:years?|yrs?)",
        r"at\s*least\s*(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\s+to\s+(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\s*(?:years?|yrs?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    if "year" in lowered or "experience" in lowered:
        numbers = [int(n) for n in re.findall(r"\b([1-9]|[1-4][0-9]|50)\b", lowered)]
        if numbers:
            return numbers[0]
    return 0


def infer_education(text: object) -> str:
    lowered = clean_text(text)
    if re.search(r"ph\.?d\.?|doctorate|d\.?phil|phd\b", lowered):
        return "phd"
    if re.search(r"master|m\.?s\.?|m\.?a\.?|m\.?e\.?|m\.?tech|mba|msc|ms\b|m\.?eng", lowered):
        return "master's"
    if re.search(r"bachelor|b\.?s\.?|b\.?a\.?|b\.?e\.?|b\.?tech|undergraduate|bsc|bs\b", lowered):
        return "bachelor's"
    return ""


def classify_domain(text: object = "", existing_domain: object = "", job_title: object = "") -> str:
    """
    Classify domain as 'sales' or 'technology' based primarily on job title.
    Falls back to text analysis if job title is not available or unclear.
    This mirrors the old Agentic Career Assistant extractor classifier.
    """
    if existing_domain:
        if isinstance(existing_domain, list):
            existing_domain = " ".join(str(item) for item in existing_domain) if existing_domain else ""
        elif not isinstance(existing_domain, str):
            existing_domain = str(existing_domain) if existing_domain else ""

        existing_lower = existing_domain.lower().strip()
        if existing_lower in ["sales", "technology"]:
            if existing_lower == "sales" and text:
                text_lower_check = text.lower() if isinstance(text, str) else str(text).lower()
                technical_indicators = [
                    "apex", "lightning", "lwc", "vlocity", "copado", "jenkins",
                    "bitbucket", "github", "jira", "developer", "engineer",
                    "programming", "coding", "python", "java", "javascript",
                ]
                has_tech_indicators = any(ind in text_lower_check for ind in technical_indicators)
                if not has_tech_indicators:
                    return existing_lower
            else:
                return existing_lower

    if job_title:
        job_title_lower = str(job_title).lower().strip()

        tech_title_keywords = [
            "developer", "engineer", "programmer", "architect", "administrator", "admin",
            "scientist", "analyst", "specialist", "consultant", "technician", "technologist",
            "software", "hardware", "systems", "network", "database", "devops", "sre",
            "security", "cybersecurity", "qa", "quality assurance", "test", "testing",
            "data engineer", "data scientist", "machine learning", "ai", "ml engineer",
            "frontend", "backend", "full stack", "mobile", "web developer",
            "salesforce developer", "salesforce architect", "salesforce admin",
        ]

        sales_title_keywords = [
            "sales", "account executive", "account manager", "account representative",
            "business development", "bd", "inside sales", "outside sales",
            "sales manager", "sales director", "sales representative", "sales rep",
            "sales specialist", "sales consultant", "sales associate",
            "territory manager", "regional sales", "national sales",
            "client acquisition", "customer success", "account management",
        ]

        for keyword in tech_title_keywords:
            if keyword in job_title_lower:
                return "technology"

        for keyword in sales_title_keywords:
            if keyword in job_title_lower:
                return "sales"

    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text_lower = text.lower()

    technical_skill_indicators = [
        "apex", "lightning", "lwc", "visualforce", "vlocity", "copado", "soql",
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "react", "angular", "vue", "node.js", "django", "flask", "spring",
        "docker", "kubernetes", "jenkins", "bitbucket", "github", "git", "jira",
        "aws", "azure", "gcp", "sql", "postgresql", "mysql", "mongodb",
        "terraform", "ansible", "ci/cd", "cicd", "devops", "mlops",
    ]
    has_technical_skills = any(indicator in text_lower for indicator in technical_skill_indicators)

    tech_keywords = [
        "software developer", "software engineer", "programming", "coding",
        "engineer", "engineering", "technical", "technology", "it", "computer science",
        "data science", "machine learning", "ai", "artificial intelligence", "devops",
        "cloud", "database", "api", "backend", "frontend",
        "full stack", "web development", "mobile development", "cybersecurity",
        "system administrator", "system admin", "infrastructure",
        "salesforce developer", "salesforce architect", "salesforce admin",
    ]

    sales_keywords = [
        "account executive", "account manager", "business development",
        "bd", "revenue", "territory", "quota", "lead generation", "prospecting",
        "closing", "deal", "pipeline", "inside sales", "outside sales",
        "account management", "client acquisition", "sales representative",
    ]

    sales_mentioned = "sales" in text_lower
    salesforce_technical_context = False
    if sales_mentioned and "salesforce" in text_lower:
        salesforce_pos = text_lower.find("salesforce")
        context_window = text_lower[max(0, salesforce_pos - 100):salesforce_pos + 200]
        technical_context_indicators = [
            "developer", "architect", "admin", "apex", "lightning", "lwc",
            "vlocity", "copado", "customization", "integration", "soql",
            "jenkins", "bitbucket", "github", "git", "deployment",
        ]
        salesforce_technical_context = any(indicator in context_window for indicator in technical_context_indicators)

    tech_count = sum(1 for keyword in tech_keywords if keyword in text_lower)
    sales_count = sum(1 for keyword in sales_keywords if keyword in text_lower)
    if sales_mentioned and not salesforce_technical_context and "sales" not in [kw for kw in sales_keywords]:
        if re.search(r"\bsales\b", text_lower) and not salesforce_technical_context:
            sales_count += 1

    if has_technical_skills:
        tech_count += 3

    if tech_count > sales_count and tech_count > 0:
        return "technology"
    if sales_count > tech_count and sales_count > 0:
        return "sales"
    if tech_count == sales_count and tech_count > 0:
        if has_technical_skills:
            return "technology"
        return "technology"
    return ""


def _salesforce_technical_context(text: str) -> bool:
    if "salesforce" not in text:
        return False
    pos = text.find("salesforce")
    window = text[max(0, pos - 100):pos + 200]
    indicators = ["developer", "architect", "admin", "apex", "lightning", "lwc", "vlocity", "copado", "soql"]
    return any(indicator in window for indicator in indicators)


class EmbeddingSkillNormalizer:
    """Embedding-backed skill canonicalizer with Neo4j alias persistence."""

    def __init__(self, neo4j_driver=None, threshold: Optional[float] = None):
        self.driver = neo4j_driver
        self.threshold = threshold or float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
        self.local_cache = {}
        self.lock = threading.Lock()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = os.getenv("SKILL_NORMALIZER_DEVICE", "cpu").lower()
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        return self._model

    def normalize(self, raw_skill: object) -> tuple[str, str, list[float]]:
        cleaned = normalize_skill(raw_skill)
        if not cleaned:
            return "", "", []
        if cleaned in self.local_cache:
            canon, embedding = self.local_cache[cleaned]
            return canon, cleaned, embedding
        embedding = self.model.encode(cleaned, convert_to_numpy=True).astype(float).tolist()
        with self.lock:
            db_match, db_embedding = self._find_db_match(cleaned, embedding)
            if db_match:
                self.local_cache[cleaned] = (db_match, db_embedding or embedding)
                if cleaned != db_match:
                    self._add_alias(db_match, cleaned)
                return db_match, cleaned, db_embedding or embedding
            self._store_skill(cleaned, embedding)
            self.local_cache[cleaned] = (cleaned, embedding)
            return cleaned, cleaned, embedding

    def _find_db_match(self, cleaned: str, embedding: list[float]) -> tuple[str | None, list[float] | None]:
        if not self.driver:
            return None, None
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            with self.driver.session() as session:
                exact = session.run(
                    """
                    OPTIONAL MATCH (s:Skill)
                    WHERE s.norm_name IS NOT NULL AND (
                        toLower(s.norm_name) = $skill OR
                        any(alias IN coalesce(s.aliases, []) WHERE toLower(alias) = $skill)
                    )
                    RETURN s.norm_name AS norm_name, s.embedding AS embedding
                    LIMIT 1
                    """,
                    skill=cleaned,
                ).single()
                if exact and exact["norm_name"]:
                    return exact["norm_name"], exact["embedding"]
                best_name = None
                best_embedding = None
                best_score = self.threshold
                rows = session.run(
                    "OPTIONAL MATCH (s:Skill) WHERE s.norm_name IS NOT NULL RETURN s.norm_name AS norm_name, s.embedding AS embedding"
                )
                for row in rows:
                    emb = row["embedding"]
                    if not emb:
                        continue
                    score = float(cosine_similarity([embedding], [emb])[0][0])
                    if score > best_score:
                        best_score = score
                        best_name = row["norm_name"]
                        best_embedding = emb
                return best_name, best_embedding
        except Exception:
            return None, None

    def _store_skill(self, norm_name: str, embedding: list[float]) -> None:
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run(
                """
                MERGE (s:Skill {norm_name: $norm_name})
                ON CREATE SET s.id = randomUUID(), s.name = $norm_name,
                    s.aliases = [], s.embedding = $embedding, s.created_at = datetime()
                ON MATCH SET s.embedding = coalesce(s.embedding, $embedding)
                """,
                norm_name=norm_name,
                embedding=embedding,
            )

    def _add_alias(self, norm_name: str, alias: str) -> None:
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run(
                """
                MATCH (s:Skill {norm_name: $norm_name})
                SET s.aliases = CASE
                    WHEN $alias IN coalesce(s.aliases, []) THEN coalesce(s.aliases, [])
                    ELSE coalesce(s.aliases, []) + $alias
                END
                """,
                norm_name=norm_name,
                alias=alias,
            )
