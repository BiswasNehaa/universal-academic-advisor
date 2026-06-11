# =============================================================================
# advisor.py
# PURPOSE : AI-powered Academic Advisor using RAG + LLM reasoning.
#
# PIPELINE PER QUERY:
#   Stage A  — Career keyword lookup       (career.json → dynamic, scalable)
#   Stage B  — Course history enrichment   (Python fast-path → LLM fallback)
#   Stage C  — FAISS retrieval             (semantic search + foundation sweep)
#   Stage D  — Python-side filtering       (prereqs, career relevance, dedupe)
#   Stage E  — LLM roadmap generation      (structured JSON output)
#   Stage F  — Pretty-print to terminal
#
# DEPENDENCIES:
#   pip install langchain langchain-groq langchain-community langchain-huggingface
#               faiss-cpu sentence-transformers python-dotenv
#
# FILES REQUIRED:
#   faiss_index/        — built by ingest.py
#   ../Data/courses.json
#   ../Data/career.json
#   .env                — must contain GROQ_API_KEY=...
# =============================================================================

import os
import re
import json
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# =============================================================================
# STARTUP — Load all shared resources once at import time.
# Every query reuses these objects; no repeated loading.
# =============================================================================

# Embedding model — MUST match the model used in ingest.py.
# Changing this without rebuilding the index will produce garbage results.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM — temperature=0 for deterministic, repeatable advice.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# FAISS vector store — built once by ingest.py, loaded read-only here.
vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True   # safe: we built this file ourselves
)

# Full course catalog — used by enrichment to map names → course codes.
COURSES_JSON_PATH = '../Data/courses.json'
with open(COURSES_JSON_PATH, 'r') as f:
    ALL_COURSES = json.load(f)

# Career skills map — maps career titles to required skill keywords.
# Adding new careers requires only editing career.json, zero code changes.
CAREER_JSON_PATH = '../Data/career.json'
with open(CAREER_JSON_PATH, 'r') as f:
    CAREER_DATA = json.load(f)

# Maximum credits a student can register per semester.
# Validated at input time; used to cap enroll_now suggestions.
MAX_CREDITS = 30


# =============================================================================
# STAGE A — Career Keyword Lookup
# =============================================================================

def get_career_keywords(career_goal: str) -> list[str]:
    """
    Match the student's free-text career goal against career.json keys
    using a simple substring/word overlap approach (no fuzz dependency).

    Returns the required_skills list for the best matching career, or []
    if no match is found (LLM will then infer relevance on its own).

    Examples:
        "AI Engineer"    → matches "ai_engineer"    → ["machine learning", ...]
        "web developer"  → matches "web_developer"  → ["html", "css", ...]
        "rocket science" → no match                 → []
    """
    goal_words = set(career_goal.lower().replace('_', ' ').split())

    best_match = None
    best_score = 0

    for career_key in CAREER_DATA.keys():
        key_words  = set(career_key.lower().replace('_', ' ').split())
        # Score = number of words shared between goal and key
        overlap    = len(goal_words & key_words)
        if overlap > best_score:
            best_score = overlap
            best_match = career_key

    # Require at least one meaningful word overlap
    if best_score >= 1 and best_match:
        print(f"[Career Matched] '{career_goal}' → '{best_match}'")
        return CAREER_DATA[best_match]['required_skills']

    print(f"[Career] No match found for '{career_goal}'. LLM will infer relevance.")
    return []


# =============================================================================
# STAGE B — Completed Course Enrichment
# =============================================================================

def enrich_completed_list(completed_names: list[str]) -> list[str]:
    """
    Expand the student's typed course list with official course codes.

    WHY THIS IS NEEDED:
        Prerequisite strings in the JSON look like:
            "Introduction to Python Programming (BPLCK105B/205B)"
        If the student typed "Intro to Python", a direct equality check fails.
        By adding the course code to the completed list, all future checks
        become simple substring matches that always work.

    TWO-PHASE APPROACH:
        Phase 1 — Python: fast, free, handles exact matches and copy-pastes.
        Phase 2 — LLM:   handles abbreviations, typos, informal names.
                          Only called for entries Python couldn't resolve.
                          This keeps API usage minimal and sessions fast.

    SCALABILITY:
        Adding new departments or courses requires only updating courses.json.
        No abbreviation dictionaries, no stopword lists, no hardcoded rules.
    """
    enriched  = list(completed_names)   # start with what the student typed
    unmatched = []                       # entries Python couldn't resolve

    # ── Phase 1: Python exact / substring matching ────────────────────────────
    for name in completed_names:
        name_upper = name.upper().strip()
        found      = False

        for course in ALL_COURSES:
            course_code = course['course_code'].upper()
            course_name = course['name'].upper()

            # Match 1: Student typed the exact course code e.g. "BCS301"
            if name_upper == course_code:
                if course_code not in enriched:
                    enriched.append(course_code)
                print(f"[Python ✓ Exact Code]  '{name}' → {course_code}")
                found = True
                break

            # Match 2: Student typed the exact course name (copy-pasted)
            if name_upper == course_name:
                if course_code not in enriched:
                    enriched.append(course_code)
                print(f"[Python ✓ Exact Name]  '{name}' → {course_code}")
                found = True
                break

            # Match 3: Student's input is a clear substring of the course name.
            # Length guard (>6) prevents short words like "LAB" matching everything.
            if len(name_upper) > 6 and name_upper in course_name:
                if course_code not in enriched:
                    enriched.append(course_code)
                print(f"[Python ✓ Substring]   '{name}' → {course_code}")
                found = True
                break
            
            # Match 4: Student typed a partial code that appears inside the full code
            # e.g. "BPLCK105B" is contained in "BPLCK105B/205B"
            if len(name_upper) >= 6 and name_upper in course_code:
                if course_code not in enriched:
                    enriched.append(course_code)
                print(f"[Python ✓ Partial Code] '{name}' → {course_code}")
                found = True
                break


            # Match 5: Full code contains the student's input as the primary part
            # e.g. "BETCK105H/205H" should match if student types "BETCK105H"
            if '/' in course_code and name_upper == course_code.split('/')[0]:
                if course_code not in enriched:
                    enriched.append(course_code)
                print(f"[Python ✓ Split Code]  '{name}' → {course_code}")
                found = True
                break
            
            
        if not found:
            unmatched.append(name)   # hand off to LLM

    # ── Phase 2: LLM semantic matching for unresolved entries ─────────────────
    if unmatched:
        print(f"\n[LLM Enrichment] {len(unmatched)} entries need semantic matching: "
              f"{unmatched}")

        # Compact catalog — only code + name to keep prompt size small.
        # No descriptions or outcomes needed here.
        catalog = [
            {"code": c['course_code'], "name": c['name']}
            for c in ALL_COURSES
        ]

        # Batch if catalog is large (future-proofing for multi-department use)
        BATCH_SIZE  = 50
        all_matches = {}   # student_entry → matched_code

        catalog_batches = [
            catalog[i : i + BATCH_SIZE]
            for i in range(0, len(catalog), BATCH_SIZE)
        ]

        for batch_num, batch in enumerate(catalog_batches):
            prompt = f"""
You are a course catalog matcher for a university system.
A student has listed completed courses using informal or abbreviated names.
Match each student entry to the BEST course in the catalog section below.

STUDENT ENTRIES:
{json.dumps(unmatched, indent=2)}

CATALOG SECTION {batch_num + 1} of {len(catalog_batches)}:
{json.dumps(batch, indent=2)}

STRICT RULES:
- Match based on MEANING and SUBJECT, not string similarity alone
- Roman numerals matter critically: "Math-I" is year-1 math, NEVER year-2
- "Intro to Python" means Python programming, NOT C, Java, IoT, or anything else
- Typos are okay to look past: "ot" in "Introduction ot Python" = "to"
- If not confident, return null — a wrong match is worse than no match
- Return ONLY a JSON array, no explanation, no markdown fences

OUTPUT:
[
  {{"student_entry": "...", "matched_code": "COURSE_CODE_OR_null"}}
]
"""
            try:
                response = llm.invoke([("user", prompt)])
                clean    = re.sub(r'```json|```', '', response.content).strip()
                matches  = json.loads(clean)

                for match in matches:
                    entry = match.get('student_entry', '')
                    code  = match.get('matched_code')
                    # First confident match per entry wins across batches
                    if code and entry not in all_matches:
                        all_matches[entry] = code.upper()

            except json.JSONDecodeError:
                print(f"[LLM Enrichment] Batch {batch_num+1}: Could not parse response.")
            except Exception as e:
                print(f"[LLM Enrichment] Batch {batch_num+1} failed: {e}")

        # Apply matched codes to enriched list
        for entry, code in all_matches.items():
            if code not in enriched:
                enriched.append(code)
            print(f"[LLM  ✓ Matched]   '{entry}' → {code}")

        # Report anything that even the LLM couldn't match
        llm_unmatched = [e for e in unmatched if e not in all_matches]
        for entry in llm_unmatched:
            print(f"[No Match]         '{entry}' — stored as typed, "
                  f"will still be used for substring checks")
            
        # In enrich_completed_list, after the LLM no-match case:
        llm_unmatched = [e for e in unmatched if e not in all_matches]
        for entry in llm_unmatched:
            print(f"[⚠️  Warning] '{entry}' could not be matched to any course.")
            print(f"             Please check the spelling. If this was a completed")
            print(f"             course, type 'forgot' to re-enter it correctly.")

    else:
        print("[Enrichment] All entries matched by Python. LLM call skipped.")

    return list(set(enriched))


# =============================================================================
# STAGE C — FAISS Retrieval
# =============================================================================

def build_retrieval_query(user_query: str, career_goal: str,
                          career_keywords: list[str]) -> str:
    """
    Enrich the raw user query with career context before hitting FAISS.

    WHY: A vague query like "what to study next?" has no career signal.
    FAISS would return random courses. Prepending the career keywords
    steers the embedding vector toward career-relevant content.
    """
    if career_keywords:
        kw_string = ", ".join(career_keywords[:4])
        return f"{career_goal} courses: {kw_string}. {user_query}"
    return f"{career_goal}. {user_query}"


def get_foundation_sweep_docs() -> list:
    """
    Always retrieve foundational courses regardless of user query.

    WHY THIS IS NEEDED:
        BCS301 (Math for CS, no prerequisites) is critical for AI but
        a vague query like "what to study next?" may never retrieve it.
        Multiple targeted sweeps guarantee it always enters the pipeline.

    Three sweeps with different angles:
        1. Core AI/ML vocabulary
        2. Mathematics and statistics foundation
        3. Programming and data fundamentals
    """
    sweep_queries = [
        "machine learning deep learning python algorithms statistics artificial intelligence",
        "mathematics statistics probability linear algebra numerical methods",
        "programming data structures python data analysis foundation courses"
    ]

    seen_ids = set()
    all_docs = []

    for query in sweep_queries:
        docs = vector_db.similarity_search(query, k=40)
        for doc in docs:
            cid = doc.metadata.get('course_id', '').upper()
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_docs.append(doc)

    return all_docs


def retrieve_candidate_docs(user_query: str, career_goal: str,
                            career_keywords: list[str]) -> list:
    """
    Merge primary semantic search + multi-angle foundation sweep.
    Deduplicates by course_id so each course appears exactly once.

    Returns a flat list of unique LangChain Documents for filtering.
    """
    # Primary: career-enriched semantic search
    enriched_query = build_retrieval_query(user_query, career_goal, career_keywords)
    primary_docs   = vector_db.similarity_search(enriched_query, k=30)

    # Foundation: always pull in key foundational courses
    foundation_docs = get_foundation_sweep_docs()

    # Merge with deduplication
    seen_ids = set()
    all_docs = []

    for doc in primary_docs + foundation_docs:
        cid = doc.metadata.get('course_id', '').upper()
        if cid not in seen_ids:
            seen_ids.add(cid)
            all_docs.append(doc)

    return all_docs


# =============================================================================
# STAGE D — Python-Side Filtering
# =============================================================================

def is_course_satisfied(completed_upper: list[str], prereq_string: str) -> bool:
    """
    Check whether a prerequisite string is satisfied by the completed list.

    Three strategies in order (short-circuit on first match):
        1. Substring  — "PYTHON PROGRAMMING" inside "INTRO TO PYTHON (BPLCK...)"
        2. Code match — extract codes like "BPLCK105B" from brackets and compare
        3. Reverse    — check if the prereq string contains a completed item

    No fuzzy matching here — we want prerequisite checks to be deterministic.
    The enrichment stage (Stage B) ensures we have enough aliases in
    completed_upper that these exact checks always work.
    """
    prereq_upper = prereq_string.upper()

    for completed in completed_upper:
        # Strategy 1: direct substring in either direction
        if completed in prereq_upper or prereq_upper in completed:
            return True

        # Strategy 2: extract course codes from brackets e.g. "(BPLCK105B/205B)"
        codes_in_prereq = re.findall(r'[A-Z]{2,6}\d{3}[A-Z0-9/]*', prereq_upper)
        for code in codes_in_prereq:
            if completed in code or code in completed:
                return True

    return False


def is_career_relevant(course: dict, career_keywords: list[str]) -> bool:
    """
    Check if a course is relevant to the student's career goal.

    Builds one searchable string from name + outcomes, then tests each keyword.
    For multi-word keywords like "machine learning", also checks individual
    significant words (length > 4) to handle variants like "learning" or "machine".

    If career_keywords is empty (unknown career), all courses pass through
    and the LLM applies its own judgment.
    """
    if not career_keywords:
        return True

    searchable = (
        course['course_name'].lower() + " " +
        course['outcomes'].lower()
    )
    
    matched_keywords=[]

    for kw in career_keywords:
        kw_lower = kw.lower()
        kw_words=kw_lower.split()
        
        if len(kw_words) ==1 :
            # Single word keyword : must appear as a whole word not substring 
            pattern=r'\b'+re.escape(kw_lower) +r'\b'
            matches= re.findall(pattern,searchable)
            # Require it appears at least twice :signals the course 
            #it is actually about this topic ,not mentioning it 
            if len(matches)>=2:
                matched_keywords.append(kw)
                ###############
                print(f"[Relevance Detail] '{kw}' found {len(matches)}x"
                      f"in {course['course_id']}")
        
        else:
            # Multi-word keyword e.g. "machine learning"
            # Check full phrase first
            if kw_lower in searchable:
                matched_keywords.append(kw)
            # Then check significant individual words (len > 5 to be stricter)
            elif any(word in searchable for word in kw_words if len(word) > 5):
                matched_keywords.append(kw)

    return len(matched_keywords) > 0
       

def deduplicate_pool(eligible_pool: list[dict]) -> list[dict]:
    """
    Remove courses with identical names from the eligible pool.

    WHY: BCS405A and BCS405B are both "Discrete Mathematical Structures".
    Both would appear in recommendations without this step, confusing the student.
    Keeps the first occurrence, discards all subsequent duplicates.
    """
    seen_names = set()
    unique     = []

    for course in eligible_pool:
        name_key = course['course_name'].upper().strip()
        if name_key not in seen_names:
            seen_names.add(name_key)
            unique.append(course)
        else:
            print(f"[Dedupe] Removed duplicate: "
                  f"{course['course_id']} — {course['course_name']}")

    return unique


def filter_candidates(docs: list, completed_upper: list[str],
                      career_keywords: list[str],
                      credit_limit: int) -> tuple[list, list]:
    """
    Master filtering function. Runs every retrieved document through
    five sequential checks. A course must pass ALL checks to be recommended.

    Check order (fail-fast — earlier checks are cheaper):
        1. Already completed   → skip entirely (no point showing done courses)
        2. Prerequisites met   → move to excluded with reason if not met
        3. 0-credit filter     → drop labs and activity courses
        4. Career relevance    → drop courses unrelated to career goal
        5. Credit cap          → stop adding once semester limit is reached
        6. Deduplication       → remove same-name courses with different codes

    Returns:
        eligible_pool — list of dicts ready for the LLM
        excluded      — list of dicts showing blocked courses (for roadmap hints)
    """
    eligible_pool  = []
    excluded       = []
    credits_so_far = 0
    
    #########################3
    watch=['BCS301','BCS358A','BCS302','BCS2303']
    
    # Short docs to process low prerequisite courses first _______________________________________
    docs_sorted=sorted(
        docs,
        key=lambda d: (
            len(d.metadata.get('prerequisites',[])), #fewer pre-req first
            d.metadata.get('credits',0)   #smaller credits next
        )
    )
    for doc in docs_sorted:
        course_id   = doc.metadata.get('course_id', '').upper()
        course_name = doc.metadata.get('name', '').upper()
        prereqs     = doc.metadata.get('prerequisites', [])
        credits     = doc.metadata.get('credits', 0)
        outcomes    = doc.metadata.get('outcomes', 'No outcomes listed.')
        
        

        # ── Check 1: Already completed? ───────────────────────────────────────
        identity = f"{course_id} {course_name}"
        if is_course_satisfied(completed_upper, identity):
            continue   # student already did this course

        # ── Check 2: Prerequisites satisfied? ─────────────────────────────────
        missing = [
            p for p in prereqs
            if not is_course_satisfied(completed_upper, p)
        ]
        if missing:
            excluded.append({
                "course_id"  : course_id,
                "course_name": doc.metadata.get('name', 'Unknown'),
                "reason"     : f"Missing: {', '.join(missing)}"
            })
            continue

        # ── Check 3: Drop 0-credit courses ────────────────────────────────────
        # Labs and activity courses (0 credits) should not appear in main advice.
        if credits == 0:
            continue

        # ── Check 4: Career relevant? ─────────────────────────────────────────
        candidate = {
            "course_id"  : course_id,
            "course_name": doc.metadata.get('name', 'Unknown'),
            "credits"    : credits,
            "outcomes"   : outcomes
        }
        if not is_career_relevant(candidate, career_keywords):
            continue
        
        # Temporary debug — add inside filter_candidates after Check 4
        print(f"[Relevance Check] {course_id}: {is_career_relevant(candidate, career_keywords)}")

        # ── Check 5: Credit cap ────────────────────────────────────────────────
        # Don't recommend more credits than the student can take this semester.
        if credits_so_far + credits > credit_limit:
            continue
        credits_so_far += credits

        eligible_pool.append(candidate)

    # ── Check 6: Remove same-name duplicates (different codes same content) ───
    eligible_pool = deduplicate_pool(eligible_pool)

    return eligible_pool, excluded


# =============================================================================
# STAGE E — LLM Roadmap Generation
# =============================================================================

def build_llm_response(eligible_pool: list, excluded: list,
                       career_goal: str,
                       career_keywords: list[str]) -> str:
    """
    Send the filtered pools to the LLM and get a structured JSON roadmap.

    The prompt enforces:
        - Strict JSON schema (no improvised keys)
        - Maximum 3 courses in enroll_now (most impactful first)
        - 3-hop roadmap chain showing the path to the career goal
        - Graceful empty-pool handling (roadmap-only mode when blocked)
        - No 0-credit or off-topic courses

    The LLM sees only the filtered pool — all junk has been removed by Python.
    Its job is purely to rank, explain, and build the narrative roadmap.
    """
    system_prompt = f"""
You are a Senior Academic Mentor at a university.
Student Career Goal: {career_goal}
Relevant Skill Areas for this career: {career_keywords if career_keywords
                                        else "Use your knowledge to infer."}

Analyze the eligible and blocked course pools and return a precise academic roadmap.

Return ONLY valid JSON. No markdown. No text outside the JSON. Use EXACTLY this schema:
{{
  "message": "One encouraging sentence tailored to the student's current progress.",
  "enroll_now": [
    {{
      "course_id"  : "...",
      "course_name": "...",
      "credits"    : 0,
      "why"        : "One line: why this matters for {career_goal}"
    }}
  ],
  "unlock_next": [
    {{
      "complete_first"      : "Course name or 'CourseA + CourseB'",
      "this_will_unlock"    : "Next course name (code)",
      "which_then_unlocks"  : "The course after that — show the chain toward {career_goal}"
    }}
  ]
}}

HARD RULES:
- enroll_now  : maximum 3 courses, most impactful for {career_goal} listed first
- unlock_next : minimum 2 hops, show the full chain toward the final career goal
- Never include 0-credit courses in enroll_now
- Never include courses unrelated to {career_goal}
- If enroll_now is empty (all courses blocked), still populate unlock_next fully
  so the student knows exactly what to do first to unlock their path
"""

    data = {
        "eligible_courses" : eligible_pool,
        "blocked_courses"  : excluded[:5]   # top 5 blocked with reasons
    }

    response = llm.invoke([
        ("system", system_prompt),
        ("user",   f"Build the roadmap from this data: {json.dumps(data)}")
    ])

    return response.content


# =============================================================================
# STAGE F — Display
# =============================================================================

def display_response(raw_response: str) -> None:
    """
    Parse the LLM's JSON response and print it cleanly.

    Strips markdown fences the LLM sometimes adds despite instructions.
    Falls back to raw text if JSON parsing fails so nothing is ever lost.
    """
    clean = re.sub(r'```json|```', '', raw_response).strip()

    try:
        data = json.loads(clean)

        print(f"\n💬  {data.get('message', '')}\n")

        enroll_now = data.get('enroll_now', [])
        if enroll_now:
            print("✅  RECOMMENDED TO ENROLL:")
            for c in enroll_now:
                print(f"    [{c.get('course_id','?')}] "
                      f"{c.get('course_name','?')} "
                      f"({c.get('credits','?')} cr) "
                      f"— {c.get('why','')}")
        else:
            print("⚠️   No eligible courses yet. "
                  "Complete the roadmap steps below to unlock your path.")

        unlock_next = data.get('unlock_next', [])
        if unlock_next:
            print("\n🗺️   YOUR ROADMAP:")
            for step in unlock_next:
                further = step.get('which_then_unlocks', '')
                further_str = f"  →  then: {further}" if further else ""
                print(f"    Finish : {step.get('complete_first', '?')}")
                print(f"    Unlocks: {step.get('this_will_unlock', '?')}"
                      f"{further_str}")
                print()

    except json.JSONDecodeError:
        # LLM returned non-JSON — print raw so no information is lost
        print("\n[Response]\n")
        print(raw_response)


# =============================================================================
# MAIN PIPELINE — Ties all stages together per query
# =============================================================================

def academic_advisor(query: str, completed_courses: list[str],
                     career_goal: str, credit_limit: int) -> None:
    """
    Full single-query pipeline.

    Args:
        query            : Student's question ("what should I study next?")
        completed_courses: Enriched list (names + codes) from Stage B
        career_goal      : Free-text goal ("AI Engineer")
        credit_limit     : Max credits student can take this semester
    """
    # Stage A: career keywords from career.json
    career_keywords = get_career_keywords(career_goal)

    # Stage C: retrieve candidates (semantic + foundation sweep)
    candidate_docs = retrieve_candidate_docs(query, career_goal, career_keywords)

    # Uppercase once — all downstream checks use this
    completed_upper = [c.upper() for c in completed_courses]

    # Stage D: filter through all checks
    eligible_pool, excluded = filter_candidates(
        candidate_docs, completed_upper, career_keywords, credit_limit
    )
    all_ids = [d.metadata.get('course_id') for d in candidate_docs]
    print(f"\n[Debug] Was BCS301 retrieved? {'BCS301' in all_ids}")
    print(f"[Debug] Was BCS358A retrieved? {'BCS358A' in all_ids}")
    print(f"[Debug] Eligible pool: {[c['course_id'] for c in eligible_pool]}")
    print(f"[Debug] Total retrieved: {len(candidate_docs)}")

    # Stage E: LLM builds the roadmap
    raw_response = build_llm_response(
        eligible_pool, excluded, career_goal, career_keywords
    )

    # Stage F: display
    display_response(raw_response)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("       Senior Academic Advisor")
    print("=" * 60 + "\n")

    # ── Student profile ───────────────────────────────────────────────────────
    career_goal = input("Career Goal (e.g., AI Engineer, Web Developer): ").strip()

    completed_input = input("Completed Courses (comma separated): ").strip()
    completed_raw   = [
        c.strip().upper()
        for c in completed_input.split(',')
        if c.strip()
    ]

    # Stage B: enrich typed names with course codes
    print("\nEnriching course history...")
    completed = enrich_completed_list(completed_raw)
    print(f"\nFinal History: {completed}\n")

    # Credit limit with validation
    while True:
        try:
            credit_limit = int(input(f"Credit limit this semester (1–{MAX_CREDITS}): "))
            if 1 <= credit_limit <= MAX_CREDITS:
                break
            print(f"Please enter a number between 1 and {MAX_CREDITS}.")
        except ValueError:
            print("Numbers only please (e.g., 16).")

    # ── Session loop ──────────────────────────────────────────────────────────
    SESSION_TIMEOUT = 600   # 10 minutes of inactivity ends the session
    session_start   = time.time()

    while True:
        if time.time() - session_start > SESSION_TIMEOUT:
            print("\n[Timeout] Session expired after 10 minutes. Goodbye!")
            break

        print("\n" + "-" * 60)
        query = input(
            "\nAsk anything (or 'forgot' to add courses / 'exit' to quit):\n> "
        ).strip()

        # ── Exit ──────────────────────────────────────────────────────────────
        if query.lower() in ["exit", "quit", "bye"]:
            print("\nGood luck on your journey! Goodbye. 🎓")
            break

        # ── Forgot: add completed courses mid-session ─────────────────────────
        if query.lower() == "forgot":
            more = input("Enter courses to add (comma separated): ").strip()
            new_raw = [
                c.strip().upper()
                for c in more.split(',')
                if c.strip()
            ]
            print("\nEnriching new entries...")
            new_enriched = enrich_completed_list(new_raw)
            completed    = list(set(completed + new_enriched))
            print(f"Updated history: {completed}")
            # Reset session timer on activity
            session_start = time.time()
            continue

        # ── Normal query ──────────────────────────────────────────────────────
        if not query:
            print("Please type a question.")
            continue

        try:
            academic_advisor(query, completed, career_goal, credit_limit)
            # Reset timer on successful query
            session_start = time.time()
        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}")
            print("Please try again or type 'exit' to quit.")