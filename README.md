# Universal Academic Advisor 🎓

An AI-powered academic advisor that works for **any college**.
Tell it your career goal and completed courses — it tells you exactly what to study next.

Built with RAG (Retrieval-Augmented Generation) using LangChain, FAISS, and LLaMA 3.3 70B.

---

## What it does

- You tell it: your career goal + courses you've completed + credit limit
- It tells you: exactly which courses to take next semester and why
- It shows you: a full roadmap from where you are to your career goal
- It never recommends a course you're not eligible for

---

## Works for any college

The sample data inside `Data/sample_courses.json` is from VTU (India).
But you can use this for **your own college** by adding your course data.
See [how to add your college's data](Data/DATA_FORMAT.md).

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | RAG pipeline |
| FAISS | Vector search |
| HuggingFace `all-MiniLM-L6-v2` | Embeddings |
| Groq + LLaMA 3.3 70B | AI reasoning |
| Streamlit | Web UI |

## How it works

```
User Input
│
├── Career Goal (e.g. "I want to be an ML Engineer")
├── Completed Courses (e.g. CS101, CS201)
└── Credit Limit (e.g. 15 credits)
│
▼
RAG Pipeline
│
├── courses.json → FAISS Vector Database
│   └── Embeds all course topics and outcomes
│
├── career.json → Career Requirements Lookup
│   └── Finds required skills for your goal
│
▼
LLaMA 3.3 70B (via Groq)
│
└── Matches your completed courses + career goal
    └── Recommends exactly what to study next
│
▼
Output
│
├── Recommended courses for next semester
├── Why each course is recommended
└── Full roadmap to your career goal
```

---

## How to run it

**1. Clone the repo**
```bash
git clone https://github.com/BiswasNehaa/universal-academic-advisor.git
cd universal-academic-advisor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file and add:
GROQ_API_KEY=your_key_here



Get a free key at [console.groq.com](https://console.groq.com)

**4. Build the vector database**
```bash
python app/ingest.py
```

**5. Run the app**
```bash
streamlit run app/app.py
```

---

## Want to contribute?

The easiest way to contribute is to **add your college's course data**.

See the [contribution guide](CONTRIBUTING.md) to get started.
All experience levels welcome.

---

## Made by

[Nehaa Biswas](https://github.com/BiswasNehaa) — 3rd year CS student