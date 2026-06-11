# How to Add Your College's Course Data

This folder contains the course data that powers the advisor.
The sample data is from VTU (India). You can add your own college's data by following this format.

---

## courses.json format

Each subject should look like this:

```json
{
  "course_code": "CS101",
  "course_name": "Introduction to Programming",
  "semester": 1,
  "credits": 4,
  "topics": [
    "Variables and Data Types",
    "Loops and Conditionals",
    "Functions",
    "Arrays"
  ],
  "outcomes": [
    "Understand basic programming concepts",
    "Write simple programs in Python or C"
  ],
  "prerequisites": []
}
```

Your full `courses.json` is a list of these objects:

```json
[
  { first course here },
  { second course here },
  ...
]
```

---

## career.json format

Each career option should look like this:

```json
{
  "career": "Machine Learning Engineer",
  "description": "Builds and deploys ML models for real-world applications",
  "required_skills": [
    "Linear Algebra",
    "Python",
    "Machine Learning",
    "Deep Learning"
  ],
  "recommended_courses": ["CS301", "CS401", "CS402"]
}
```

---

## How to contribute your college's data

1. Fork this repository
2. Create a new folder inside `Data/` named after your college — example: `Data/MIT/` or `Data/IIT_Bombay/`
3. Add your `courses.json` and `career.json` inside that folder
4. Open a Pull Request with the title: `Add course data for [Your College Name]`

That's it. No code changes needed. Just data.

---

## Questions?

Open an issue and ask. All experience levels welcome.