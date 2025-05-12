import streamlit as st
import fitz  # PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# ---------- OpenAI Config ----------
openai.api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else "YOUR_API_KEY"

# ---------- Skill List ----------
SKILL_LIST = [
    "python", "sql", "flask", "docker", "git", "pandas", "machine learning",
    "tensorflow", "nlp", "data visualization", "streamlit", "aws", "api", "linux"
]

nlp = spacy.load("en_core_web_sm")

# ---------- PDF Text Extraction ----------
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = " ".join(page.get_text() for page in doc)
    return text

# ---------- NLP Preprocessing ----------
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# ---------- Skill Extraction ----------
def extract_skills(text):
    doc = nlp(text.lower())
    tokens = set([token.text for token in doc if not token.is_stop and not token.is_punct])
    skills_found = [skill for skill in SKILL_LIST if any(word in tokens for word in skill.lower().split())]
    return sorted(set(skills_found))

# ---------- OpenAI Smart Suggestion ----------
def generate_gpt_suggestion(skill, job_desc):
    prompt = f"""
You're helping improve a technical resume. The job description includes:
{job_desc}

Suggest a professional bullet point about the candidate's experience with "{skill}" that they might add to their resume.
It should be specific, technical, and start with a strong action verb.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Could not generate suggestion for {skill}: {str(e)}"

# ---------- Streamlit UI ----------
st.title("📄 Resume Gap Analyzer (with GPT Suggestions)")
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_text_input = st.text_area("Paste Job Description")

if resume_file and job_text_input:
    resume_text = extract_text_from_pdf(resume_file)
    processed_resume = preprocess(resume_text)
    processed_job = preprocess(job_text_input)

    # TF-IDF Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_job])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Skill Gap
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text_input)
    missing_skills = sorted(set(job_skills) - set(resume_skills))

    # Output
    st.subheader("🔎 Match Score")
    st.metric(label="Resume vs Job Match", value=f"{round(similarity_score * 100, 1)}%")

    st.subheader("📌 Missing Skills")
    if missing_skills:
        for skill in missing_skills:
            st.write("❌", skill)
    else:
        st.success("No significant skill gaps found!")

    st.subheader("🧠 Suggested Resume Additions (Powered by OpenAI)")
    for skill in missing_skills:
        suggestion = generate_gpt_suggestion(skill, job_text_input)
        st.markdown(f"**{skill.title()}**: {suggestion}")