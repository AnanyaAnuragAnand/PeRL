
# --- Import libraries ---
import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import random
import feedparser
import urllib.parse

# --- Download necessary NLTK resources ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --- Page configuration ---
st.set_page_config(
    page_title="PeRL: Personalized Research Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for background and title ---
st.markdown(
    """
    <style>
    /* Change background color */
    .stApp {
        background-color: #FFF9C4;  /* Light yellow */
    }

    /* Style the main title */
    .custom-title {
        font-size: 40px;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
    }

    /* Color letters individually */
    .pe { color: #FF1493; }   /* Deep Pink */
    .r { color: #FF1493; }    /* Deep Pink */
    .l { color: #FF1493; }    /* Deep Pink */
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div class='custom-title'>"
    "<span class='pe'>Pe</span><span class='r'>R</span><span class='l'>L</span>: "
    "<span class='pe'>Pe</span>rsonalized "
    "<span class='r'>R</span>esearch "
    "<span class='l'>L</span>earning Assistant"
    "</div>",
    unsafe_allow_html=True
)


st.write("Paste scientific text or fetch papers from arXiv to get adaptive summaries and quizzes.")

# --- User input ---
user_text = st.text_area("Paste abstract, methods, or text here:")
difficulty = st.selectbox("Select your expertise level:", ["Beginner", "Intermediate", "Expert"])

# --- Initialize session state ---
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "fetch_clicked" not in st.session_state:
    st.session_state.fetch_clicked = False

# --- Load summarization model ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", 
                   model="ananyaanuraganand/t5-finetuned-arxiv", 
                   tokenizer="ananyaanuraganand/t5-finetuned-arxiv")

summarizer = load_summarizer()

# --- Load question-generation pipeline ---
@st.cache_resource
def load_qg_pipeline():
    # Using T5 small question generation model
    return pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")

qg_pipeline = load_qg_pipeline()

# --- Generate MCQ quiz from summary ---
def generate_mcq_quiz(summary_text, level="Beginner"):
    sentences = nltk.sent_tokenize(summary_text)
    quiz = []

    # Determine number of questions
    num_questions = min(5, len(sentences))

    for _ in range(num_questions):
        sent = random.choice(sentences)

        # QG prompt based on level
        if level == "Beginner":
            q_prompt = f"Generate a simple factual question for a beginner based on: {sent}"
        elif level == "Intermediate":
            q_prompt = f"Generate a multiple choice question with some technical detail based on: {sent}"
        else:
            q_prompt = f"Generate a challenging conceptual question for experts based on: {sent}"

        question_text = qg_pipeline(q_prompt, max_length=64, num_return_sequences=1)[0]['generated_text']

        # Correct answer: take the main sentence itself (simplified)
        correct_answer = sent

        # Distractors: pick other sentences
        distractors = [s for s in sentences if s != correct_answer]
        distractors = random.sample(distractors, min(3, len(distractors)))

        options = distractors + [correct_answer]
        random.shuffle(options)

        quiz.append({
            "question": question_text,
            "options": options,
            "answer": correct_answer
        })
    return quiz

# --- Summarize pasted text ---
if st.button("Summarize"):
    if user_text.strip():
        # Summarization length based on expertise
        if difficulty == "Beginner":
            max_len, min_len = 60, 30
            prompt = "Summarize in simple terms for a beginner: "
        elif difficulty == "Intermediate":
            max_len, min_len = 120, 50
            prompt = "Summarize concisely with some technical detail: "
        else:
            max_len, min_len = 200, 80
            prompt = "Summarize in detail for an expert: "

        raw_summary = summarizer(prompt + user_text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

        # Clean sentences
        sentences = sent_tokenize(raw_summary)
        cleaned_sentences = [s.strip().capitalize().rstrip(' .') + '.' for s in sentences]
        summary = " ".join(cleaned_sentences)
        st.session_state.summary = summary

        # Generate MCQs
        st.session_state.quiz = generate_mcq_quiz(summary, level=difficulty)

# --- Display summary and quiz ---
if st.session_state.summary:
    st.subheader("Summary")
    st.write(st.session_state.summary)

if st.session_state.quiz:
    st.subheader("Quiz")
    for i, q in enumerate(st.session_state.quiz, 1):
        st.markdown(f"**Q{i}: {q['question']}**")
        selected = st.radio(f"Select your answer:", q["options"], key=f"quiz_{i}")
        if st.button(f"Show Answer {i}", key=f"ans_{i}"):
            st.success(f"✅ Correct answer: {q['answer']}")

# --- Related Papers / Suggestions Section ---
st.subheader("Related Paper Suggestions")

# Button to generate suggestions based on current summary
if st.button("Generate Related Paper Suggestions"):
    if st.session_state.summary:
        with st.spinner("Generating related paper suggestions..."):
            # Function to generate suggestions from your finetuned T5 model
            def generate_related_papers(summary_text, max_results=5):
                prompt = f"Suggest {max_results} related paper titles or research keywords based on: {summary_text}"
                generated = summarizer(prompt, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                # Split suggestions by line or comma
                suggestions = [s.strip() for s in generated.replace("\n", ",").split(",") if s.strip()]
                return suggestions[:max_results]

            related_suggestions = generate_related_papers(st.session_state.summary, max_results=5)

        st.write("Suggested related topics or papers:")
        for s in related_suggestions:
            st.write(f"- {s}")

        # Optional: fetch actual papers from arXiv
        if st.button("Fetch Related Papers from arXiv"):
            with st.spinner("Fetching related papers from arXiv..."):
                def fetch_arxiv_for_suggestions(suggestions, max_results_per_suggestion=3):
                    all_papers = []
                    for term in suggestions:
                        query_encoded = urllib.parse.quote(term)
                        feed_url = f"http://export.arxiv.org/api/query?search_query=all:{query_encoded}&start=0&max_results={max_results_per_suggestion}"
                        feed = feedparser.parse(feed_url)
                        for entry in feed.entries:
                            title = entry.title
                            abstract = entry.summary.replace('\n', ' ').strip()
                            doi = entry.get('arxiv_doi', 'N/A')
                            all_papers.append({"title": title, "abstract": abstract, "doi": doi, "term": term})
                    return all_papers

                related_papers = fetch_arxiv_for_suggestions(related_suggestions, max_results_per_suggestion=2)

            st.write("Fetched papers from arXiv based on suggestions:")
            for i, paper in enumerate(related_papers, 1):
                st.markdown(f"**Paper {i} (from suggestion: {paper['term']}): {paper['title']}**")
                st.write(paper['abstract'])
                st.markdown(f"**DOI:** {paper['doi']}")
    else:
        st.warning("Please summarize some text first to generate suggestions.")
