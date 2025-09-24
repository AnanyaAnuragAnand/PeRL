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

st.set_page_config(page_title="PeRL: Personalized Research Learning Assistant", page_icon="🧠")

st.title("PeRL: Personalized Research Learning Assistant (Open-Source Version)")
st.write("Paste scientific text or fetch papers from arXiv to get adaptive summaries and quizzes.")

# --- User input ---
user_text = st.text_area("Paste abstract, methods, or text here:")
difficulty = st.selectbox("Select your expertise level:", ["Beginner", "Intermediate", "Expert"])

# --- Initialize session state ---
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "answers" not in st.session_state:
    st.session_state.answers = {}

# --- Load summarization model (cached to speed up reruns) ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", 
                   model="ananyaanuraganand/t5-finetuned-arxiv", tokenizer="ananyaanuraganand/t5-finetuned-arxiv")  # lightweight, fast

summarizer = load_summarizer()

# --- Function to fetch arXiv papers with DOI support ---
def fetch_arxiv_abstracts(query, max_results=5):
    """
    Fetches the latest papers from arXiv based on the query.
    Returns a list of dicts: [{'title': ..., 'abstract': ..., 'doi': ...}, ...]
    """
    query_encoded = urllib.parse.quote(query)  # Encode spaces and special characters
    base_url = "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}"
    feed_url = base_url.format(query_encoded, max_results)
    feed = feedparser.parse(feed_url)
    
    papers = []
    for entry in feed.entries:
        title = entry.title
        abstract = entry.summary.replace('\n', ' ').strip()
        doi = entry.get('arxiv_doi', 'N/A')  # DOI if available
        papers.append({"title": title, "abstract": abstract, "doi": doi})
    
    return papers

# --- Summarize pasted text ---
if st.button("Summarize"):
    if user_text.strip():
        # Set summarization length and prompt based on expertise level
        if difficulty == "Beginner":
            max_len = 60
            min_len = 30
            prompt = "Summarize in simple terms for a beginner: "
        elif difficulty == "Intermediate":
            max_len = 120
            min_len = 50
            prompt = "Summarize concisely with some technical detail: "
        else:  # Expert
            max_len = 200
            min_len = 80
            prompt = "Summarize in detail for an expert: "

        raw_summary = summarizer(prompt + user_text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

        # Capitalize sentences and clean periods
        sentences = sent_tokenize(raw_summary)
        cleaned_sentences = [s.strip().capitalize().rstrip(' .') + '.' for s in sentences]
        st.session_state.summary = " ".join(cleaned_sentences)

        # Generate quiz questions
        num_questions = min(3, len(sentences))
        st.session_state.quiz = random.sample(sentences, num_questions)

        # Reset previous answers
        st.session_state.answers = {}

# --- Display summary ---
if st.session_state.summary:
    st.subheader("Summary")
    st.write(st.session_state.summary)

# --- Display quiz ---
if st.session_state.quiz:
    st.subheader("Quiz (Identify the main points)")
    for i, q in enumerate(st.session_state.quiz, 1):
        if f"q{i}" not in st.session_state.answers:
            st.session_state.answers[f"q{i}"] = None
        st.markdown(f"**Q{i}:** {q}")
        st.session_state.answers[f"q{i}"] = st.radio("Do you understand this point?", ["Yes", "Somewhat", "No"], key=f"q{i}")

    # Submit quiz button
    if st.button("Submit Quiz"):
        yes_count = sum(1 for ans in st.session_state.answers.values() if ans == "Yes")
        some_count = sum(1 for ans in st.session_state.answers.values() if ans == "Somewhat")
        no_count = sum(1 for ans in st.session_state.answers.values() if ans == "No")

        st.subheader("Quiz Feedback")
        st.write(f"✅ Understood well: {yes_count}")
        st.write(f"⚠️ Somewhat understood: {some_count}")
        st.write(f"❌ Not understood: {no_count}")

        if no_count > 0:
            st.info("Consider reviewing the parts marked 'Not understood' for better clarity!")

# --- Fetch arXiv papers ---
st.subheader("Or fetch recent papers from arXiv")
arxiv_query = st.text_input("Enter a topic or keyword to search papers:")

if st.button("Fetch Papers"):
    if arxiv_query.strip():
        with st.spinner("Fetching papers from arXiv..."):
            papers = fetch_arxiv_abstracts(arxiv_query, max_results=5)
        
        if papers:
            for i, paper in enumerate(papers, 1):
                st.markdown(f"**Paper {i}: {paper['title']}**")
                st.write(paper['abstract'])
                st.markdown(f"**DOI:** {paper['doi']}")  # Display DOI

                # Automatic summary
                max_len = 100
                raw_summary = summarizer(paper['abstract'], max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
                sentences = sent_tokenize(raw_summary)
                cleaned_sentences = [s.strip().capitalize().rstrip(' .') + '.' for s in sentences]
                summary = " ".join(cleaned_sentences)
                
                st.markdown(f"**Summary:** {summary}")

