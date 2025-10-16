
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


st.write("Paste scientific text to achieve adaptive summaries, quizzes and important literature.")

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
def generate_mcq_quiz(user_text, level="Beginner"):
    sentences = nltk.sent_tokenize(user_text)
    quiz = []

    # Determine number of questions
    num_questions = min(5, len(sentences))

    for _ in range(num_questions):
        sent = random.choice(sentences)

        # QG prompt based on level
        if level == "Beginner":
            q_prompt = f"Generate question based on: {sent}"
        elif level == "Intermediate":
            q_prompt = f"Generate question based on: {sent}"
        else:
            q_prompt = f"Generate question based on: {sent}"

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

# # --- Fetch similar research papers using Semantic Scholar ---
# import requests

# def fetch_semantic_scholar(query, max_results=5):
#     """
#     Fetch similar papers from Semantic Scholar API based on query text.
#     Returns a list of dicts with title, authors, abstract, and url.
#     """
#     base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
#     params = {
#         "query": query,
#         "limit": max_results,
#         "fields": "title,authors,abstract,url"
#     }
#     try:
#         response = requests.get(base_url, params=params)
#         response.raise_for_status()
#         data = response.json()
#         papers = []
#         for item in data.get("data", []):
#             paper = {
#                 "title": item.get("title", "No title"),
#                 "authors": ", ".join([a.get("name", "") for a in item.get("authors", [])]),
#                 "abstract": item.get("abstract", "No abstract available"),
#                 "url": item.get("url", "#")
#             }
#             papers.append(paper)
#         return papers
#     except Exception as e:
#         st.error(f"Error fetching papers: {e}")
#         return []

# # --- User input for fetching similar papers ---
# fetch_query = st.text_input("Enter topic or text to fetch similar papers:")

# if st.button("Fetch Similar Papers"):
#     if fetch_query.strip():
#         st.session_state.fetch_clicked = True
#         fetched_papers = fetch_semantic_scholar(fetch_query)
#         if fetched_papers:
#             st.subheader("Similar Papers")
#             for i, paper in enumerate(fetched_papers, 1):
#                 st.markdown(f"**{i}. {paper['title']}**")
#                 st.markdown(f"*Authors:* {paper['authors']}")
#                 st.markdown(f"*Abstract:* {paper['abstract']}")
#                 st.markdown(f"[View Paper]({paper['url']})")
#         else:
#             st.info("No papers found for this query.")

##################
#########
# --- Install KeyBERT if not already installed ---
# pip install keybert sentence-transformers

from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

# --- Initialize KeyBERT ---
kw_model = KeyBERT()

# def extract_keywords(text, num_keywords=5):
#     """
#     Extracts the top scientific keywords from user input text using KeyBERT.
#     """
#     if not text.strip():
#         return []
#     keywords = kw_model.extract_keywords(
#         text, 
#         keyphrase_ngram_range=(1, 2), 
#         stop_words='english', 
#         top_n=num_keywords
#     )
#     # Extract just the keyword strings
#     return [kw[0] for kw in keywords]
import re
from sklearn.feature_extraction.text import CountVectorizer

def extract_keywords_simple(text, num_keywords=5):
    """
    A lightweight keyword extractor that picks the most frequent scientific terms.
    Useful as a fallback if KeyBERT is unavailable or to avoid heavy computation.
    """
    if not text or not text.strip():
        return []

    # Basic cleanup
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text.lower())

    # Use CountVectorizer to extract top frequent words and 2-word phrases
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english').fit([cleaned_text])
    bag_of_words = vectorizer.transform([cleaned_text])

    # Get frequencies
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True)

    # Take top N keywords (remove duplicates and extra spaces)
    keywords = []
    for word, freq in sorted_words:
        if word not in keywords:
            keywords.append(word.strip())
        if len(keywords) >= num_keywords:
            break

    return keywords

# def fetch_papers_by_keywords_better(text, num_keywords=5, max_per_keyword=3):
#     """
#     Extract keywords using KeyBERT, query Semantic Scholar for each keyword separately,
#     and combine the results to display.
#     """
#     keywords = extract_keywords(text, num_keywords=num_keywords)
#     if not keywords:
#         return [], []

#     all_papers = []
#     seen_titles = set()

#     for kw in keywords:
#         papers = fetch_semantic_scholar(kw, max_results=max_per_keyword)
#         for p in papers:
#             if p['title'] not in seen_titles:
#                 all_papers.append(p)
#                 seen_titles.add(p['title'])

#     return keywords, all_papers

import requests
import time
import logging
import streamlit as st

def fetch_papers_by_keywords_better(text, num_keywords=5, limit=3, delay=1):
    """
    Fetch papers from Semantic Scholar using multiple keywords.
    Handles 429 errors gracefully and shows a single friendly warning.
    """
    # --- Extract keywords ---
    keywords = extract_keywords_simple(text, num_keywords=num_keywords)

    all_papers = []
    rate_limit_hit = False  # to warn user only once

    for keyword in keywords:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&limit={limit}&fields=title,authors,abstract,url"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            papers = data.get("data", [])
            all_papers.extend(papers)

        except requests.exceptions.HTTPError as e:
            # Handle 429 rate limit gracefully
            if response.status_code == 429:
                logging.warning(f"[Rate limit] Keyword '{keyword}' was rate-limited.")
                if not rate_limit_hit:
                    st.warning("Some searches were rate-limited by Semantic Scholar. Please wait a moment or try again later.")
                    rate_limit_hit = True
            else:
                logging.error(f"HTTP error for '{keyword}': {e}")

        except Exception as e:
            logging.error(f"Unexpected error for '{keyword}': {e}")

        # Add a small delay to avoid rapid-fire requests (important for free APIs)
        time.sleep(delay)

    return keywords, all_papers


# --- Streamlit section ---
if st.button("Fetch Papers Based on Keywords"):
    if user_text.strip():
        keywords, papers = fetch_papers_by_keywords_better(user_text, num_keywords=5)
        if keywords:
            st.subheader("Extracted Keywords")
            st.write(", ".join(keywords))
        # if papers:
        #     st.subheader("Papers Based on Keywords")
        #     for i, paper in enumerate(papers, 1):
        #         st.markdown(f"**{i}. {paper['title']}**")
        #         st.markdown(f"*Authors:* {paper['authors']}")
        #         st.markdown(f"*Abstract:* {paper['abstract']}")
        #         st.markdown(f"[View Paper]({paper['url']})")
        # else:
        #     st.info("No papers found for the extracted keywords.")
        if papers:
            st.subheader("Papers Based on Keywords")
            for i, paper in enumerate(papers, 1):
                title = paper.get("title", "No title")
                authors = paper.get("authors", [])
                abstract = paper.get("abstract", None)
                url = paper.get("url", "#")
        
                # Format author names as a clean comma-separated string
                author_names = ", ".join([a.get("name", "") for a in authors])
        
                st.markdown(f"**{i}. {title}**")
                if author_names:
                    st.markdown(f"*Authors:* {author_names}")
                else:
                    st.markdown("*Authors:* Not available")
        
                if abstract:
                    st.markdown(f"*Abstract:* {abstract}")
                else:
                    st.markdown("*Abstract:* Not available")
        
                st.markdown(f"[View Paper]({url})")
                st.markdown("---")
        else:
            st.info("No papers found for the extracted keywords.")

