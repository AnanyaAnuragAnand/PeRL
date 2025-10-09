# code 1:
# import streamlit as st
# from transformers import pipeline
# import nltk
# from nltk.tokenize import sent_tokenize
# import random
# import feedparser
# import urllib.parse

# # --- Download necessary NLTK resources ---
# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)

# st.set_page_config(page_title="PeRL: Personalized Research Learning Assistant", page_icon="🧠")

# st.title("PeRL: Personalized Research Learning Assistant (Open-Source Version)")
# st.write("Paste scientific text or fetch papers from arXiv to get adaptive summaries and quizzes.")

# # --- User input ---
# user_text = st.text_area("Paste abstract, methods, or text here:")
# difficulty = st.selectbox("Select your expertise level:", ["Beginner", "Intermediate", "Expert"])

# # --- Initialize session state ---
# if "summary" not in st.session_state:
#     st.session_state.summary = ""
# if "quiz" not in st.session_state:
#     st.session_state.quiz = []
# if "answers" not in st.session_state:
#     st.session_state.answers = {}

# # --- Load summarization model (cached to speed up reruns) ---
# @st.cache_resource
# def load_summarizer():
#     return pipeline("summarization", 
#                    model="ananyaanuraganand/t5-finetuned-arxiv", tokenizer="ananyaanuraganand/t5-finetuned-arxiv")  # lightweight, fast

# summarizer = load_summarizer()

# # --- Function to fetch arXiv papers with DOI support ---
# def fetch_arxiv_abstracts(query, max_results=5):
#     """
#     Fetches the latest papers from arXiv based on the query.
#     Returns a list of dicts: [{'title': ..., 'abstract': ..., 'doi': ...}, ...]
#     """
#     query_encoded = urllib.parse.quote(query)  # Encode spaces and special characters
#     base_url = "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}"
#     feed_url = base_url.format(query_encoded, max_results)
#     feed = feedparser.parse(feed_url)
    
#     papers = []
#     for entry in feed.entries:
#         title = entry.title
#         abstract = entry.summary.replace('\n', ' ').strip()
#         doi = entry.get('arxiv_doi', 'N/A')  # DOI if available
#         papers.append({"title": title, "abstract": abstract, "doi": doi})
    
#     return papers

# # --- Summarize pasted text ---
# if st.button("Summarize"):
#     if user_text.strip():
#         # Set summarization length and prompt based on expertise level
#         if difficulty == "Beginner":
#             max_len = 60
#             min_len = 30
#             prompt = "Summarize in simple terms for a beginner: "
#         elif difficulty == "Intermediate":
#             max_len = 120
#             min_len = 50
#             prompt = "Summarize concisely with some technical detail: "
#         else:  # Expert
#             max_len = 200
#             min_len = 80
#             prompt = "Summarize in detail for an expert: "

#         raw_summary = summarizer(prompt + user_text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

#         # Capitalize sentences and clean periods
#         sentences = sent_tokenize(raw_summary)
#         cleaned_sentences = [s.strip().capitalize().rstrip(' .') + '.' for s in sentences]
#         st.session_state.summary = " ".join(cleaned_sentences)

#         # Generate quiz questions
#         num_questions = min(3, len(sentences))
#         st.session_state.quiz = random.sample(sentences, num_questions)

#         # Reset previous answers
#         st.session_state.answers = {}

# # --- Display summary ---
# if st.session_state.summary:
#     st.subheader("Summary")
#     st.write(st.session_state.summary)

# # --- Display quiz ---
# if st.session_state.quiz:
#     st.subheader("Quiz (Identify the main points)")
#     for i, q in enumerate(st.session_state.quiz, 1):
#         if f"q{i}" not in st.session_state.answers:
#             st.session_state.answers[f"q{i}"] = None
#         st.markdown(f"**Q{i}:** {q}")
#         st.session_state.answers[f"q{i}"] = st.radio("Do you understand this point?", ["Yes", "Somewhat", "No"], key=f"q{i}")

#     # Submit quiz button
#     if st.button("Submit Quiz"):
#         yes_count = sum(1 for ans in st.session_state.answers.values() if ans == "Yes")
#         some_count = sum(1 for ans in st.session_state.answers.values() if ans == "Somewhat")
#         no_count = sum(1 for ans in st.session_state.answers.values() if ans == "No")

#         st.subheader("Quiz Feedback")
#         st.write(f"✅ Understood well: {yes_count}")
#         st.write(f"⚠️ Somewhat understood: {some_count}")
#         st.write(f"❌ Not understood: {no_count}")

#         if no_count > 0:
#             st.info("Consider reviewing the parts marked 'Not understood' for better clarity!")

# # --- Fetch arXiv papers ---
# st.subheader("Or fetch recent papers from arXiv")
# arxiv_query = st.text_input("Enter a topic or keyword to search papers:")

# if st.button("Fetch Papers"):
#     if arxiv_query.strip():
#         with st.spinner("Fetching papers from arXiv..."):
#             papers = fetch_arxiv_abstracts(arxiv_query, max_results=5)
        
#         if papers:
#             for i, paper in enumerate(papers, 1):
#                 st.markdown(f"**Paper {i}: {paper['title']}**")
#                 st.write(paper['abstract'])
#                 st.markdown(f"**DOI:** {paper['doi']}")  # Display DOI

#                 # Automatic summary
#                 max_len = 100
#                 raw_summary = summarizer(paper['abstract'], max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
#                 sentences = sent_tokenize(raw_summary)
#                 cleaned_sentences = [s.strip().capitalize().rstrip(' .') + '.' for s in sentences]
#                 summary = " ".join(cleaned_sentences)
                
#                 st.markdown(f"**Summary:** {summary}")
# CODE 2: 
# import streamlit as st
# from transformers import pipeline
# import nltk
# from nltk.tokenize import sent_tokenize
# import random
# import feedparser
# import urllib.parse

# # --- Download necessary NLTK resources ---
# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)

# st.set_page_config(page_title="PeRL: Personalized Research Learning Assistant", page_icon="🧠")

# st.title("PeRL: Personalized Research Learning Assistant (Open-Source Version)")
# st.write("Paste scientific text or fetch papers from arXiv to get adaptive summaries and quizzes.")

# # --- User input ---
# user_text = st.text_area("Paste abstract, methods, or text here:")
# difficulty = st.selectbox("Select your expertise level:", ["Beginner", "Intermediate", "Expert"])

# # --- Initialize session state ---
# if "summary" not in st.session_state:
#     st.session_state.summary = ""
# if "quiz" not in st.session_state:
#     st.session_state.quiz = []

# # --- Load summarization model ---
# @st.cache_resource
# def load_summarizer():
#     return pipeline("summarization", 
#                    model="ananyaanuraganand/t5-finetuned-arxiv", 
#                    tokenizer="ananyaanuraganand/t5-finetuned-arxiv")

# summarizer = load_summarizer()

# # --- Load question-generation pipeline ---
# @st.cache_resource
# def load_qg_pipeline():
#     # Using T5 small question generation model
#     return pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")

# qg_pipeline = load_qg_pipeline()

# # --- Function to fetch arXiv papers ---
# def fetch_arxiv_abstracts(query, max_results=5):
#     query_encoded = urllib.parse.quote(query)
#     base_url = "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}"
#     feed_url = base_url.format(query_encoded, max_results)
#     feed = feedparser.parse(feed_url)
    
#     papers = []
#     for entry in feed.entries:
#         title = entry.title
#         abstract = entry.summary.replace('\n', ' ').strip()
#         doi = entry.get('arxiv_doi', 'N/A')
#         papers.append({"title": title, "abstract": abstract, "doi": doi})
    
#     return papers

# # --- Generate MCQ quiz from summary ---
# def generate_mcq_quiz(summary_text, level="Beginner"):
#     sentences = nltk.sent_tokenize(summary_text)
#     quiz = []

#     # Determine number of questions
#     num_questions = min(5, len(sentences))

#     for _ in range(num_questions):
#         sent = random.choice(sentences)

#         # QG prompt based on level
#         if level == "Beginner":
#             q_prompt = f"Generate a simple factual question for a beginner based on: {sent}"
#         elif level == "Intermediate":
#             q_prompt = f"Generate a multiple choice question with some technical detail based on: {sent}"
#         else:
#             q_prompt = f"Generate a challenging conceptual question for experts based on: {sent}"

#         question_text = qg_pipeline(q_prompt, max_length=64, num_return_sequences=1)[0]['generated_text']

#         # Correct answer: take the main sentence itself (simplified)
#         correct_answer = sent

#         # Distractors: pick other sentences
#         distractors = [s for s in sentences if s != correct_answer]
#         distractors = random.sample(distractors, min(3, len(distractors)))

#         options = distractors + [correct_answer]
#         random.shuffle(options)

#         quiz.append({
#             "question": question_text,
#             "options": options,
#             "answer": correct_answer
#         })
#     return quiz

# # --- Summarize pasted text ---
# if st.button("Summarize"):
#     if user_text.strip():
#         # Summarization length based on expertise
#         if difficulty == "Beginner":
#             max_len, min_len = 60, 30
#             prompt = "Summarize in simple terms for a beginner: "
#         elif difficulty == "Intermediate":
#             max_len, min_len = 120, 50
#             prompt = "Summarize concisely with some technical detail: "
#         else:
#             max_len, min_len = 200, 80
#             prompt = "Summarize in detail for an expert: "

#         raw_summary = summarizer(prompt + user_text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

#         # Clean sentences
#         sentences = sent_tokenize(raw_summary)
#         cleaned_sentences = [s.strip().capitalize().rstrip(' .') + '.' for s in sentences]
#         summary = " ".join(cleaned_sentences)
#         st.session_state.summary = summary

#         # Generate MCQs
#         st.session_state.quiz = generate_mcq_quiz(summary, level=difficulty)

# # --- Display summary and quiz ---
# if st.session_state.summary:
#     st.subheader("Summary")
#     st.write(st.session_state.summary)

# if st.session_state.quiz:
#     st.subheader("Quiz")
#     for i, q in enumerate(st.session_state.quiz, 1):
#         st.markdown(f"**Q{i}: {q['question']}**")
#         selected = st.radio(f"Select your answer:", q["options"], key=f"quiz_{i}")
#         if st.button(f"Show Answer {i}", key=f"ans_{i}"):
#             st.success(f"✅ Correct answer: {q['answer']}")

# # --- Fetch arXiv papers ---
# st.subheader("Or fetch recent papers from arXiv")
# arxiv_query = st.text_input("Enter a topic or keyword to search papers:")

# if st.button("Fetch Papers"):
#     if arxiv_query.strip():
#         with st.spinner("Fetching papers from arXiv..."):
#             papers = fetch_arxiv_abstracts(arxiv_query, max_results=5)
        
#         if papers:
#             for i, paper in enumerate(papers, 1):
#                 st.markdown(f"**Paper {i}: {paper['title']}**")
#                 st.write(paper['abstract'])
#                 st.markdown(f"**DOI:** {paper['doi']}")
#                 # Auto summary
#                 max_len = 100
#                 raw_summary = summarizer(paper['abstract'], max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
#                 sentences = sent_tokenize(raw_summary)
#                 cleaned_sentences = [s.strip().capitalize().rstrip(' .') + '.' for s in sentences]
#                 summary = " ".join(cleaned_sentences)
#                 st.markdown(f"**Summary:** {summary}")

import streamlit as st
from transformers import pipeline
import re
import random
import feedparser
import urllib.parse
from collections import Counter

# --- Streamlit page config ---
st.set_page_config(page_title="PeRL: Personalized Research Learning Assistant", page_icon="🧠")
st.title("PeRL: Personalized Research Learning Assistant (Open-Source Version)")
st.write("Paste scientific text or fetch papers from arXiv to get adaptive summaries, quizzes, and recommendations.")

# --- User input ---
user_text = st.text_area("Paste abstract, methods, or text here:")
difficulty = st.selectbox("Select your expertise level:", ["Beginner", "Intermediate", "Expert"])

# --- Initialize session state ---
for key in ["summary", "quiz", "user_answers", "score", "recommended_papers"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key == "user_answers" else [] if key in ["quiz", "recommended_papers"] else None

# --- Regex-based sentence tokenizer ---
def simple_sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 0]

# --- Simple keyword extraction ---
def extract_keywords(text, top_n=5):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]

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
    return pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")

qg_pipeline = load_qg_pipeline()

# --- Fetch arXiv papers ---
def fetch_arxiv_abstracts(query, max_results=5):
    query_encoded = urllib.parse.quote(query)
    base_url = "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}"
    feed_url = base_url.format(query_encoded, max_results)
    feed = feedparser.parse(feed_url)
    
    papers = []
    for entry in feed.entries:
        title = entry.title
        abstract = entry.summary.replace('\n', ' ').strip()
        doi = entry.get('arxiv_doi', 'N/A')
        papers.append({"title": title, "abstract": abstract, "doi": doi})
    return papers

# --- Generate MCQ quiz ---
def generate_mcq_quiz(summary_text, level="Beginner"):
    sentences = simple_sent_tokenize(summary_text)
    quiz = []
    num_questions = min(5, len(sentences))
    
    for _ in range(num_questions):
        sent = random.choice(sentences)
        if level == "Beginner":
            q_prompt = f"Generate a simple factual question for a beginner based on: {sent}"
        elif level == "Intermediate":
            q_prompt = f"Generate a multiple choice question with some technical detail based on: {sent}"
        else:
            q_prompt = f"Generate a challenging conceptual question for experts based on: {sent}"
        
        question_text = qg_pipeline(q_prompt, max_length=64, num_return_sequences=1)[0]['generated_text']
        correct_answer = sent
        distractors = [s for s in sentences if s != correct_answer]
        distractors = random.sample(distractors, min(3, len(distractors)))
        options = distractors + [correct_answer]
        random.shuffle(options)
        quiz.append({"question": question_text, "options": options, "answer": correct_answer})
    return quiz

# --- Summarize text and generate quiz ---
if st.button("Summarize"):
    if user_text.strip():
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
        sentences = simple_sent_tokenize(raw_summary)
        summary = " ".join([s.strip().capitalize().rstrip(' .') + '.' for s in sentences])
        st.session_state.summary = summary
        st.session_state.quiz = generate_mcq_quiz(summary, level=difficulty)
        st.session_state.user_answers = {}
        st.session_state.score = None
        st.session_state.recommended_papers = []

# --- Display summary and quiz ---
if st.session_state.summary:
    st.subheader("Summary")
    st.write(st.session_state.summary)

if st.session_state.quiz:
    st.subheader("Quiz")
    for i, q in enumerate(st.session_state.quiz, 1):
        st.markdown(f"**Q{i}: {q['question']}**")
        selected = st.radio("Select your answer:", q["options"], key=f"quiz_{i}")
        st.session_state.user_answers[f"q{i}"] = selected

    if st.button("Submit Quiz"):
        score = sum(1 for i, q in enumerate(st.session_state.quiz, 1) 
                    if st.session_state.user_answers.get(f"q{i}") == q["answer"])
        percent_score = score / len(st.session_state.quiz) * 100
        st.session_state.score = percent_score
        st.success(f"Your quiz score: {percent_score:.1f}%")

        # Recommend more papers if score < 80%
        if percent_score < 80:
            st.info("We recommend reading more papers on this topic for better understanding.")
            keywords = extract_keywords(st.session_state.summary)
            query = " ".join(keywords)
            papers = fetch_arxiv_abstracts(query, max_results=5)
            recommended = []
            for p in papers:
                raw_sum = summarizer(p['abstract'], max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                recommended.append({
                    "title": p['title'],
                    "abstract": p['abstract'],
                    "summary": raw_sum,
                    "doi": p['doi']
                })
            st.session_state.recommended_papers = recommended

# --- Display recommended papers ---
if st.session_state.recommended_papers:
    st.subheader("Recommended Papers for Further Reading")
    for i, p in enumerate(st.session_state.recommended_papers, 1):
        st.markdown(f"**Paper {i}: {p['title']}**")
        st.write(p['abstract'])
        st.markdown(f"**Summary:** {p['summary']}")
        st.markdown(f"**DOI:** {p['doi']}")

