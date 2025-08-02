import streamlit as st
import pandas as pd
import plotly.express as px
from legal_analyzer import read_txt, read_docx, read_pdf, summarize, analyze_document
from transformers import pipeline

# --- Initialize models once ---
qa_pipeline = pipeline("question-answering")
# (You can add sentiment, emotion models here if you want in-depth chatbot answers)

# --- Page Setup ---
st.set_page_config(page_title="ClauseWise - Legal Document Analyzer", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .title-style {
            font-size: 2.2rem;
            font-weight: 700;
            color: white;
        }
        .header-bar {
            background-color: #5B2EEC;
            padding: 1rem 2rem;
            border-radius: 0 0 10px 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .custom-box {
            border: 2px dashed #ccc;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }
        .file-icon {
            font-size: 2rem;
            color: #555;
        }
        .section-title {
            font-size: 1.3rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header-bar">
    <div class="title-style">⚖️ ClauseWise — AI-Powered Legal Document Analyzer</div>
    <div>
        <button style="background-color:#6c63ff;color:white;border:none;padding:0.5rem 1rem;border-radius:5px;">Help</button>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# --- Upload Section ---
st.markdown("### 📁 Document Upload")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Single Document Analysis")
    uploaded_file = st.file_uploader(
        "Drag & drop your legal document here",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )
    st.markdown("Supports PDF, DOCX, TXT (Max 10MB)", unsafe_allow_html=True)

with col2:
    st.markdown("#### Document Comparison")
    doc1 = st.file_uploader("Document 1", type=["pdf", "docx", "txt"], key="doc1")
    doc2 = st.file_uploader("Document 2", type=["pdf", "docx", "txt"], key="doc2")

st.divider()

# --- Analysis Options ---
st.markdown("#### 🧠 Analysis Options")
opt1, opt2, opt3 = st.columns(3)
with opt1:
    deep_clause = st.checkbox("Deep Clause Analysis", value=True)
with opt2:
    timeline_extraction = st.checkbox("Timeline Extraction", value=True)
with opt3:
    tone_analysis = st.checkbox("Tone Analysis", value=True)

# --- Sidebar Chatbot ---
with st.sidebar:
    st.markdown("## 💬 Ask ClauseBot")
    question = st.text_input("Ask about your document:")
    if question:
        if uploaded_file:
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext == "txt":
                context = read_txt(uploaded_file)
            elif ext == "docx":
                context = read_docx(uploaded_file)
            elif ext == "pdf":
                context = read_pdf(uploaded_file)
            else:
                context = ""
            if context:
                answer = qa_pipeline(question=question, context=context)
                st.success(f"**Answer:** {answer['answer']}")
            else:
                st.warning("Please upload a supported document first.")
        else:
            st.warning("Please upload a document to use ClauseBot.")

# --- Helper function to load text based on extension ---
def load_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == "txt":
        return read_txt(file)
    elif ext == "docx":
        return read_docx(file)
    elif ext == "pdf":
        return read_pdf(file)
    else:
        return None

# --- Main: Analyze Single Document ---
if st.button("🔍 Analyze Document"):
    if uploaded_file:
        text = load_text(uploaded_file)
        if not text:
            st.error("Unsupported file format or error reading file.")
        else:
            # Show Summary
            st.markdown("## 📑 Document Summary")
            summary_text = summarize(text)
            st.write(summary_text)

            # Analyze Clauses
            st.markdown("## ⚖️ Clause Analysis")
            analysis = analyze_document(text)

            # Prepare data for charts
            risk_levels = [clause["importance"] for clause in analysis]
            categories = [clause["category"] for clause in analysis]
            sentiments = [clause["sentiment"] for clause in analysis]

            # Show Charts
            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("### 📊 Risk Level Distribution")
                risk_counts = pd.Series(risk_levels).value_counts().sort_index()
                st.bar_chart(risk_counts)

            with col_b:
                st.markdown("### 🥧 Clause Category Breakdown")
                category_counts = pd.Series(categories).value_counts().reset_index()
                category_counts.columns = ["Category", "Count"]
                fig_pie = px.pie(category_counts, values="Count", names="Category", title="Clause Categories")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_c:
                st.markdown("### 😐 Sentiment Overview")
                sentiment_counts = pd.Series(sentiments).value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                fig_bar = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment", title="Clause Sentiments")
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")

            # Show clause details with suggestions and feedback
            for clause in analysis:
                with st.expander(f"Clause {clause['clause_num']} — {clause['category']} (Importance: {clause['importance']})"):
                    st.markdown(f"**Text:** {clause['clause_text']}")
                    if timeline_extraction and clause["dates"]:
                        st.markdown("📅 **Dates found:**")
                        for dtext, dparsed in clause["dates"]:
                            st.markdown(f"- {dtext} → {dparsed}")
                    if tone_analysis:
                        st.markdown(f"🧠 **Sentiment:** {clause['sentiment']}, **Tone:** {clause['tone']}")
                    if clause["boilerplate"]:
                        st.warning("⚠️ Detected as boilerplate clause.")
                    st.info(f"💡 **Suggestion:** {clause['rewrite_suggestion']}")
                    st.radio("📢 Was this suggestion helpful?", ["👍 Yes", "👎 No"], key=f"feedback_{clause['clause_num']}")

            # Export summary button
            st.download_button("📤 Download Summary", summary_text.encode('utf-8'), file_name="ClauseWise_Summary.txt")

    else:
        st.warning("Please upload a document to analyze.")

# --- Document Comparison ---
if st.button("📄 Compare Documents"):
    if doc1 and doc2:
        text1 = load_text(doc1)
        text2 = load_text(doc2)
        if text1 and text2:
            clauses1 = set(text1.split('\n'))
            clauses2 = set(text2.split('\n'))

            missing_in_2 = clauses1 - clauses2
            missing_in_1 = clauses2 - clauses1

            st.markdown("## 📂 Document Comparison Results")
            st.markdown(f"### Clauses missing in Document 2 ({len(missing_in_2)})")
            for c in missing_in_2:
                st.write(f"- {c[:150]}...")

            st.markdown(f"### Clauses missing in Document 1 ({len(missing_in_1)})")
            for c in missing_in_1:
                st.write(f"- {c[:150]}...")

        else:
            st.error("Error reading one or both documents.")
    else:
        st.warning("Please upload both documents to compare.")

# --- Load Sample Button (Optional placeholder) ---
if st.button("👁 Load Sample"):
    st.info("Sample loading not yet implemented.")
