import streamlit as st
from ingest import ingest_pdfs
from rag import answer_question
#from dotenv import load_dotenv
import os
from collections import defaultdict

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

MAX_QUERIES = 20

#load_dotenv()

api_key = st.secrets["GROQ_API_KEY"]
admin_pass = st.secrets["ADMIN_PASSWORD"]





if "show_admin" not in st.session_state:
    st.session_state.show_admin = False

if st.button("Admin"):
    st.session_state.show_admin = True

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

st.set_page_config(
    page_title="Internal Knowledge Assistant",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# st.caption(
#     "Answers questions using your organization’s documents only, with exact source references."
# )

st.markdown(
    """
    <div style="
        font-size: 1.05rem;
        color: #b0b7c3;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    ">
        This assistant answers questions using your organization’s documents only.
        All responses are grounded in approved files and include source references.
    </div>
    """,
    unsafe_allow_html=True
)

st.title("📄 Internal Knowledge Assistant")
st.caption("Ask questions across company documents with accurate answers and sources")

if st.session_state.show_admin:
    with st.sidebar:
        st.subheader("Admin Login")

        if not st.session_state.is_admin:
            admin_input = st.text_input("Admin password", type="password")

            if admin_input and admin_input == admin_pass:
                st.session_state.is_admin = True
                st.success("Admin mode enabled")
        else:
            st.success("Admin mode active")
            if st.button("Logout admin"):
                st.session_state.is_admin = False
                st.rerun()

if st.session_state.is_admin:
    st.subheader("Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF documents (This will replace existing Knowledge Base)",
        type=["pdf"],
        accept_multiple_files=True
    )
    if uploaded_files:
        confirm = st.checkbox(
            "I understand this will replace the existing documents"
        )

        if confirm and st.button("Rebuild Knowledge Base"):
            with st.spinner("Rebuilding knowledge base..."):
                st.session_state.vectorstore = ingest_pdfs(uploaded_files)
                st.success("Knowledge base updated successfully")
else:
    uploaded_files = None


if st.session_state.is_admin and uploaded_files:
    with st.spinner("Indexing documents..."):
        st.session_state.vectorstore = ingest_pdfs(uploaded_files)
        st.success("Documents indexed successfully")

if "vectorstore" not in st.session_state:
    st.info("Knowledge base not set up yet. Please contact admin.")
    st.stop()


query = st.chat_input("Ask a question about the documents")
if st.session_state.query_count >= MAX_QUERIES:
    st.warning("Demo limit reached. Please contact admin.")
    st.stop()

if query:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_question(
                st.session_state.vectorstore,
                query,
                #st.secrets["GROQ_API_KEY"]
                api_key
            )
            st.markdown(answer)

            with st.expander("Sources"):
                grouped_sources = defaultdict(set)
                for doc in sources:
                    source = doc.metadata.get("source", "Unknown Document")
                    page = doc.metadata.get("page", "?")
                    grouped_sources[source].add(page)
                    
                for source, pages in grouped_sources.items():
                    page_list = ", ".join(str(p) for p in sorted(pages))
                    st.markdown(f"- **{source}**, pages {page_list[0]}")

st.session_state.query_count += 1

            # st.subheader("Debug: Top Retrieved Chunks")
            # for i, d in enumerate(sources[:2]):
            #     st.write(f"Chunk {i+1}")
            #     st.write(d.page_content[:500])
            

