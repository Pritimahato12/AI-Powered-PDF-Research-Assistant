import hashlib
from io import BytesIO

import streamlit as st

from modules.pdf_extractor import extract_text_from_pdf
from modules.text_preprocessing import preprocess_pages, create_chunks
from modules.embeddings import load_embedding_model, generate_embeddings
from modules.vector_store import create_vector_index, search_hybrid_chunks
from modules.llm import load_llm, generate_answer


st.set_page_config(
    page_title="AI PDF Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("AI-Powered PDF Research Assistant")
st.caption(
    "Ask questions across multiple research papers, compare documents, "
    "and inspect source citations."
)


@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource
def get_ollama_client():
    return load_llm()


@st.cache_data(show_spinner=False)
def process_uploaded_pdfs(file_items):
    """Extract, clean and chunk PDFs. Cached by filename and file bytes."""
    all_chunks = []
    document_stats = []

    for filename, pdf_bytes in file_items:
        extracted_pages = extract_text_from_pdf(BytesIO(pdf_bytes))
        processed_pages = preprocess_pages(extracted_pages)
        document_chunks = create_chunks(processed_pages)

        for chunk in document_chunks:
            chunk["source"] = filename

        all_chunks.extend(document_chunks)
        document_stats.append({
            "filename": filename,
            "pages": len(extracted_pages),
            "chunks": len(document_chunks)
        })

    return all_chunks, document_stats


@st.cache_data(show_spinner=False)
def get_cached_embeddings(chunk_texts):
    model = get_embedding_model()
    return generate_embeddings(model, [{"text": text} for text in chunk_texts])


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "corpus_signature" not in st.session_state:
    st.session_state.corpus_signature = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


uploaded_files = st.file_uploader(
    "Upload one or more research papers",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDF research papers and search across them."
)

if not uploaded_files:
    st.info("Upload at least one PDF to begin.")
    st.stop()

file_items = tuple(
    (file.name, file.getvalue())
    for file in uploaded_files
)

signature_hash = hashlib.sha256()
for filename, content in file_items:
    signature_hash.update(filename.encode("utf-8"))
    signature_hash.update(content)
corpus_signature = signature_hash.hexdigest()

if st.session_state.corpus_signature != corpus_signature:
    st.session_state.messages = []
    st.session_state.last_sources = []
    st.session_state.corpus_signature = corpus_signature

with st.spinner("Processing uploaded research papers..."):
    try:
        chunks, document_stats = process_uploaded_pdfs(file_items)
    except Exception as exc:
        st.error(f"Could not process the uploaded PDFs: {exc}")
        st.stop()

if not chunks:
    st.error("No readable text was found in the uploaded PDFs.")
    st.stop()

st.success(f"Processed {len(document_stats)} PDF(s) successfully.")

with st.expander("Uploaded document details", expanded=True):
    for item in document_stats:
        st.write(
            f"**{item['filename']}** — "
            f"{item['pages']} pages, {item['chunks']} chunks"
        )

model = get_embedding_model()

with st.spinner("Preparing semantic and keyword search indexes..."):
    try:
        chunk_texts = tuple(chunk["text"] for chunk in chunks)
        embeddings = get_cached_embeddings(chunk_texts)
        vector_index = create_vector_index(embeddings)
    except Exception as exc:
        st.error(f"Could not build the search index: {exc}")
        st.stop()

st.caption(
    f"Search ready: {len(chunks)} passages · "
    f"{embeddings.shape[1]} embedding dimensions · "
    "FAISS + BM25 hybrid retrieval"
)

st.subheader("Research assistant")

mode = st.radio(
    "Choose how to use your documents",
    ["Ask a question", "Compare documents"],
    horizontal=True
)

if mode == "Compare documents" and len(document_stats) < 2:
    st.warning("Upload at least two PDFs to compare information across documents.")

# Render previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used for this answer"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']} — page {source['page_number']}**"
                    )
                    st.caption(source["text"])

question = st.chat_input(
    "Ask about your papers or compare their findings..."
)

if question:
    if mode == "Compare documents" and len(document_stats) < 2:
        st.warning("Please upload at least two PDFs before comparing documents.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                results = search_hybrid_chunks(
                    question=question,
                    model=model,
                    index=vector_index,
                    chunks=chunks,
                    top_k=6,
                    candidate_k=15
                )

                if not results:
                    answer = (
                        "I could not find relevant passages in the uploaded documents."
                    )
                    sources = []
                else:
                    context_parts = []
                    for number, result in enumerate(results, start=1):
                        context_parts.append(
                            f"[Source {number}: {result['source']}, "
                            f"page {result['page_number']}]\n{result['text']}"
                        )

                    context = "\n\n".join(context_parts)

                    # Pass recent conversation turns for follow-up questions.
                    history = st.session_state.messages[-7:-1]

                    client = get_ollama_client()
                    answer = generate_answer(
                        client=client,
                        question=question,
                        context=context,
                        history=history,
                        mode=mode
                    )

                    sources = results

                st.markdown(answer)

                if sources:
                    with st.expander("Sources used for this answer", expanded=True):
                        for source in sources:
                            st.markdown(
                                f"**{source['source']} — page "
                                f"{source['page_number']}**"
                            )
                            st.caption(
                                f"Hybrid score: {source['score']:.4f} "
                                f"(semantic: {source['semantic_score']:.4f}, "
                                f"keyword: {source['keyword_score']:.4f})"
                            )
                            st.write(source["text"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as exc:
                error_message = f"Unable to generate an answer: {exc}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })

with st.sidebar:
    st.header("Conversation")
    st.write(f"Messages: {len(st.session_state.messages)}")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    st.divider()
    st.markdown("**Active features**")
    st.write("✓ Multiple PDF uploads")
    st.write("✓ Session chat history")
    st.write("✓ Filename and page citations")
    st.write("✓ Cross-document comparison mode")
    st.write("✓ Hybrid FAISS + BM25 search")
