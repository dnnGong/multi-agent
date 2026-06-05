#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import streamlit as st
from openai import OpenAI

from agents import Head_Agent
from rag import DEFAULT_PDF_PATH, DEFAULT_STORE_PATH, LocalVectorStore


st.set_page_config(page_title="Multi-Agent ML RAG Assistant", page_icon=":material/school:", layout="wide")


def _reset_agent() -> None:
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("last_path", None)


def _get_secret(name: str) -> str:
    return st.secrets.get(name, os.environ.get(name, "")) if hasattr(st, "secrets") else os.environ.get(name, "")


with st.sidebar:
    st.header("RAG Settings")

    openai_key = st.text_input(
        "OpenAI API key",
        value=_get_secret("OPENAI_API_KEY"),
        type="password",
        help="Required for embeddings, rewriting, moderation, and answer generation.",
    )
    backend = st.radio("Retriever backend", ["Local JSON", "Pinecone"], horizontal=True)

    pdf_path = st.text_input("PDF path", value=str(DEFAULT_PDF_PATH))
    store_path = st.text_input("Local vector store", value=str(DEFAULT_STORE_PATH))
    top_k = st.slider("Retrieved chunks", min_value=3, max_value=8, value=5)

    if backend == "Pinecone":
        pinecone_key = st.text_input("Pinecone API key", value=_get_secret("PINECONE_API_KEY"), type="password")
        pinecone_index = st.text_input("Pinecone index", value=os.environ.get("PINECONE_INDEX_NAME", "machine-learning-textbook"))
        pinecone_namespace = st.text_input("Pinecone namespace", value=os.environ.get("PINECONE_NAMESPACE", "ns-2500"))
    else:
        pinecone_key = ""
        pinecone_index = ""
        pinecone_namespace = "ns-2500"

    st.divider()
    st.subheader("Build Local Index")
    chunk_size = st.number_input("Chunk size", min_value=300, max_value=3000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=800, value=150, step=25)
    build_clicked = st.button("Build / Rebuild local vector store", use_container_width=True)
    if build_clicked:
        if not openai_key:
            st.error("Please enter OPENAI_API_KEY first.")
        else:
            with st.spinner("Extracting PDF and creating embeddings..."):
                try:
                    client = OpenAI(api_key=openai_key)
                    store = LocalVectorStore(store_path)
                    store.build_from_pdf(
                        client=client,
                        pdf_path=pdf_path,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )
                    _reset_agent()
                    st.success(f"Built {len(store.documents)} chunks at {store_path}")
                except Exception as exc:
                    st.error(f"Failed to build vector store: {exc}")

    st.divider()
    if st.button("Reset conversation", use_container_width=True):
        _reset_agent()
        st.rerun()


st.title("Multi-Agent ML RAG Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

agent_config = {
    "backend": backend,
    "store_path": store_path,
    "pinecone_index": pinecone_index,
    "pinecone_namespace": pinecone_namespace,
    "top_k": int(top_k),
}
if st.session_state.get("agent_config") != agent_config:
    st.session_state.pop("agent", None)
    st.session_state.agent_config = agent_config

if "agent" not in st.session_state and openai_key:
    try:
        st.session_state.agent = Head_Agent(
            openai_key=openai_key,
            pinecone_key=pinecone_key,
            pinecone_index_name=pinecone_index,
            pinecone_namespace=pinecone_namespace,
            vector_store_path=store_path,
            use_local_rag=(backend == "Local JSON"),
            retrieval_k=int(top_k),
        )
    except Exception as exc:
        st.info(f"Configure the retriever in the sidebar. Details: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about machine learning, e.g. What is overfitting?")
if prompt:
    if not openai_key:
        st.error("Please enter OPENAI_API_KEY in the sidebar.")
        st.stop()
    if "agent" not in st.session_state:
        st.error("Agent is not ready. Build the local store or check Pinecone settings.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking through the agent pipeline..."):
            try:
                answer, path = st.session_state.agent.run_one_turn(prompt)
                docs = list(getattr(st.session_state.agent, "last_docs", []))
                st.session_state.agent.conv_history.append({"role": "user", "content": prompt})
                st.session_state.agent.conv_history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                answer = f"Error: {exc}"
                path = []
                docs = []
        st.markdown(answer)
        if path:
            with st.expander("Agent path", expanded=False):
                st.code(" -> ".join(path))
        if docs:
            with st.expander("Retrieved sources", expanded=False):
                for i, doc in enumerate(docs, start=1):
                    page = doc.meta.get("page", "n/a")
                    st.markdown(f"**Source {i}** | score `{doc.score:.4f}` | page `{page}`")
                    st.write(doc.text[:700] + ("..." if len(doc.text) > 700 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_path = path

if Path(store_path).exists():
    with st.sidebar:
        st.success("Local vector store found")
else:
    with st.sidebar:
        st.warning("Local vector store not built yet")
