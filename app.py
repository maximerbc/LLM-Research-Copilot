import re
import subprocess
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from get_embedding_function import get_embedding_function

# Load environment (for OPENAI_API_KEY, etc.)
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


@st.cache_resource
def get_db():
    """Load the Chroma DB once and reuse it."""
    embedding = get_embedding_function()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
    )


@st.cache_resource
def get_llm(model_name: str):
    """Load the Ollama model once and reuse it."""
    return ChatOllama(model=model_name)


@st.cache_resource
def get_ollama_models():
    """Return available Ollama model names; fall back to empty."""
    try:
        output = subprocess.check_output(["ollama", "list"], text=True)
    except Exception:
        return []

    models = []
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in lines[1:]:
        model = line.split()[0]
        if model:
            models.append(model)
    return models


def build_prompt_and_context(results, query_text: str):
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text,
    )
    return prompt, context_text


def quality_score(answer_text: str, context_text: str) -> float:
    """Simple grounding score: fraction of answer tokens found in context."""
    answer_tokens = {
        t for t in re.findall(r"[A-Za-z0-9]+", answer_text.lower()) if len(t) > 2
    }
    context_tokens = {
        t for t in re.findall(r"[A-Za-z0-9]+", context_text.lower()) if len(t) > 2
    }
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens.intersection(context_tokens)
    return len(overlap) / len(answer_tokens)


def run_rag(query_text: str, model_name: str, k: int):
    db = get_db()
    llm = get_llm(model_name)

    # 1) Retrieve top-k chunks
    results = db.similarity_search_with_score(query_text, k=k)

    # 2) Build context and prompt
    prompt, context_text = build_prompt_and_context(results, query_text)

    # 3) Ask the LLM
    response = llm.invoke(prompt)
    answer_text = response.content

    score = quality_score(answer_text, context_text)

    return answer_text, results, score


def main():
    st.set_page_config(page_title="LLM Reasearch Rag Viewer", layout="wide")
    st.title("LLM Research RAG Viewer")

    st.markdown("Ask a question about your LLM architecture documents and see the retrieved chunks + answer.")

    available_models = set(get_ollama_models())
    model_choices = ["mistral:latest", "fine_tuned_model:latest"]
    models_to_compare = st.sidebar.multiselect(
        "Models to compare",
        model_choices,
        default=model_choices,
    )
    for model_name in model_choices:
        if model_name not in available_models:
            st.sidebar.warning(f"Model not found in Ollama: {model_name}")

    k = st.sidebar.slider("Retrieved chunks (k)", min_value=2, max_value=6, value=3)

    query_text = st.text_area("Question", height=80, placeholder="e.g. Why does the Transformer use multi-head attention instead of a single attention mechanism?")

    if st.button("Run RAG") and query_text.strip():
        with st.spinner("Retrieving context and generating answers…"):
            results = get_db().similarity_search_with_score(query_text.strip(), k=k)
            prompt, context_text = build_prompt_and_context(results, query_text.strip())

            model_outputs = []
            for model_name in models_to_compare:
                llm = get_llm(model_name)
                response = llm.invoke(prompt)
                answer_text = response.content
                score = quality_score(answer_text, context_text)
                model_outputs.append((model_name, answer_text, score))

        # Question
        st.markdown("### Question")
        st.write(query_text.strip())

        # Answer
        st.markdown("### Answers (with quality score)")
        st.caption("Quality score = fraction of answer tokens found in retrieved context (0 to 1).")
        if not model_outputs:
            st.info("Select at least one model to compare.")
        else:
            cols = st.columns(len(model_outputs))
            for col, (model_name, answer_text, score) in zip(cols, model_outputs):
                with col:
                    st.markdown(f"**{model_name}**")
                    st.metric("Quality score", f"{score:.2f}")
                    st.write(answer_text)

        # Retrieved chunks
        st.markdown(f"### Retrieved context ({k} chunks)")
        if not results:
            st.info("No chunks were retrieved from the vector database.")
        else:
            for i, (doc, score) in enumerate(results, start=1):
                with st.expander(f"Chunk {i}  •  score: {score:.3f}"):
                    md = doc.metadata
                    st.write(
                        f"**Source:** {md.get('source', 'unknown')}  "
                        f"**Page:** {md.get('page', '?')}  "
                        f"**ID:** {md.get('id', 'n/a')}"
                    )
                    st.write(doc.page_content)


if __name__ == "__main__":
    main()
