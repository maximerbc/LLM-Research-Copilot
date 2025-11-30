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
def get_llm():
    """Load the Ollama/Mistral model once and reuse it."""
    return ChatOllama(model="mistral")


def run_rag(query_text: str):
    db = get_db()
    llm = get_llm()

    # 1) Retrieve top-k chunks
    results = db.similarity_search_with_score(query_text, k=5)

    # 2) Build context and prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text,
    )

    # 3) Ask the LLM
    response = llm.invoke(prompt)
    answer_text = response.content

    return answer_text, results


def main():
    st.set_page_config(page_title="CNN RAG Viewer", layout="wide")
    st.title("CNN RAG Viewer")

    st.markdown("Ask a question about your CNN documents and see the retrieved chunks + answer.")

    query_text = st.text_area("Question", height=80, placeholder="e.g. What is a convolutional neural network?")

    if st.button("Run RAG") and query_text.strip():
        with st.spinner("Retrieving context and generating answer…"):
            answer, results = run_rag(query_text.strip())

        # Question
        st.markdown("### Question")
        st.write(query_text.strip())

        # Answer
        st.markdown("### Answer")
        st.write(answer)

        # Retrieved chunks
        st.markdown("### Retrieved context (5 chunks)")
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
