import streamlit as st
from rag import WellnessRAG
from ollama_client import ask_ollama


# Page settings
st.set_page_config(
    page_title="AI Health & Wellness Assistant",
    page_icon="🧠",
    layout="centered"
)


# Title
st.title("🧠 AI Health & Wellness Assistant")

st.write(
    "Ask general questions about healthy habits, exercise, sleep, "
    "food, hydration, and stress management."
)


# Safety message
st.warning(
    "Educational project only. This assistant is not a doctor and "
    "does not diagnose diseases, prescribe medicines, or replace "
    "professional medical care."
)


# Load RAG system only once
@st.cache_resource
def load_rag():
    return WellnessRAG()


rag = load_rag()


# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
question = st.chat_input(
    "Example: How can I improve my sleep?"
)


if question:

    # Show user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                # Retrieve relevant information from ChromaDB
                context, sources = rag.search(
                    question,
                    top_k=3
                )

                # Send question + retrieved information to Ollama
                answer = ask_ollama(
                    question,
                    context
                )

                # Display answer
                st.markdown(answer)

                # Show source information
                if sources:

                    with st.expander(
                        "Sources used from knowledge base"
                    ):

                        for source in sources:
                            st.write(f"• {source}")

                # Save answer in chat history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as error:

                st.error(
                    f"Error: {error}"
                )

                st.info(
                    "Make sure Ollama is running and the selected "
                    "model has been downloaded."
                )


# Sidebar
with st.sidebar:

    st.header("Project Information")

    st.write("**AI Stack**")

    st.write(
        "Python + Streamlit + Ollama + "
        "Hugging Face + ChromaDB"
    )

    st.write("**RAG Flow**")

    st.write(
        "1. User Question\n"
        "2. Create Embedding\n"
        "3. Search ChromaDB\n"
        "4. Retrieve Relevant Context\n"
        "5. Send Context to Ollama\n"
        "6. Generate Answer"
    )

    # Clear chat button
    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()