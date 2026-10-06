import ollama


# Ollama model
MODEL_NAME = "gemma3:4b"


# Instructions for the AI
SYSTEM_PROMPT = """

You are an educational AI health and wellness assistant.

Give general wellness information only.

Do not diagnose diseases.

Do not prescribe medicines.

Do not give dangerous treatment instructions.

Use the provided knowledge-base context
when it is relevant.

If the context does not contain enough information,
say that clearly.

Give simple and practical answers.

For serious or emergency symptoms,
tell the user to seek professional medical help.

Keep answers easy for a student to understand.

""".strip()


def ask_ollama(question, context):

    # Create prompt
    prompt = f"""

Knowledge Base Context:

{context}


User Question:

{question}


Answer the user's question using the
knowledge base context when possible.

""".strip()


    # Send request to Ollama
    response = ollama.chat(

        model=MODEL_NAME,

        messages=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": prompt
            }

        ]
    )


    # Return AI answer
    return response.message.content