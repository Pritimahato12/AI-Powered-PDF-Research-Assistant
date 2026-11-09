import os
from pyexpat.errors import messages
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()


# Step 1: Create Groq client
def load_llm():

    api_key = os.getenv("GROQ_API_KEY")

    # Also support Streamlit Cloud secrets
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file or Streamlit secrets."
        )

    client = Groq(api_key=api_key)

    return client


# Step 2: Generate AI answer
def generate_answer(
    client,
    question,
    context,
    history=None,
    mode="Ask a question"
):

    if mode == "Compare documents":

        instruction = """
        Compare the uploaded research documents.
        Explain their similarities and differences.
        Mention source filenames and page numbers.
        Use only the supplied context.
        """

    else:

        instruction = """
        Answer the user's question using only the
        information provided in the retrieved context.

        If the answer is unavailable in the context,
        clearly say so.

        Mention the source filename and page number
        for important information.
        """

    system_prompt = f"""
    You are an AI-powered research paper assistant.

    {instruction}

    Do not invent facts or source citations.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Add previous conversation messages safely
    if history:
        for msg in history[-6:]:

            role = msg.get("role")
            content = msg.get("content")

            # Only send supported message fields to Groq
            if role in ["user", "assistant"] and isinstance(content, str):
                messages.append({
                    "role": role,
                    "content": content
                })

    # Add current question and retrieved context
    user_prompt = f"""
    Retrieved Context:

    {context}

    User Question:

    {question}

    Generate a clear and accurate answer.
    """

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    # Send request to Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        max_completion_tokens=1200
    )

    return response.choices[0].message.content