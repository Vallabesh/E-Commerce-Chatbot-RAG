import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()


def talk(query):

    prompt = f"""
    You are a friendly AI assistant for an e-commerce website.

    Respond naturally to the user's question.

    User: {query}
    """

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content
