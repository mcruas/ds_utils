import openai
import os
from dotenv import load_dotenv

def get_openai_api_key():
    """Returns the OpenAI API key from .env file."""
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def chat_gpt(prompt,  information=None, system_text=None, 
             model="gpt-3.5-turbo-16k", max_tokens=1000, temperature=0.0):
    messages=[]
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if information:
        messages.append({"role": "user", "content": information})
    messages.append({"role": "user", "content": prompt})

    completion = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return print(completion.choices[0].message.content)
    # return messages