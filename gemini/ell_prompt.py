import ell
import os
import google.generativeai as genai
import os
from dotenv import load_dotenv

from anthropic import Anthropic
from gemini_class import register


@ell.simple(model="gemini-1.5-flaah", client=register, max_tokens=100)
def hello(name: str):
    """You are a helpful assistant."""
    return f"Say hello to {name}!"

@ell.simple(model="claude-3-5-sonnet-20240620", client=Anthropic(), max_tokens=100)
def hello(name: str):
    """You are a helpful assistant."""
    return f"Say hello to {name}!"

hello("Elliot")