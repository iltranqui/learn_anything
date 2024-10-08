import os
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192*8,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
  ]
)

# Read the contents of the text_only.txt file
with open('metadata.csv', 'r') as file:
    contents = file.read()

Instructions = 'Generate a list and a smalle explainaed for each parameter \n\n'

# Send the contents as input to the chat session
response = chat_session.send_message(Instructions + contents)

print(response.text)

# Save the summary to a text file
with open('summary.txt', 'w') as file:
    file.write(response.text)