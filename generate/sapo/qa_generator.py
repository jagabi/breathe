import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

def generate(system_prompt: str, user_input: str) -> str:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-3-flash-preview"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_input),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="MEDIUM",
        ),
        system_instruction=[
            types.Part.from_text(text=system_prompt),
        ],
    )

    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_response += chunk.text
    print()
    return full_response

if __name__ == "__main__":
    generate("You are a dataset generator.", "Generate 1 test QA pair.")
