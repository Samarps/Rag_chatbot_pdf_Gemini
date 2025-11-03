from google import genai
from config import GEMINI_API_KEY

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


# print("Available models:")
# for model in client.models.list():
#     print(model.name)


model_name = "models/gemini-2.5-flash"

resp = client.models.generate_content(
    model = model_name,
    contents = "Hello Gemini! Tell me a maths facts, a short one?"
)

print("Response from Gemini:")
print(resp.text)