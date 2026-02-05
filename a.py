import google.generativeai as genai

# Replace with your actual Gemini API key
API_KEY = "AIzaSyAFLaH3K3Y4PjZ7C0BmVwPhhIItIfu3b2wdf"

genai.configure(api_key=API_KEY)

print("Available Models:")
for model in genai.list_models():
    # Only list models that support content generation
    if 'generateContent' in model.supported_generation_methods:
        print(f" - {model.name}")

print("\nAvailable Embedding Models:")
for model in genai.list_models():
    # Only list models that support embeddings
    if 'embedContent' in model.supported_generation_methods:
        print(f" - {model.name}")
