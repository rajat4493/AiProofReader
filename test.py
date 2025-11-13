import google.generativeai as genai, os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")
resp = model.generate_content("Write a short poem about Rajat.")
print(resp)