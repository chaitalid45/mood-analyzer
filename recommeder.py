from openai import OpenAI
import json

with open("api_store","r") as file:
    api_1 = json.load(file)

apiKey = api_1['open_ai_api']

client = OpenAI(api_key=apiKey)

class classAgent:
    def __init__(self, mood):
        self.mood = mood
    
    def song_suggestion(self):
        prompt = f"""
You are an expert DJ and professional music curator specializing in FILM SONGS.

Your task:
Recommend REAL, well-known FILM songs based on the user's mood.

Languages required:
- Marathi (film songs only)
- Hindi (Bollywood film songs only)
- English (songs from movies preferred; popular chart songs acceptable if highly relevant)

Rules you must strictly follow:
- The songs must be REAL and must EXIST.
- Prefer songs from movies (not indie, not random low-known tracks).
- Understand the mood deeply and map it to the closest emotional category.
- If the mood is vague, intelligently refine it.
- Do NOT repeat the same song if the user asks again for the same mood — always try to suggest different but relevant songs.
- Do NOT invent song names.
- Return only valid JSON (no extra text).

Output format (must follow exactly):
{{
  "marathi_song": "<real Marathi film song>",
  "hindi_song": "<real Bollywood film song>",
  "english_song": "<real English film song>"
}}

User mood: {self.mood}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    def activity_suggestion(self):
        prompt = f"""
You are a helpful mood coach.

Task:
Based on the user's mood, suggest exactly 3 activities that support or improve the mood.

Rules:
- Understand the mood deeply and interpret it correctly.
- If the mood is vague, refine it into a clearer emotional category.
- Suggest mostly common, easy activities (like walking, music, journaling, rest, talking to someone).
- Ensure that at least 1 activity feels slightly creative, interesting, or different (not all three should feel generic).
- Activities should be safe, practical, and suitable for everyday life.
- Do NOT include any message, explanation, or extra text.
- someone like painting, playing badminton, cleaning room etc.
- Return only valid JSON.

Output format (must follow exactly):

{{
  "activities": [
    "<activity 1>",
    "<activity 2>",
    "<activity 3>"
  ]
}}

User mood: {self.mood}

"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
