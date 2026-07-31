import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

chat = client.chats.create(
    model="gemini-flash-latest",
    config={
        "system_instruction": """Du bist ein hilfreicher Karriereberater. Antworte immer auf Deutsch.
    
    Stelle gezielte Fragen, um Folgendes über den Nutzer herauszufinden:
- Abschluss/Ausbildung
- Berufserfahrung (Jahre, Rollen)
- Fachliche Skills (z.B. Software, Methoden, Sprachen)
- Berufliche Interessen/Präferenzen (z.B. bevorzugte Branche, Arbeitsweise)

Stelle die Fragen nacheinander, nicht alle auf einmal. Sobald du genug Informationen gesammelt hast, fasse das Profil des Nutzers strukturiert zusammen, z.B. so:

PROFIL:
- Abschluss: ...
- Erfahrung: ...
- Skills: ...
- Interessen: ..."""
     
    }
    
    )


while True:
    nachricht = input("Du: ")
    if nachricht == "exit":
        break

    response = chat.send_message(nachricht)
    print(f"KI: {response.text}")
    
