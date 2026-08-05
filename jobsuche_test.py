import numpy as np
import requests
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(vektor_a, vektor_b):
    dot_product = np.dot(vektor_a, vektor_b)
    norm_a = np.linalg.norm(vektor_a)
    norm_b = np.linalg.norm(vektor_b)
    return dot_product / (norm_a * norm_b)

# Echte Jobs von der Bundesagentur für Arbeit API holen
url = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/app/jobs"
headers = {"X-API-Key": "jobboerse-jobsuche"}
params = {"was": "", "wo": "Berlin", "page": 1, "size": 10}

response = requests.get(url, headers=headers, params=params)
daten = response.json()

job_texte = []
for job in daten["stellenangebote"]:
    text_string = f"{job['beruf']} - {job['titel']}, bei {job['arbeitgeber']}"
    job_texte.append(text_string)

# Unser Nutzerprofil (aus dem Chat geparst)
nutzerprofil_text = "Master im Maschinenbau, 5 Jahre Berufserfahrung, CAD, MS-Office, Autodesk Inventor, Autodesk Vault, Autodesk AutoCAD, Creo Parametric, Atlassian Jira, Microsoft Navision Deutsch"

profil_embedding = model.encode(nutzerprofil_text)

aehnlichkeiten = []
for job_text in job_texte:
    job_embedding = model.encode(job_text)
    ähnlichkeit = cosine_similarity(profil_embedding, job_embedding)
    aehnlichkeiten.append(ähnlichkeit)

kombiniert = list(zip(job_texte, aehnlichkeiten))
sortiert = sorted(kombiniert, key=lambda x: x[1], reverse=True)

for job, wert in sortiert:
    print(f"{job}: {wert:.2f}")