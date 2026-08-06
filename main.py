from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from google import genai
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import requests
import base64
import asyncio
import httpx
from dotenv import load_dotenv 
import logging

load_dotenv()

Base = declarative_base()
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/job_matching_db")
engine = create_engine(DATABASE_URL)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Nutzer(Base):
    __tablename__ = "nutzer"
    
    id = Column(Integer, primary_key=True)
    profil_text = Column(String)

class Swipe(Base):
    __tablename__ = "swipes"

    id = Column(Integer, primary_key=True)
    nutzer_id = Column(Integer, ForeignKey("nutzer.id"))
    job_titel = Column(String)
    richtung = Column(String)

class ChatNachrichtDB(Base):
    __tablename__ = "chat_nachrichten"
    id = Column(Integer, primary_key=True)
    nutzer_id = Column(Integer, ForeignKey("nutzer.id"))
    rolle = Column(String)
    inhalt = Column(String)

Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class NutzerProfil(BaseModel):
    profil_text: str
    fachbereich: list[str]
    nutzer_id : int
    ort: str

class ChatNachricht(BaseModel):
    nachricht: str
    nutzer_id: int

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

async def suche_jobs(client, headers, fachbereich, ort):
    params = {"was": fachbereich, "wo": ort, "page": 1, "size": 10}
    url = f"https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v6/jobs"
    response = await client.get(url, headers=headers, params=params)
    return response.json()

def hole_chat_historie(nutzer_id: int):
    alle_chats = session.query(ChatNachrichtDB).filter(ChatNachrichtDB.nutzer_id == nutzer_id).all()

    chats = []

    for chat in alle_chats:
        chats.append({"role": chat.rolle, "parts": [{"text": chat.inhalt}]})
    return chats

@app.post("/chat")

async def chatte(chatnachricht: ChatNachricht):
    bisherige_historie = hole_chat_historie(chatnachricht.nutzer_id)
    chat = client.aio.chats.create(model="gemini-flash-latest", history=bisherige_historie)
    response = await chat.send_message(chatnachricht.nachricht)

    neue_nutzer_nachricht = ChatNachrichtDB(nutzer_id= chatnachricht.nutzer_id, rolle="user", inhalt= chatnachricht.nachricht)
    neue_ki_nachricht = ChatNachrichtDB(nutzer_id= chatnachricht.nutzer_id, rolle="model", inhalt= response.text )

    session.add(neue_nutzer_nachricht)
    session.add(neue_ki_nachricht)
    session.commit()

    return {"antwort": response.text}

def cosine_similarity(vektor_a, vektor_b):
    dot_product = np.dot(vektor_a, vektor_b)
    norm_a = np.linalg.norm(vektor_a)
    norm_b = np.linalg.norm(vektor_b)
    return dot_product / (norm_a * norm_b)

def hole_swipe_jobs(nutzer_id: int, richtung: str):
    swipes = session.query(Swipe).filter(Swipe.nutzer_id == nutzer_id).filter(Swipe.richtung == richtung).all()

    alle_jobtitel = []

    for swipe in swipes:
        alle_jobtitel.append(swipe.job_titel)
    return alle_jobtitel

def durchschnittliche_aehnlichkeit(neuer_job_text, bisherige_jobs):
    if len(bisherige_jobs) == 0:
        return 0

    neuer_job_embedding = model.encode(neuer_job_text)

    aehnlichkeiten = []

    for job_text in bisherige_jobs:
        job_embedding = model.encode(job_text)
        ähnlichkeit = cosine_similarity(neuer_job_embedding, job_embedding)
        aehnlichkeiten.append(ähnlichkeit)

    durchschnitt = sum(aehnlichkeiten) / len(aehnlichkeiten)
    return durchschnitt 

async def hole_jobdetails(client, referenznummer, headers):
    kodiert = base64.b64encode(referenznummer.encode()).decode()
    url = f"https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobdetails/{kodiert}"
    response = await client.get(url, headers=headers)
    return response.json()

@app.post("/matching")

async def finde_jobs(profil: NutzerProfil):
    headers = {"X-API-Key": "jobboerse-jobsuche"}
    logger.info(f"Matching-Anfrage für Nutzer {profil.nutzer_id}")
    
    async with httpx.AsyncClient() as client:
        aufgaben_suche = [suche_jobs(client,headers, fach, profil.ort) for fach in profil.fachbereich]
        alle_suchergebnisse = await asyncio.gather(*aufgaben_suche)

        alle_stellenangebote = []
        gesehene_refnrs = set()

        for suchergebnis in alle_suchergebnisse:
            if "ergebnisliste" in suchergebnis:
                for job in suchergebnis["ergebnisliste"]:
                    if job["referenznummer"] not in gesehene_refnrs:
                        alle_stellenangebote.append(job)
                        gesehene_refnrs.add(job["referenznummer"])

        logger.info(f"{len(alle_stellenangebote)} Jobs gefunden (nach Duplikat-Filterung)")
                
        aufgaben = [hole_jobdetails(client, job["referenznummer"], headers) for job in alle_stellenangebote]
        alle_details = await asyncio.gather(*aufgaben)
       
    job_texte = []

    for basis, detail in zip(alle_stellenangebote, alle_details):
            beschreibung = detail.get("stellenangebotsBeschreibung", "")
            beruf = basis.get("hauptberuf", "")
            titel = basis.get("stellenangebotsTitel", "")
            firma = basis.get("firma", "")
            text_string = f"{beruf} - {titel}, bei {firma}, {beschreibung}"
            job_texte.append(text_string)

    rechts_geswipte_jobs = hole_swipe_jobs(profil.nutzer_id, "rechts")
    links_geswipte_jobs = hole_swipe_jobs(profil.nutzer_id, "links")

    profil_embedding = model.encode(profil.profil_text)
    
    aehnlichkeiten = []
    for job_text in job_texte:
        job_embedding = model.encode(job_text)
        basis_wert = cosine_similarity(profil_embedding, job_embedding)

        bonus = durchschnittliche_aehnlichkeit(job_text, rechts_geswipte_jobs)
        malus = durchschnittliche_aehnlichkeit(job_text, links_geswipte_jobs)

        finaler_wert = basis_wert + (bonus * 0.2) - (malus * 0.2)
        aehnlichkeiten.append(finaler_wert)
    
    kombiniert = list(zip(job_texte, aehnlichkeiten))
    sortiert = sorted(kombiniert, key=lambda x: x[1], reverse=True)
    
    ergebnis = []
    for job, wert in sortiert:
        ergebnis.append({"job": job, "aehnlichkeit": float(wert)})
    
    return ergebnis



@app.get("/matches/{nutzer_id}")

def hole_matches(nutzer_id: int):
    return hole_swipe_jobs(nutzer_id, "rechts")

class SwipeAktion(BaseModel):
    job_titel: str
    richtung: str
    nutzer_id: int

@app.post("/swipe")
def swipen(swipeaktion: SwipeAktion):
    neuer_swipe = Swipe(nutzer_id=swipeaktion.nutzer_id, job_titel=swipeaktion.job_titel, richtung=swipeaktion.richtung)
    session.add(neuer_swipe)
    session.commit()
    return {"id": neuer_swipe.id, "job_titel": neuer_swipe.job_titel, "richtung": neuer_swipe.richtung}

@app.get("/swipes/{nutzer_id}")
def hole_swipes(nutzer_id: int):
    alle_swipes = session.query(Swipe).filter(Swipe.nutzer_id == nutzer_id).all()

    ergebnis = []
    for swipe in alle_swipes:
        ergebnis.append({"job_titel": swipe.job_titel, "richtung": swipe.richtung})
    return ergebnis




