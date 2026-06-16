import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# ==============================
# NUOVI IMPORT PER RAG + LLM
# ==============================
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

# ==============================
# SEZIONE DETECTION (INVARIATA)
# ==============================

MODEL_PATH = r"H:/Il mio Drive/Chabot_Interior Design/coding/YOLO11/runs/detect/train/weights/best_yolo11.pt"
model = YOLO(MODEL_PATH)

st.title("Chatbot Vitra")
st.write("Carica un’immagine e il modello farà l'analisi!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "image" in msg:
            st.image(msg["image"])

if "last_image_name" not in st.session_state:
    st.session_state.last_image_name = None

# ==============================
# NUOVA SEZIONE: CONFIG RAG + LLM
# (riuso del codice che hai già scritto)
# ==============================

# Percorsi come nel tuo modulo RAG
ROOT = Path(__file__).resolve().parents[1]
API_KEY_PATH  = ROOT / "coding/openai_key.txt"
CHROMA_PATH   = ROOT / "coding/chroma_vitra_multi"
PDF_DIR       = ROOT / "dataset_RAG"

# Lettura chiave API da file (come nel tuo codice)
with open(API_KEY_PATH, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

llm_client = OpenAI(api_key=api_key)

# ------------------------------
# Funzione per riconoscere il prodotto nella query (COPIATA)
# ------------------------------
def detect_product_in_query(query: str, product_map: Dict[str, str]) -> Optional[str]:
    """
    Cerca se nella domanda l'utente nomina in modo esplicito uno dei prodotti.
    """
    q = query.lower()

    for product_id, product_name in product_map.items():
        words = product_name.lower().split()
        match_count = sum(1 for w in words if w in q)

        if match_count >= max(2, len(words) // 2):
            return product_id

    return None

# ------------------------------
# Costruzione prompt RAG (COPIATA)
# ------------------------------
def build_rag_prompt(question: str, context_chunks: List[str], language: str = "IT") -> str:

    if language.upper() == "IT":
        instructions = (
            "Sei un assistente tecnico per i prodotti Vitra.\n"
            "Rispondi in italiano, in modo chiaro, ordinato e preciso.\n"
            "DEVI usare esclusivamente le informazioni presenti nel CONTEXT.\n"
            "Se l'utente chiede qualcosa che NON è nel CONTEXT, devi dirlo.\n"
        )
    else:
        instructions = (
            "You are a technical assistant for Vitra products.\n"
            "Answer clearly, using ONLY the information in the CONTEXT.\n"
        )

    context_text = "\n---\n".join(context_chunks)

    prompt = (
        f"{instructions}\n\n"
        f"CONTEXT:\n<<<\n{context_text}\n>>>\n\n"
        f"DOMANDA UTENTE:\n{question}\n\n"
        f"RISPOSTA:\n"
    )

    return prompt

# ------------------------------
# RAG + LLM (COPIATA, SOLO RIUSO IN STREAMLIT)
# ------------------------------
def rag_answer_chroma(
    question: str,
    collection,
    embedder,
    product_map: Dict[str, str],
    language: str = "IT",
    k: int = 5
) -> str:

    # 1) Product detection
    product_id = detect_product_in_query(question, product_map)

    if product_id:
        product_name = product_map[product_id]
        print(f"[RAG] Prodotto riconosciuto: {product_name}")
        where_filter: Dict[str, Any] = {"product": product_name}
    else:
        print("[RAG] Nessun prodotto riconosciuto → cerco su tutto il corpus.")
        where_filter = {}

    # 2) Embedding della domanda
    q_emb = embedder.encode(question, convert_to_numpy=True)

    # 3) Query a Chroma
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=k,
        where=where_filter if where_filter else None
    )

    retrieved_chunks = results["documents"][0]
    retrieved_metas   = results["metadatas"][0]

    # DEBUG su console
    print("\n[RAG] CHUNK SELEZIONATI:\n")
    for i, (txt, meta) in enumerate(zip(retrieved_chunks, retrieved_metas), start=1):
        print(f"CHUNK {i}:")
        print(f"  product : {meta.get('product')}")
        print(f"  language: {meta.get('language')}")
        print(f"  section : {meta.get('section')}")
        print("  testo (prime 300 chars):")
        print("  " + txt[:300].replace("\n", " "))
        print("-" * 80)

    # Prompt RAG
    prompt = build_rag_prompt(question, retrieved_chunks, language)

    # Chiamata all'LLM
    response = llm_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=400
    )

    return response.choices[0].message.content.strip()

# ------------------------------
# Mappa prodotti (COPIATA)
# ------------------------------
def build_product_name_map(pdf_dir: Path) -> Dict[str, str]:
    """
    Legge i nomi dei PDF e crea la mappa:
    { product_id_slug: product_human_name }
    """
    mapping = {}

    for pdf_path in pdf_dir.glob("*.pdf"):
        base_name = pdf_path.stem

        lang = None
        if base_name.upper().endswith("-EN"):
            lang = "EN"
            base_name = base_name[:-3]
        elif base_name.upper().endswith("-IT"):
            lang = "IT"
            base_name = base_name[:-3]

        for prefix in ["Factbook", "Factsheet"]:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):].strip()

        product_name = " ".join(base_name.split())
        if not product_name:
            continue

        product_id = product_name.lower().replace(" ", "_")
        mapping[product_id] = product_name

    return mapping

# ------------------------------
# Piccola funzione nuova: guess lingua IT/EN
# (heuristica grezza per scegliere il parametro language)
# ------------------------------
def guess_language(question: str) -> str:
    q = " " + question.lower() + " "
    italian_markers = [" il ", " lo ", " la ", " un ", " una ", " che ", " cosa ", " come ", " perché", " perche", " quindi "]
    if any(m in q for m in italian_markers):
        return "IT"
    return "EN"

# ------------------------------
# Inizializzazione risorse RAG in modo cache-ato
# ------------------------------
@st.cache_resource
def init_rag_resources():
    print("[RAG] CHROMA_PATH :", CHROMA_PATH.resolve())
    print("[RAG] PDF_DIR     :", PDF_DIR.resolve())
    print("[RAG] API_KEY_PATH:", API_KEY_PATH.resolve())

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection("vitra_multi")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    product_map = build_product_name_map(PDF_DIR)
    print("[RAG] Mappa prodotti:", product_map)

    return collection, embedder, product_map

# Carico una sola volta le risorse RAG
collection, embedder, product_map = init_rag_resources()

# ==============================
# INPUT CHAT + UPLOAD IMMAGINE
# ==============================

prompt = st.chat_input("Scrivi qualcosa o carica un’immagine…")
uploaded_image = st.file_uploader("Carica immagine per YOLO", type=["jpg", "jpeg", "png"])

# ==============================
# MODIFICA: gestione prompt UTENTE con RAG+LLM
# (prima c'era solo il messaggio fisso)
# ==============================
if prompt:
    # Salvo il messaggio dell'utente nella history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Stimo la lingua della domanda (IT / EN)
    lang = guess_language(prompt)

    # Chiamo il RAG per generare la risposta
    try:
        answer = rag_answer_chroma(
            question=prompt,
            collection=collection,
            embedder=embedder,
            product_map=product_map,
            language=lang,
            k=5
        )
    except Exception as e:
        # In caso di errore lato backend, mostro un messaggio leggibile
        answer = f"Si è verificato un errore nel modulo RAG/LLM: {e}"

    # Mostro risposta come messaggio chat dell'assistente
    with st.chat_message("assistant"):
        st.write(answer)

    # Salvo anche la risposta in session_state per lo storico
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# ==============================
# SEZIONE DETECTION IMMAGINE (NO RIPETIZIONE)
# ==============================
if uploaded_image:
    # Esegui YOLO solo se è un nuovo file
    if uploaded_image.name != st.session_state.last_image_name:
        st.session_state.last_image_name = uploaded_image.name

        img = Image.open(uploaded_image)

        st.session_state.messages.append({
            "role": "user",
            "content": "Ecco la mia immagine:",
            "image": img
        })

        results = model.predict(img)
        result_img = results[0].plot()

        with st.chat_message("assistant"):
            st.write("Ecco cosa ho trovato nell’immagine!")
            st.image(result_img)

        st.session_state.messages.append({
            "role": "assistant",
            "content": "Ecco cosa ho trovato nell’immagine!",
            "image": result_img
        })
    # Se l’immagine è la stessa di prima → non rifacciamo detection

