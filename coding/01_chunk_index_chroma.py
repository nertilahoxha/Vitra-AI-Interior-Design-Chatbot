"""
STEP 1 — DALLE SCHEDE PDF A CHROMA (PIPELINE COMPLETA)

Obiettivo pratico:
- Leggere TUTTI i PDF in `dataset_RAG`
- Estrarre il testo, spezzarlo in chunk quasi-semantici
- Aggiungere metadati:
    * product (nome prodotto)
    * language (EN/IT, dal filename)
    * section (dimensions, materials, colours, use, other)
- Calcolare gli embedding con SBERT (MiniLM)
- Salvare tutto in una collection Chroma persistente: `vitra_multi`
"""

import os
import re
from pathlib import Path
from typing import List

import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# PATH DI PROGETTO


PDF_DIR = Path(r"dataset_RAG")              
CHROMA_PATH = Path(r"coding/chroma_vitra_multi")

# Nome della collection Chroma
CHROMA_COLLECTION_NAME = "vitra_multi"

# Modello di embedding (lo stesso usato in app Streamlit)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



# UTILITY: PARSING FILENAMES: product, language


def parse_pdf_filename(filename):
    """
    Esempi di filename attesi:
    - Factbook Panton Chair Classic-EN.pdf
    - Factsheet Eames Fiberglass Armchair-IT.pdf

    Ritorna:
        product_name (stringa pulita) es. "Panton Chair Classic"
        lang         "EN" | "IT"
    oppure (None, None) se non riconosciuto.
    """
    base_name = os.path.splitext(filename)[0]
    print(base_name)

    # Controllo lingua
    if base_name.upper().endswith("-EN"):
        lang = "EN"
    elif base_name.upper().endswith("-IT"):
        lang = "IT"
    else:
        return None, None

    # togliamo il suffisso "-EN"/"-IT" (3 caratteri: "-", "E", "N")
    base_name = base_name[:-3]

    # Rimuovi prefisso Factbook/Factsheet
    for prefix in ["Factbook", "Factsheet"]:
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):].strip()

    # Pulizia spazi multipli
    product_name = " ".join(base_name.split())
    print(f"Filename: {filename} → Product: {product_name}, Language: {lang}")
    return product_name, lang



# 2) CARICAMENTO PDF - DICTIONARY product


all_pdfs_dict = {}

for filename in os.listdir(PDF_DIR):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = PDF_DIR / filename

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()  # lista di Document (uno per pagina)

    full_text = "\n".join(d.page_content for d in docs)

    product_name, lang = parse_pdf_filename(filename)
    if product_name is None:
        continue

    if product_name not in all_pdfs_dict:
        all_pdfs_dict[product_name] = {}

    all_pdfs_dict[product_name][lang] = full_text

print("Dizionario PDF creato correttamente!")
print(f"PDF caricati (prodotti unici): {len(all_pdfs_dict)}")



# 3) DA FULL TEXT A CHUNK


documents: List[Document] = []
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

for product, langs in all_pdfs_dict.items():
    for lang, text in langs.items():
        # normalizziamo gli spazi (per evitare spezzettamenti strani)
        text_new = re.sub(r"\s+", " ", text)
        chunks = splitter.split_text(text_new)

        for chunk in chunks:
            metadata = {
                "product": product,
                "language": lang,
                # "section" lo aggiungiamo dopo con euristiche
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

print(f"Totale chunk creati da tutti i PDF: {len(documents)}")



SECTION_KEYWORDS = {
    # -----------------------------
    # DIMENSIONS — misure tecniche
    # -----------------------------
    "dimensions": [
        "dimension", "dimensions", "size", "sizes", "depth", "height", "width",
        "misura", "misure", "altezza", "larghezza", "profondità",
        "mm", "cm", "inch", "inches",
        "seat height", "seat depth", "seat width",
        "overall height", "overall width", "overall depth",
        "h.", "w.", "d.",
        "stackable", "stackability",       # spesso nel blocco misure
        "weight", "peso",
    ],

    # --------------------------------------
    # MATERIALS — scocca, struttura, materiali
    # --------------------------------------
    "materials": [
        "material", "materials", "materiale", "materiali",
        "shell", "structure", "frame", "base",
        "plastic", "polypropylene", "pp",
        "fibreglass", "fiberglass", "fibra di vetro",
        "wood", "legno", "oak", "beech", "plywood",
        "steel", "metal", "metallo", "aluminium", "aluminum",
        "chrome", "chromed", "powder-coated", "verniciato",
        "polyurethane", "foam", "schiuma",
        "padding", "imbottitura",
        "glides", "feet", "seduta", "scocca", "struttura",
    ],

    # ------------------------------------------------
    # FABRICS — rivestimenti, tessuti, pelli (nuova)
    # ------------------------------------------------
    "fabrics": [
        "fabric", "fabrics", "textile", "textiles",
        "upholstery", "upholstered", "cover", "covers",
        "rivestimento", "rivestimenti", "tessuto", "tessuti",
        "leather", "pelle", "premium leather", "fabric options",
        "seat cover", "removable cover",
        "kvadrat", "camira", "vitra fabric",
    ],

    # ----------------------------------------
    # COLOURS — palette, finiture, varianti
    # ----------------------------------------
    "colours": [
        "colour", "colors", "colours", "colore", "colori",
        "finish", "finishes", "finiture",
        "glossy", "matte", "matt",
        "RAL",                       # codice colori industriali
        "stained", "tinted", "lacquered", "pigmented",
        "palette", "tones", "shades",
    ],

    # -----------------------------------------
    # USE — contesti d’uso, indoor/outdoor, ecc.
    # -----------------------------------------
    "use": [
        "use", "usage", "application", "applications",
        "indoor", "outdoor", "esterno", "interno",
        "home", "ufficio", "office", "workspace",
        "contract", "hospitality", "restaurant",
        "public spaces", "educational", "conference",
    ],
}



for doc in documents:
    text_low = doc.page_content.lower()
    section = "other"

    for sec_name, keywords in SECTION_KEYWORDS.items():
        if any(kw in text_low for kw in keywords):
            section = sec_name
            break

    doc.metadata["section"] = section



# EMBEDDING + INDICIZZAZIONE IN CHROMA


def build_chroma_index(docs: List[Document]):
    """
    Prende la lista di Document (chunk) e:
    - calcola gli embedding con SentenceTransformers
    - li inserisce in una collection Chroma persistente.

    Usiamo Chroma "raw" (chromadb), non LangChain wrapper.
    """
    # 1) Inizializziamo modello di embedding
    print(f"Carico modello di embedding: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 2) Inizializziamo Chroma
    print(f"Inizializzo Chroma in: {CHROMA_PATH}")
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),  # MODIFICA: Path relativo → cast a stringa
        settings=Settings(anonymized_telemetry=False),
    )

    # Se la collection esiste già, la droppiamo per ricostruirla pulita
    existing_collections = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing_collections:
        print(f"Collection '{CHROMA_COLLECTION_NAME}' esiste già → la elimino.")
        client.delete_collection(CHROMA_COLLECTION_NAME)

    collection = client.create_collection(CHROMA_COLLECTION_NAME)

    # 3) Prepariamo dati per l'add
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    ids = [f"chunk_{i}" for i in range(len(docs))]

    # 4) Calcoliamo gli embedding a batch
    batch_size = 64
    for start in tqdm.tqdm(range(0, len(texts), batch_size), desc="Indicizzazione in Chroma"):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_metas = metadatas[start:end]
        batch_ids = ids[start:end]

        embeddings = embedder.encode(
            batch_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        collection.add(
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=embeddings.tolist(),
            ids=batch_ids,
        )

    print("Indicizzazione completata")
    print(f"Collection: {CHROMA_COLLECTION_NAME}")
    print(f"Percorso Chroma (relativo): {CHROMA_PATH}")



# MAIN


if __name__ == "__main__":
    print("== STEP 1: Corpus chunk dai PDF ==")
    print(f"Cartella PDF (relativa): {PDF_DIR}")
    print(f"Numero totale di chunk: {len(documents)}")

    print("\n== STEP 2: Indicizzazione in Chroma con embedding SBERT ==")
    build_chroma_index(documents)

    print("\nTutto fatto.")

