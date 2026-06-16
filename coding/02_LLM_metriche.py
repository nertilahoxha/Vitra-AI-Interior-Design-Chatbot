"""
RAG COMPLETO: CHROMA + LLM

    1) riconoscere il prodotto nella domanda (Panton, Wire Chair, ecc.)
    2) interrogare Chroma per ottenere i chunk pertinenti
    3) costruire un prompt "RAG" con i chunk come CONTEXT
    4) chiamare un LLM per generare la risposta

"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from openai import OpenAI
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# IMPORT AGGIUNTIVO PER LE METRICHE (coseno, ecc.)
import numpy as np   # usato per calcolare la similarità coseno tra vettori



# PATH DI PROGETTO


CHROMA_PATH  = Path(r"H:/Il mio Drive/Chabot_Interior Design/coding/chroma_vitra_multi")   
PDF_DIR      = Path(r"H:/Il mio Drive/Chabot_Interior Design/dataset_RAG")          
API_KEY_PATH = Path(r"H:/Il mio Drive/Chabot_Interior Design/coding/openai_key.txt")       



# INIZIALIZZAZIONE LLM CLIENT


# Lettura diretta della chiave (una sola riga, senza spazi extra)
with open(API_KEY_PATH, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

llm_client = OpenAI(api_key=api_key)



# RICONOSCIMENTO PRODOTTO NELLA DOMANDA


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



# COSTRUZIONE DEL PROMPT RAG


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



# RAG + LLM


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

    # DEBUG
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



# COSTRUZIONE DELLA MAPPA PRODOTTI


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



# ============================================================
#           SEZIONE NUOVA: METRICHE DI VALUTAZIONE
# ============================================================

# 1) FUNZIONE DI UTILITÀ: SIMILARITÀ COSENO
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calcola la similarità coseno tra due vettori numpy.
    Valori possibili:
    - 1   → stessa direzione (massima similarità)
    - 0   → ortogonali (nessuna correlazione lineare)
    - -1  → direzioni opposte

    Output: float in [-1, 1].
    """
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8  # evitiamo divisione per 0
    return float(np.dot(vec_a, vec_b) / denom)


def compute_retrieval_metrics_single_query(
    query_embedding: np.ndarray,
    results: Dict[str, Any],
    relevant_ids: List[str]
) -> Dict[str, Any]:
    """
    Calcola le metriche di retrieval per UNA singola query.

    Parametri:
    - query_embedding: embedding numpy della domanda (shape: [dim])
    - results: dizionario restituito da collection.query(...) per una sola query
      Deve contenere:
        * results["ids"][0]         → lista di ID dei documenti/chunk recuperati
        * results["embeddings"][0] → lista di vettori embedding dei chunk
    - relevant_ids: lista di ID (stringhe) considerati "corretti" per questa domanda,
      cioè i chunk che contengono la risposta giusta (ground truth).

    Metriche calcolate:
    - recall_at_k: 1.0 se almeno un ID rilevante è nei top-k, 0.0 altrimenti
    - reciprocal_rank (RR): 1 / (rank del primo hit) se c'è un rilevante, 0.0 altrimenti
      (per una singola query, MRR = RR)
    - cosine_similarities: lista con la similarità coseno query ↔ ciascun chunk recuperato

    Output:
    - dict con chiavi: "recall_at_k", "reciprocal_rank", "cosine_similarities", "retrieved_ids"
    """

    # ID recuperati da Chroma per questa query (lista di stringhe)
    retrieved_ids = results.get("ids", [[]])[0]

    # Embedding dei chunk recuperati (shape: [k, dim])
    # NB: per avere "embeddings" devi chiamare collection.query(..., include=["embeddings", ...])
    retrieved_embs = np.array(results.get("embeddings", [[]])[0])

    # -------------------------
    # Recall@k
    # -------------------------
    # Verifichiamo se almeno un ID recuperato è nella lista dei relevant_ids
    hits_positions = [idx for idx, _id in enumerate(retrieved_ids) if _id in relevant_ids]

    if hits_positions:
        # Almeno un chunk "giusto" trovato nei top-k → recall@k = 1
        recall_at_k = 1.0
    else:
        # Nessun chunk rilevante nei top-k
        recall_at_k = 0.0

    # -------------------------
    # Reciprocal Rank (RR)
    # -------------------------
    if hits_positions:
        # rank è index+1 (perché la lista è 0-based)
        first_hit_rank = hits_positions[0] + 1
        reciprocal_rank = 1.0 / first_hit_rank
    else:
        reciprocal_rank = 0.0

    # -------------------------
    # Similarità coseno query ↔ chunk recuperati
    # -------------------------
    cosine_sims = []
    if retrieved_embs.size > 0:
        for chunk_emb in retrieved_embs:
            sim = cosine_similarity(query_embedding, chunk_emb)
            cosine_sims.append(sim)

    metrics = {
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "cosine_similarities": cosine_sims,
        "retrieved_ids": retrieved_ids,
    }

    return metrics


def evaluate_faithfulness_with_llm(
    question: str,
    context_chunks: List[str],
    answer: str,
    llm_client: OpenAI,
    model_name: str = "gpt-4.1-mini"
) -> str:
    """
    Valuta la *faithfulness* (groundedness) della risposta del LLM
    rispetto ai chunk di CONTEXT usando un LLM come "giudice".

    IDEA:
    - Forniamo all'LLM:
        * la domanda originale
        * il CONTEXT (chunk recuperati dal RAG)
        * la RISPOSTA generata dal chatbot
    - Chiediamo un giudizio strutturato:
        * punteggio da 1 a 5 per FAITHFULNESS
        * breve spiegazione testuale

    Output:
    - stringa con un mini-report (può essere stampata o salvata su file)
      Volendo, puoi far restituire JSON e poi fare parsing.
    """

    context_str = "\n---\n".join(context_chunks)

    eval_prompt = f"""
Sei un valutatore tecnico.

Valuta la risposta del chatbot rispetto al CONTEXT fornito.

QUESTION (DOMANDA UTENTE):
{question}

CONTEXT (CHUNK DEL RAG):
<<<
{context_str}
>>>

ANSWER (RISPOSTA DEL CHATBOT):
<<<
{answer}
>>>

Devi rispondere in questo formato:

FAITHFULNESS: X/5
SPIEGAZIONE: testo breve qui

Dove:
- FAITHFULNESS misura quanto l'ANSWER è fedele al CONTEXT
  (1 = quasi tutto inventato, 5 = completamente supportata dal CONTEXT).
Non inventare nuove informazioni, valuta solo la coerenza con il CONTEXT.
"""

    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.0,
        max_tokens=300
    )

    # Qui otteniamo un breve report testuale, es:
    # "FAITHFULNESS: 4/5\nSPIEGAZIONE: ..."
    report = response.choices[0].message.content.strip()
    return report



# MAIN


if __name__ == "__main__":

    print("[RAG] CHROMA_PATH :", CHROMA_PATH.resolve())
    print("[RAG] PDF_DIR     :", PDF_DIR.resolve())
    print("[RAG] API_KEY_PATH:", API_KEY_PATH.resolve())

    # 1) Chroma client
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection("vitra_multi")

    # 2) embedder (stesso dello STEP 1)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 3) Mappa prodotti
    product_map = build_product_name_map(PDF_DIR)
    print("[RAG] Mappa prodotti:", product_map)

    # 4) Test
    question = "I would like dimensions, colors and material of panton chair. i would like to know mail info contact vitra. Answewr in english."

    # Risposta generata dal RAG (come prima)
    answer = rag_answer_chroma(
        question=question,
        collection=collection,
        embedder=embedder,
        product_map=product_map,
        language="IT",
        k=5
    )

    print("\n=== RISPOSTA DEL CHATBOT (RAG + LLM) ===\n")
    print(answer)

    # ============================================================
    #      ESEMPIO DI UTILIZZO DELLE METRICHE DI RETRIEVAL
    # ============================================================

    # 1) Rifacciamo la query a Chroma ma chiedendo anche:
    #    - embeddings (per la similarità coseno)
    #    - documents e metadatas (per ispezione)
    # NB: gli "ids" vengono sempre restituiti da Chroma, anche se
    #     NON li mettiamo in include.                # <<< MODIFICA (solo commento)
    q_emb_eval = embedder.encode(question, convert_to_numpy=True)

    eval_results = collection.query(
        query_embeddings=[q_emb_eval.tolist()],
        n_results=5,
        include=["embeddings", "documents", "metadatas"]
    )

    # 2) Definire quali ID consideriamo "rilevanti" (ground truth).
    #    VERSIONE GENERALE:
    #    - riconosciamo il prodotto nella domanda (detect_product_in_query)
    #    - prendiamo TUTTI i chunk in Chroma con metadata "product"
    #      uguale a quel prodotto
    #    → questo funziona per QUALSIASI prodotto presente in Chroma,
    #      oggi e in futuro, senza dover scrivere a mano liste di ID.   # <<< MODIFICA

    relevant_ids_example: List[str] = []   # inizialmente vuota          # <<< MODIFICA

    # Proviamo a riconoscere il prodotto nella domanda
    eval_product_id = detect_product_in_query(question, product_map)     # <<< MODIFICA
    if eval_product_id:
        eval_product_name = product_map[eval_product_id]
        print(f"[METRICHE] Valutazione retrieval per prodotto: {eval_product_name}")

        # Prendiamo TUTTI gli id dei chunk in Chroma che appartengono a questo prodotto
        # usando il metadata "product". Questo rende la metrica generica:
        # se domani aggiungi un nuovo prodotto con metadata coerente,
        # verrà automaticamente incluso.                                 # <<< MODIFICA
        relevant_docs = collection.get(
            where={"product": eval_product_name},
            include=[]  # non ci servono documenti/embeddings qui
        )

        # Qui otteniamo una lista piatta di id (non per-query)
        relevant_ids_example = relevant_docs.get("ids", []) or []

    # Se abbiamo trovato almeno un chunk "rilevante" per il prodotto,
    # calcoliamo le metriche; altrimenti stampiamo un messaggio esplicativo. # <<< MODIFICA
    if relevant_ids_example:
        retrieval_metrics = compute_retrieval_metrics_single_query(
            query_embedding=q_emb_eval,
            results=eval_results,
            relevant_ids=relevant_ids_example
        )

        print("\n=== METRICHE DI RETRIEVAL (SINGOLA QUERY) ===")
        print(f"Recall@k        : {retrieval_metrics['recall_at_k']:.3f}")
        print(f"Reciprocal Rank : {retrieval_metrics['reciprocal_rank']:.3f}")
        print("Cosine similarities (query ↔ ogni chunk):")
        for i, (chunk_id, sim) in enumerate(
            zip(retrieval_metrics["retrieved_ids"], retrieval_metrics["cosine_similarities"]),
            start=1
        ):
            print(f"  #{i}  id={chunk_id}   cos_sim={sim:.3f}")
    else:
        print("\n[METRICHE] Nessun relevant_id trovato per questa domanda/prodotto: "
              "controlla che il metadata 'product' sia coerente oppure "
              "passa una lista custom di 'relevant_ids_example' se vuoi "
              "una ground truth più specifica.")                          # <<< MODIFICA

    # ============================================================
    #      ESEMPIO DI METRICA DI GENERAZIONE: FAITHFULNESS
    # ============================================================

    # Usiamo i documenti/chunk che abbiamo appena recuperato per la valutazione
    context_chunks_eval: List[str] = eval_results["documents"][0]

    faithfulness_report = evaluate_faithfulness_with_llm(
        question=question,
        context_chunks=context_chunks_eval,
        answer=answer,
        llm_client=llm_client,
        model_name="gpt-4.1-mini"
    )

    print("\n=== VALUTAZIONE DI FAITHFULNESS (LLM-AS-A-JUDGE) ===\n")
    print(faithfulness_report)
