"""
OPTIONAL STEP — VISUALIZZAZIONE UMAP DEGLI EMBEDDINGS DA CHROMA

Prerequisito:
- Hai già eseguito lo script STEP 1 che:
    * legge i PDF
    * crea i chunk
    * calcola gli embedding con SBERT
    * salva tutto in una collection Chroma persistente: `vitra_multi`
"""

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from pathlib import Path

import chromadb
from chromadb.config import Settings
from collections import Counter   # import per contare i chunk per sezione

# Path dove Chroma ha salvato la collection (deve coincidere con STEP 1)
CHROMA_PATH = Path("coding/chroma_vitra_multi")
CHROMA_COLLECTION_NAME = "vitra_multi"


def plot_umap_embeddings(chroma_path: Path, collection_name: str):
    """
    Recupera tutti gli embedding da Chroma e produce
    una visualizzazione UMAP in 2D salvata come PNG.
    """

    print("\n== STEP EXTRA: UMAP sugli embeddings ==")

    # 1) Carichiamo Chroma
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # Controllo che la collection esista
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        raise ValueError(
            f"La collection '{collection_name}' non esiste in {chroma_path}. "
            f"Assicurati di aver eseguito prima lo script che costruisce l'indice."
        )

    collection = client.get_collection(collection_name)

    # 2) Recuperiamo TUTTO (batching automatico Chroma)
    print("Recupero embeddings da Chroma...")
    results = collection.get(include=["embeddings", "metadatas"])
    X = np.array(results["embeddings"])

    if X.size == 0:
        raise ValueError("Nessun embedding trovato nella collection. Indice vuoto?")

    print(f"Embedding shape: {X.shape}")

    # 3) Riduzione dimensionale UMAP (2D)
    print("Calcolo UMAP (potrebbe richiedere qualche secondo)...")
    umap_model = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    X_umap = umap_model.fit_transform(X)

    # 4) Colore per sezione (materials/colours/dimensions/other/fabrics)
    sections = [m.get("section", "unknown") for m in results["metadatas"]]
    unique_sections = list(set(sections))
    color_map = {sec: i for i, sec in enumerate(unique_sections)}
    colors = [color_map[s] for s in sections]

    # conteggio chunk per sezione
    counts = Counter(sections)

    # legenda più leggibile con spiegazione
    SECTION_DISPLAY_NAMES = {
        "materials":  "materials — materiali, componenti",
        "dimensions": "dimensions — misure, H/W/D, mm/cm",
        "colours":    "colours — palette colori, finiture",
        "use":        "use — ambienti / indoor-outdoor",
        "fabrics":    "fabrics — tessuti, rivestimenti",
        "other":      "other — descrizioni generiche / non classificate",
        "unknown":    "unknown — senza metadati"
    }

    # 5) Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=colors,
        cmap="tab10",
        alpha=0.7,
        s=18,
    )
    plt.title("UMAP — Embeddings SBERT (MiniLM) dei chunk PDF")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # ----- LEGENDA DETTAGLIATA -----
    handles = []
    labels = []

    for sec in sorted(unique_sections):
        color_val = scatter.cmap(scatter.norm(color_map[sec]))

        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                color=color_val,
            )
        )

        nice_name = SECTION_DISPLAY_NAMES.get(sec, sec)
        labels.append(f"{nice_name} (n={counts[sec]})")

    plt.legend(
        handles,
        labels,
        title="Sezione semantica (chunk)",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    # -------------------------------------------

    # 6) Salvataggio PNG
    out_path = chroma_path / "umap_embeddings.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"UMAP salvato in: {out_path}")


if __name__ == "__main__":
    print("== UMAP sugli embeddings Chroma ==")
    print(f"Chroma path: {CHROMA_PATH}")
    print(f"Collection:  {CHROMA_COLLECTION_NAME}")

    plot_umap_embeddings(CHROMA_PATH, CHROMA_COLLECTION_NAME)

    print("\nTutto fatto.")
