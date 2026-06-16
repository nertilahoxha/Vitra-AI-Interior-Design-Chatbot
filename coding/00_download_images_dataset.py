import os
import time
import unicodedata
# MODIFICA: la libreria è stata rinominata. Usa 'ddgs' invece di 'duckduckgo_search'.
from ddgs import DDGS
# MODIFICA: per intercettare il rate limit in modo pulito.
try:
    from ddgs.exceptions import RatelimitException
except Exception:
    RatelimitException = Exception  # fallback se la versione non espone l'eccezione

import requests


# ---------- CONFIGURAZIONE ----------

OUTPUT_DIR = r"dataset"                                     
IMAGES_PER_OBJECT = 300                                         # numero di immagini per riga
TIMEOUT = 8                                                     # timeout (sec) per il download
DELAY_BETWEEN_REQUESTS = 0.5                                    # pausa tra un'immagine e l'altra
DELAY_BETWEEN_QUERIES = 5                                       # pausa tra una query e la successiva

QUERY = [
    #"Panton Chair Classic vitra",
    #"Wire Chair DKL Charles & Ray Eames vitra",
    #"TIP TON Edward Barber & Jay Osgerby vitra",
    "APC Jasper Morrison vitra",
    #"Mynt Erwan Bouroullec vitra",
    #"Mikado Side Chair vitra",
    #"Eames Plastic Side Chair RE DSW vitra",
    #"HAL Jasper Morrison vitra",
    #"Standard Jean Prouvé vitra",
    #"Plywood Group DCW Charles & Ray Eames vitra",


]
# ------------------------------------

# Header HTTP "da browser" per ridurre 403 e blocchi
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def safe_folder_name(text: str) -> str:
    """
    Converte un nome qualsiasi in un nome di cartella "pulito":
    - rimuove accenti
    - sostituisce spazi con underscore
    - toglie caratteri strani
    """
    # Normalizza
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # spazi to underscore
    text = text.strip().replace(" ", "_")

    # Tiene solo lettere, numeri, underscore e trattini
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    text = "".join(c for c in text if c in allowed)

    if not text:
        text = "object"

    return text


def download_images_for_query(query: str, max_images: int, output_dir: str):
    
    #Cerca immagini con DDGS per la query data e ne scarica fino a max_images
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n Query: {query}")
    print(f"   Cartella: {output_dir}")
    folder_name = os.path.basename(output_dir)

    count = 0

    # MODIFICA: nuova firma di DDGS.images()
    # - parametro 'query' (non 'keywords')
    # - 'backend="auto"' per ridurre i 403 scegliendo provider alternativi
    # - 'region="it-it"' facoltativo
    # - 'images()' ora restituisce una LISTA di dict (non più un generatore)
    try:
        results = DDGS(timeout=10).images(
            query=query,
            region="it-it",
            safesearch="off",
            max_results=max_images,   # chiedo quel che mi serve
            backend="auto"
        )
    except RatelimitException as e:
        print(f"  Rate limit rilevato: {e}. Attendo 30s e riprovo una volta...")
        time.sleep(30)
        try:
            results = DDGS(timeout=10).images(
                query=query,
                region="it-it",
                safesearch="off",
                max_results=max_images,
                backend="auto"
            )
        except Exception as e2:
            print(f"  Secondo tentativo fallito per '{query}': {e2}")
            return
    except Exception as e:
        print(f"  Errore durante la ricerca immagini per '{query}': {e}")
        return

    # MODIFICA: 'results' è una lista; iterazione direttamente su di essa
    for r in results:
        if count >= max_images:
            break

        img_url = r.get("image")
        if not img_url:
            continue

        filename = os.path.join(output_dir, f"{folder_name}_{count:03d}.jpg")

        try:
            print(f"  Scarico immagine {count+1}/{max_images} da {img_url}")
            resp = requests.get(img_url, timeout=TIMEOUT, headers=HEADERS)
            if resp.status_code == 200 and resp.content:
                with open(filename, "wb") as f:
                    f.write(resp.content)
                count += 1
            else:
                print(f"  Risposta non valida (status {resp.status_code})")
        except Exception as e:
            print(f"  Errore scaricando {img_url}: {e}")

        # Pausa server
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f" Scaricate {count} immagini per '{query}'")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    queries = QUERY

    print(f" Identificate {len(queries)} query.")

    # Per ogni query, crea sottocartella e download immagini
    for q in queries:
        folder_name = safe_folder_name(q)
        object_dir = os.path.join(OUTPUT_DIR, folder_name)
        download_images_for_query(q, IMAGES_PER_OBJECT, object_dir)

        # Pausa query per rate limit
        time.sleep(DELAY_BETWEEN_QUERIES)

    print("\n Dataset scaricato.")


if __name__ == "__main__":
    main()
