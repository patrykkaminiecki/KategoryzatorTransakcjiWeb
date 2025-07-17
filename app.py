import pandas as pd
import io, re
import streamlit as st
from pathlib import Path
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ------------------------
# 1) DEFINICJA KATEGORII (zaktualizowana)
# ------------------------
CATEGORIES = {
    'Przychody': ['Patryk', 'Jolka', 'Świadczenia', 'Inne'],
    'Rachunki': ['Prąd', 'Gaz', 'Woda', 'Odpady', 'Internet', 'Telefon',
                 'Subskrypcje', 'Przedszkole', 'Żłobek', 'Podatki'],
    'Transport': ['Paliwo', 'Ubezpieczenie', 'Parking', 'Przeglądy'],
    'Kredyty': ['Hipoteka', 'Samochód', 'TV+Dyson', 'Gmina Kolbudy'],
    'Jedzenie': ['Zakupy Spożywcze'],
    'Zdrowie': ['Apteka', 'Lekarz', 'Kosmetyki', 'Fryzjer'],
    'Odzież': ['Ubrania', 'Buty'],
    'Dom i Ogród': ['Dom', 'Ogród', 'Zwierzęta'],
    'Inne': ['Prezenty', 'Rozrywka', 'Hobby', 'Edukacja'],
    'Oszczędności': [
        'Poduszka bezpieczeństwa - Lokata',
        'Poduszka bezpieczeństwa - Konto',
        'Poduszka bezpieczeństwa - Obligacje',
        'Fundusz celowy',
        'Inwestycje'
    ],
    'Nadpłata Długów': ['Hipoteka', 'Samochód', 'TV+Dyson', 'Gmina Kolbudy'],
    'Wakacje': ['Wakacje'],
    'Gotówka': ['Wpłata', 'Wypłata']
}
ASSIGNMENTS_FILE = Path("assignments.csv")
CATEGORY_PAIRS = [f"{cat} — {sub}" for cat, subs in CATEGORIES.items() for sub in subs]

# 2) EMBEDDINGI, 3) CATEGORIZER, 4) load_bank_csv  -- tak jak wcześniej (bez zmian) --

@st.cache_resource
def get_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def get_pair_embs():
    return get_embed_model().encode(CATEGORY_PAIRS, convert_to_numpy=True)

EMBED_MODEL = get_embed_model()
PAIR_EMBS = get_pair_embs()

def clean_desc(s):
    text = str(s).replace("'", "").replace('"', "")
    return re.sub(r'\s+', ' ', text).strip()

class Categorizer:
    def __init__(self):
        self.map = {}
        if ASSIGNMENTS_FILE.exists():
            df = pd.read_csv(ASSIGNMENTS_FILE).drop_duplicates('description', keep='last')
            for _, row in df.iterrows():
                self.map[clean_desc(row['description'])] = (row['category'], row['subcategory'])

    def suggest(self, key: str, amount: float):
        kc = clean_desc(key)
        if kc in self.map and self.map[kc][0]:
            return self.map[kc]
        emb = EMBED_MODEL.encode([key], convert_to_numpy=True)
        sims = cosine_similarity(emb, PAIR_EMBS)[0]
        idx, score = int(np.argmax(sims)), sims.max()
        if score > 0.5:
            return tuple(CATEGORY_PAIRS[idx].split(" — "))
        return ('Przychody','Inne') if amount >= 0 else ('Inne', CATEGORIES['Inne'][0])

    def assign(self, key: str, cat: str, sub: str):
        kc = clean_desc(key)
        if not kc: return
        self.map[kc] = (cat, sub)
        ASSIGNMENTS_FILE.parent.mkdir(exist_ok=True)
        pd.DataFrame([{"description": k,"category": c,"subcategory": s}
                      for k,(c,s) in self.map.items()])\
          .to_csv(ASSIGNMENTS_FILE, index=False)

@st.cache_data
def load_bank_csv(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    for enc, sep in [('cp1250',';'),('utf-8',';'),('utf-8',',')]:
        try:
            lines = raw.decode(enc, errors='ignore').splitlines()
            idx = next(i for i,l in enumerate(lines) if 'Data' in l and 'Kwota' in l)
            return pd.read_csv(io.StringIO("\n".join(lines[idx:])), sep=sep, decimal=',')
        except:
            pass
    raise ValueError("Nie udało się wczytać pliku CSV.")

def main():
    st.set_page_config(page_title="Kategoryzator Finansowy", layout="wide")
    # ... CSS i header ...
    st.title("🗂 Kategoryzator transakcji + Raporty")
    cat = Categorizer()

    # wczytanie pliku, filtrowanie, bulk-assign, tabela itp. – tak jak poprzednio

    # …. 
    # Po sekcji: Raport tekstowy:
    # ====================================================================
    st.markdown("## 📊 Raport: ilość i suma wg kategorii")
    # … generujemy `total` i `grouped` jak poprzednio …
    # (kod identyczny jak wcześniej)

    # -------------------------------------------------------------------------------------------------
    # NOWA SEKCJA: Oszczędności YTD (obejmuje cały rok, niezależnie od filtra poniżej)
    # -------------------------------------------------------------------------------------------------
    st.markdown("## 💰 Oszczędności YTD")
    # Używamy pełnego `edited` z Date - nie filtrujemy po sidebar
    edited['Date'] = pd.to_datetime(edited['Date'], errors='coerce')
    year = datetime.now().year
    ytd = edited[(edited['category']=='Oszczędności') & (edited['Date'].dt.year == year)]
    # Agregacja główna
    ytd_total = ytd['Amount'].sum()
    st.markdown(f"**Łącznie Oszczędności w {year}: {ytd_total:,.2f} zł**".replace(",", " "))
    # Podkategorie
    sub_ytd = ytd.groupby('subcategory')['Amount'].sum().reset_index().sort_values('Amount', ascending=False)
    # Expander per podkategoria
    for _, row in sub_ytd.iterrows():
        name = row['subcategory']
        val  = row['Amount']
        pct  = val / ytd_total if ytd_total!=0 else 0
        with st.expander(f"**{name}** – {val:,.2f} zł ({pct:.0%})", expanded=False):
            st.write(f"Podkategoria **{name}**: {val:,.2f} zł ({pct:.0%})")

    # … dalej wykresy kołowe kategorii i podkategorii (jak wcześniej) …

if __name__=="__main__":
    main()
