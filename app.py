import pandas as pd
import io, re
import streamlit as st
from pathlib import Path
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------------
# 1) DEFINICJA KATEGORII
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
    'Oszczędności': ['Poduszka bezpieczeństwa', 'Fundusz celowy', 'Inwestycje', 'Wypłata z oszczędności'],
    'Nadpłata Długów': ['Hipoteka', 'Samochód', 'TV+Dyson', 'Gmina Kolbudy'],
    'Wakacje': ['Wakacje'],
    'Gotówka': ['Wpłata', 'Wypłata']
}
ASSIGNMENTS_FILE = Path("assignments.csv")
CATEGORY_PAIRS = [f"{cat} — {sub}" for cat, subs in CATEGORIES.items() for sub in subs]

# --------------------------------------------------
# 2) EMBEDDINGI
# --------------------------------------------------
@st.cache_resource
def get_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def get_pair_embs():
    return get_embed_model().encode(CATEGORY_PAIRS, convert_to_numpy=True)

EMBED_MODEL = get_embed_model()
PAIR_EMBS = get_pair_embs()

# ------------------------------------
# 3) CATEGORIZER
# ------------------------------------
def clean_desc(s):
    text = str(s).replace("'", "").replace('"', "")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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
        if not kc:
            return
        self.map[kc] = (cat, sub)
        # utwórz katalog jeśli potrzeba
        ASSIGNMENTS_FILE.parent.mkdir(exist_ok=True)
        # natychmiast zapis do pliku
        pd.DataFrame([
            {"description": k, "category": c, "subcategory": s}
            for k,(c,s) in self.map.items()
        ]).to_csv(ASSIGNMENTS_FILE, index=False)

# ------------------------------------
# 4) Wczytanie CSV z banku
# ------------------------------------
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

# ------------------------------------
# 5) GŁÓWNA FUNKCJA
# ------------------------------------
def main():
    st.title("🗂 Kategoryzator transakcji + Raporty")
    cat = Categorizer()  # zawsze wczytujemy najnowsze assignments.csv

    # 5.1) sidebar: plik + filtr dat
    st.sidebar.header("Filtr dat")
    uploaded = st.sidebar.file_uploader("Wybierz plik CSV", type="csv")
    if not uploaded:
        st.sidebar.info("Wczytaj plik CSV, aby rozpocząć."); return
    try:
        df_raw = load_bank_csv(uploaded)
    except Exception as e:
        st.error(str(e)); return

    # przygotowanie DataFrame
    cols = [c.strip() for c in df_raw.columns if c is not None]
    df = df_raw.copy(); df.columns = cols
    df = df.rename(columns={
        'Data transakcji':'Date','Dane kontrahenta':'Description','Tytuł':'Tytuł',
        'Nr rachunku':'Nr rachunku','Kwota transakcji (waluta rachunku)':'Amount',
        'Kwota blokady/zwolnienie blokady':'Kwota blokady'
    })[['Date','Description','Tytuł','Nr rachunku','Amount','Kwota blokady']]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].notna()]

    # 5.2) filtr dat
    mode = st.sidebar.radio("Tryb filtrowania", ["Zakres dat","Pełny miesiąc"])
    if mode == "Zakres dat":
        mn, mx = df['Date'].min(), df['Date'].max()
        start, end = st.sidebar.date_input("Zakres dat", [mn, mx], min_value=mn, max_value=mx)
        start, end = pd.to_datetime(start), pd.to_datetime(end)
        df = df[(df['Date']>=start)&(df['Date']<=end)]
    else:
        yrs = sorted(df['Date'].dt.year.unique())
        months = {1:'Styczeń',2:'Luty',3:'Marzec',4:'Kwiecień',5:'Maj',
                  6:'Czerwiec',7:'Lipiec',8:'Sierpień',9:'Wrzesień',
                  10:'Październik',11:'Listopad',12:'Grudzień'}
        y = st.sidebar.selectbox("Rok", yrs, index=len(yrs)-1)
        mname = st.sidebar.selectbox("Miesiąc", list(months.values()), index=6)
        m = {v:k for k,v in months.items()}[mname]
        df = df[(df['Date'].dt.year==y)&(df['Date'].dt.month==m)]

    # 5.3) Bulk‑assign
    df['key'] = (df['Nr rachunku'].astype(str).fillna('') + '|' + df['Description']).map(clean_desc)
    groups = df.groupby('key').groups.values()
    st.markdown("#### Krok 1: Przypisz kategorie grupom")
    for idxs in groups:
        key = df.loc[idxs[0],'key']
        if key in cat.map and cat.map[key][0]:
            continue
        amt = df.loc[idxs[0],'Amount']
        st.write(f"**{key}** – {amt:.2f} PLN")
        sugg = cat.suggest(key, amt)
        sel_cat = st.selectbox("Kategoria", list(CATEGORIES.keys()),
                               index=list(CATEGORIES.keys()).index(sugg[0]), key=f"cat_{key}")
        opts = CATEGORIES[sel_cat]
        default = opts.index(sugg[1]) if sugg[1] in opts else 0
        sel_sub = st.selectbox("Podkategoria", opts, index=default, key=f"sub_{key}")
        cat.assign(key, sel_cat, sel_sub)

    st.markdown("---")
    st.success("Krok 1: zakończony – assignments.csv zaktualizowany.")

    # 5.4) Finalna tabela
    df['category']    = df['key'].map(lambda k: cat.map.get(k,("", ""))[0])
    df['subcategory'] = df['key'].map(lambda k: cat.map.get(k,("", ""))[1])
    final = df[['Date','Description','Tytuł','Amount','Kwota blokady','category','subcategory']]

    # 5.5) Edytowalna tabela
    edited = st.data_editor(final,
        column_config={
            'Date': st.column_config.Column("Data"),
            'Description': st.column_config.Column("Opis"),
            'Tytuł': st.column_config.Column("Tytuł"),
            'Amount': st.column_config.NumberColumn("Kwota", format="%.2f"),
            'Kwota blokady': st.column_config.NumberColumn("Blokada", format="%.2f"),
            'category': st.column_config.SelectboxColumn("Kategoria", options=list(CATEGORIES.keys())),
            'subcategory': st.column_config.SelectboxColumn("Podkategoria",
                             options=[s for subs in CATEGORIES.values() for s in subs])
        },
        hide_index=True, use_container_width=True
    )

    # 5.6) Zapis ręcznych zmian
    if st.button("💾 Zapisz zmiany do assignments.csv"):
        # klucze w tej samej kolejności
        keys_list = df['key'].tolist()
        for idx, row in enumerate(edited.itertuples(index=False)):
            key = keys_list[idx]
            cat.assign(key, row.category, row.subcategory)
        st.success("Zapisano assignments.csv")

    # 5.7) Raport z edited
    @st.cache_data
    def get_report_tables(df_final):
        grp = df_final.groupby(['category','subcategory'])['Amount'].agg(['count','sum']).reset_index()
        grp = grp[grp['count']>0]
        tot = grp.groupby('category').agg({'count':'sum','sum':'sum'}).reset_index()
        tot = pd.concat([tot[tot['category']=='Przychody'],
                         tot[tot['category']!='Przychody'].sort_values('category')],
                        ignore_index=True)
        return grp, tot

    grouped, total = get_report_tables(edited)

   st.markdown("## 📊 Raport: ilość i suma wg kategorii")

    def fmt(val):
        return f"{abs(val):,.2f}".replace(",", " ").replace(".", ",")
    
    for _, row in total.iterrows():
        cat = row['category']
        count = row['count']
        total_sum = fmt(row['sum'])
    
        # 🔸 Nagłówek kategorii: nazwa większa i pogrubiona, reszta normalnie
        expander_label = f"<span style='font-size:18px'><strong>{cat}</strong></span> ({count}) – {total_sum}"
    
        subs = grouped[grouped['category'] == cat].copy()
        subs['subcategory'] = subs['subcategory'].fillna('').replace('', 'brak podkategorii')
    
        with st.expander(expander_label, expanded=False):
            for _, sub in subs.iterrows():
                sub_cat = sub['subcategory']
                sub_count = sub['count']
                sub_sum = fmt(sub['sum'])
    
                # 🔹 Wiersz podkategorii: pogrubiona nazwa podkategorii, reszta normalnie
                st.markdown(
                    f"<span style='font-size:16px'>• <strong>{sub_cat}</strong> ({sub_count}) – {sub_sum}</span>",
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
