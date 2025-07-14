import pandas as pd
import io
import streamlit as st
from pathlib import Path
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import git  # opcjonalnie, jeśli auto‑push

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

# --------------------------------------------------
# 2) PRZYGOTUJ EMBEDDINGI DLA PAR KATEGORIA — PODKATEGORIA
# --------------------------------------------------
CATEGORY_PAIRS = [f"{cat} — {sub}" for cat, subs in CATEGORIES.items() for sub in subs]
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
PAIR_EMBS = EMBED_MODEL.encode(CATEGORY_PAIRS, convert_to_numpy=True)

# ------------------------------------
# 3) KLASA DO ZARZĄDZANIA PRZYPISANIAMI
# ------------------------------------
class Categorizer:
    def __init__(self):
        self.map = {}
        if ASSIGNMENTS_FILE.exists():
            try:
                df = pd.read_csv(ASSIGNMENTS_FILE)
                for _, row in df.iterrows():
                    self.map[str(row['description'])] = (row['category'], row['subcategory'])
            except Exception:
                st.warning("Plik assignments.csv istnieje, ale jest uszkodzony lub pusty.")

    def suggest(self, key: str, amount: float):
        if key in self.map and self.map[key][0]:
            return self.map[key]
        emb = EMBED_MODEL.encode([key], convert_to_numpy=True)
        sims = cosine_similarity(emb, PAIR_EMBS)[0]
        best_idx = int(np.argmax(sims)); best_score = sims[best_idx]
        if best_score > 0.5:
            cat, sub = CATEGORY_PAIRS[best_idx].split(" — ")
            return (cat, sub)
        return ('Przychody','Inne') if amount >= 0 else ('Inne', CATEGORIES['Inne'][0])

    def assign(self, key: str, cat: str, sub: str):
        self.map[key] = (cat, sub)

    def save(self):
        pd.DataFrame([
            {"description": k, "category": c, "subcategory": s}
            for k,(c,s) in self.map.items()
        ]).to_csv(ASSIGNMENTS_FILE, index=False)

# ------------------------------------
# 4) OPCJONALNIE: AUTO‑PUSH DO GITHUB
# ------------------------------------
def auto_git_commit():
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["GITHUB_REPO"]
    repo_url = f"https://{token}@github.com/{repo_name}.git"
    if not Path(".git").exists():
        git.Repo.clone_from(repo_url, ".", branch="main")
    repo = git.Repo(".")
    repo.remotes.origin.set_url(repo_url)
    repo.index.add([str(ASSIGNMENTS_FILE)])
    if repo.is_dirty():
        name,email = st.secrets["GITHUB_AUTHOR"].replace(">","").split(" <")
        repo.index.commit("Automatyczny zapis assignments.csv", author=git.Actor(name,email))
        repo.remotes.origin.push()

# ------------------------------------
# 5) FUNKCJA WCZYTANIA CSV Z BANKU
# ------------------------------------
def load_bank_csv(uploaded)->pd.DataFrame:
    raw=uploaded.getvalue()
    for enc,sep in [('cp1250',';'),('utf-8',';'),('utf-8',',')]:
        try:
            lines=raw.decode(enc,errors='ignore').splitlines()
            idx=next(i for i,l in enumerate(lines) if 'Data' in l and 'Kwota' in l)
            return pd.read_csv(io.StringIO("\n".join(lines[idx:])),sep=sep,decimal=',')
        except:
            pass
    raise ValueError("Nie udało się wczytać pliku CSV.")

# --------------------------
# 6) GŁÓWNA FUNKCJA STREAMLIT
# --------------------------
def main():
    st.title("🗂 Kategoryzator transakcji bankowych + Raporty")

    cat = Categorizer()

    # --- 6.1) Filtry dat w sidebarze ---
    st.sidebar.header("Filtr dat")
    uploaded = st.sidebar.file_uploader("Wybierz plik CSV", type=["csv"])
    if not uploaded:
        st.sidebar.info("Najpierw wczytaj plik CSV.")
        return

    try:
        df_raw = load_bank_csv(uploaded)
    except Exception as e:
        st.error(str(e)); return

    df_raw = df_raw.loc[:,df_raw.columns.notna()]
    df_raw.columns=[c.strip() for c in df_raw.columns]
    df_raw.rename(columns={
        'Data transakcji':'Date','Dane kontrahenta':'Description','Tytuł':'Tytuł',
        'Nr rachunku':'Nr rachunku','Kwota transakcji (waluta rachunku)':'Amount',
        'Kwota blokady/zwolnienie blokady':'Kwota blokady'
    },inplace=True)
    df = df_raw[['Date','Description','Tytuł','Nr rachunku','Amount','Kwota blokady']].copy()
    # Konwersja kolumny 'Date' i usunięcie błędnych wierszy
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].notna()]
    
        # ✅ Filtrowanie po dacie: wybór trybu
    filter_mode = st.sidebar.radio("Wybierz tryb filtrowania", ["Zakres dat", "Pełny miesiąc"])
    
    # 🔹 Przygotuj dane: konwersja kolumny 'Date' i czyszczenie błędnych
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].notna()]  # tylko poprawne daty
    
    if filter_mode == "Zakres dat":
        min_d, max_d = df['Date'].min(), df['Date'].max()
        start, end = st.sidebar.date_input("Zakres dat", [min_d, max_d], min_value=min_d, max_value=max_d)
        mask = (df['Date'] >= pd.to_datetime(start)) & (df['Date'] <= pd.to_datetime(end))
        df = df.loc[mask]
    
    elif filter_mode == "Pełny miesiąc":
        available_years = sorted(df['Date'].dt.year.unique())
        available_months = {
            1: 'Styczeń', 2: 'Luty', 3: 'Marzec', 4: 'Kwiecień',
            5: 'Maj', 6: 'Czerwiec', 7: 'Lipiec', 8: 'Sierpień',
            9: 'Wrzesień', 10: 'Październik', 11: 'Listopad', 12: 'Grudzień'
        }
    
        year_filter = st.sidebar.selectbox("Rok", available_years, index=len(available_years) - 1)
        month_filter = st.sidebar.selectbox("Miesiąc", list(available_months.values()), index=6)  # domyślnie lipiec
    
        rev_months = {v: k for k, v in available_months.items()}
        m = rev_months[month_filter]
        y = year_filter
        df = df[(df['Date'].dt.year == y) & (df['Date'].dt.month == m)]


    # --- 6.2) Bulk‑assign (pozostało bez zmian) ---
    acct_nums = df['Nr rachunku'].dropna().unique().tolist()
    acct_groups = [df.index[df['Nr rachunku']==a].tolist() for a in acct_nums]
    no_acct = df.index[df['Nr rachunku'].isna()].tolist()
    desc_groups = [[i] for i in no_acct]
    groups = acct_groups+desc_groups

    st.markdown("#### Krok 1: Przypisz kategorie grupom")
    for idxs in groups:
        first=idxs[0]
        acct=df.loc[first,'Nr rachunku']; key=str(acct) if pd.notna(acct) else str(df.loc[first,'Description'])
        if key in cat.map and cat.map[key][0]: continue
        descs=[str(x) for x in df.loc[idxs,'Description'].unique()]; titles=[str(x) for x in df.loc[idxs,'Tytuł'].unique()]
        amt=df.loc[first,'Amount']
        st.write(f"**{key}** – {amt:.2f} PLN")
        st.write(f"- Opisy: {', '.join(descs[:3])}{'...' if len(descs)>3 else ''}")
        sugg=cat.suggest(key,amt) or ("","")
        sel_cat=st.selectbox("Kategoria", list(CATEGORIES.keys()),
                             index=list(CATEGORIES.keys()).index(sugg[0]) if sugg[0] in CATEGORIES else 0,
                             key=f"cat_{key}")
        sel_sub=st.selectbox("Podkategoria", CATEGORIES[sel_cat],
                             index=CATEGORIES[sel_cat].index(sugg[1]) if sugg[1] in CATEGORIES.get(sel_cat,[]) else 0,
                             key=f"sub_{key}")
        for i in idxs: cat.assign(key,sel_cat,sel_sub)

    st.markdown("---")
    st.success("Krok 1 zakończony: grupy mają kategorie.")

    # --- 6.3) Finalna tabela ---
    keys_list=[str(r['Nr rachunku']) if pd.notna(r['Nr rachunku']) else str(r['Description'])
               for _,r in df.iterrows()]
    df['category']=[cat.map.get(k,("", ""))[0] for k in keys_list]
    df['subcategory']=[cat.map.get(k,("", ""))[1] for k in keys_list]
    final=df[['Date','Description','Tytuł','Amount','Kwota blokady','category','subcategory']]

    edited=st.data_editor(final,
        column_config={
            'Date':st.column_config.Column("Data"),
            'Description':st.column_config.Column("Opis"),
            'Tytuł':st.column_config.Column("Tytuł"),
            'Amount':st.column_config.NumberColumn("Kwota",format="%.2f"),
            'Kwota blokady':st.column_config.NumberColumn("Blokada",format="%.2f"),
            'category':st.column_config.SelectboxColumn("Kategoria",options=list(CATEGORIES.keys())),
            'subcategory':st.column_config.SelectboxColumn("Podkategoria",
                options=[s for subs in CATEGORIES.values() for s in subs])
        },
        hide_index=True,use_container_width=True
    )

    if st.button("💾 Zapisz zmiany"):
        for idx,row in enumerate(edited.itertuples(index=False)):
            key=keys_list[idx]; cat.assign(key,row.category,row.subcategory)
        cat.save(); st.success("Zapisano assignments.csv")
        try: auto_git_commit(); st.success("Wysłano do GitHuba")
        except: st.warning("Push nieudany")

        # --- 6.4) Raport i wykres ---
    st.markdown("## 📊 Raport: ilość i suma według kategorii")
    
    def format_amount(val):
        return f"{val:,.2f}".replace(",", " ").replace(".","‚").replace("‚", ",")
    
    grouped = final.groupby(['category','subcategory'])['Amount'].agg(['count','sum']).reset_index()
    grouped['formatted'] = grouped.apply(lambda r: f"{r['subcategory']} ({r['count']}) – {format_amount(r['sum'])}", axis=1)
    
    total = grouped.groupby('category').agg({'count':'sum','sum':'sum'}).reset_index()
    total['label'] = total.apply(lambda r: f"{r['category']} ({r['count']}) – {format_amount(r['sum'])}", axis=1)
    
    # Kategoria "Przychody" na górze
    total = pd.concat([
        total[total['category'] == 'Przychody'],
        total[total['category'] != 'Przychody'].sort_values('category')
    ])
    
    for _, row in total.iterrows():
        cat = row['category']
        with st.expander(label=row['label']):
            subs = grouped[grouped['category'] == cat]
            for _, r in subs.iterrows():
                st.markdown(f"- {r['formatted']}")



if __name__=="__main__":
    main()
