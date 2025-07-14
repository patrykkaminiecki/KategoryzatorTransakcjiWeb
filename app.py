import pandas as pd
import io
import streamlit as st
from pathlib import Path
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import git   # opcjonalnie, jeśli używasz auto‑push

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
        # 1) Historia
        if key in self.map and self.map[key][0]:
            return self.map[key]
        # 2) Embedding + cosine_similarity
        emb = EMBED_MODEL.encode([key], convert_to_numpy=True)
        sims = cosine_similarity(emb, PAIR_EMBS)[0]
        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]
        if best_score > 0.5:
            cat, sub = CATEGORY_PAIRS[best_idx].split(" — ")
            return (cat, sub)
        # 3) Fallback wg znaku kwoty
        if amount >= 0:
            return ('Przychody', 'Inne')
        else:
            # dla wydatków domyślnie Inne → Prezenty
            return ('Inne', CATEGORIES['Inne'][0])

    def assign(self, key: str, cat: str, sub: str):
        self.map[key] = (cat, sub)

    def save(self):
        pd.DataFrame([
            {"description": k, "category": c, "subcategory": s}
            for k, (c, s) in self.map.items()
        ]).to_csv(ASSIGNMENTS_FILE, index=False)

# -----------------------------------------------------
# 4) OPCJONALNIE: AUTO‑PUSH DO GITHUB (GitPython + SECRETS)
# -----------------------------------------------------
def auto_git_commit():
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["GITHUB_REPO"]
    author = st.secrets["GITHUB_AUTHOR"]
    repo_url = f"https://{token}@github.com/{repo_name}.git"
    if not Path(".git").exists():
        git.Repo.clone_from(repo_url, ".", branch="main")
    repo = git.Repo(".")
    repo.remotes.origin.set_url(repo_url)
    repo.index.add([str(ASSIGNMENTS_FILE)])
    if repo.is_dirty():
        name, email = author.replace(">", "").split(" <")
        repo.index.commit("Automatyczny zapis assignments.csv", author=git.Actor(name, email))
        repo.remotes.origin.push()

# ------------------------------------
# 5) FUNKCJA WCZYTANIA CSV Z BANKU
# ------------------------------------
def load_bank_csv(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    for enc, sep in [('cp1250',';'),('utf-8',';'),('utf-8',',')]:
        try:
            lines = raw.decode(enc, errors='ignore').splitlines()
            idx = next(i for i, line in enumerate(lines) if 'Data' in line and 'Kwota' in line)
            data = '\n'.join(lines[idx:])
            return pd.read_csv(io.StringIO(data), sep=sep, decimal=',')
        except Exception:
            continue
    raise ValueError("Nie udało się wczytać pliku CSV.")

# --------------------------
# 6) GŁÓWNA FUNKCJA STREAMLIT
# --------------------------
def main():
    st.title("🗂 Kategoryzator transakcji bankowych (pełna automatyzacja)")

    cat = Categorizer()

    uploaded = st.file_uploader("Wybierz plik CSV z banku", type=["csv"])
    if not uploaded:
        st.info("Wczytaj plik, by rozpocząć.")
        return

    # 6.1) Parsowanie
    try:
        df_raw = load_bank_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        return

    # 6.2) Mapowanie kolumn
    df = df_raw.loc[:, df_raw.columns.notna()]
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        'Data transakcji': 'Date',
        'Dane kontrahenta': 'Description',
        'Tytuł': 'Tytuł',
        'Nr rachunku': 'Nr rachunku',
        'Kwota transakcji (waluta rachunku)': 'Amount',
        'Kwota blokady/zwolnienie blokady': 'Kwota blokady'
    }, inplace=True)
    df = df[['Date','Description','Tytuł','Nr rachunku','Amount','Kwota blokady']]

    # 6.3) Grupowanie: rachunek lub opis
    acct_nums = df['Nr rachunku'].dropna().unique().tolist()
    acct_groups = [df.index[df['Nr rachunku']==a].tolist() for a in acct_nums]
    no_acct = df.index[df['Nr rachunku'].isna()].tolist()
    desc_groups = [[i] for i in no_acct]
    groups = acct_groups + desc_groups

    # 6.4) Bulk‑assign
    st.markdown("### Automatyczne przypisywanie kategorii (możesz skorygować)")
    for idxs in groups:
        first = idxs[0]
        acct = df.loc[first,'Nr rachunku']
        key = str(acct) if pd.notna(acct) else str(df.loc[first,'Description'])

        if key in cat.map and cat.map[key][0]:
            continue

        descs = [str(x) for x in df.loc[idxs,'Description'].unique()]
        titles = [str(x) for x in df.loc[idxs,'Tytuł'].unique()]
        amount = df.loc[first,'Amount']

        st.write(f"**Klucz:** {key}")
        st.write(f"- Opisy: {', '.join(descs[:3])}{'...' if len(descs)>3 else ''}")
        st.write(f"- Tytuły: {', '.join(titles[:3])}{'...' if len(titles)>3 else ''}")
        st.write(f"- Kwota: {amount:.2f}")

        sugg = cat.suggest(key, amount) or ("","")
        sel_cat = st.selectbox("Kategoria", list(CATEGORIES.keys()),
                               index=list(CATEGORIES.keys()).index(sugg[0]) if sugg[0] in CATEGORIES else 0,
                               key=f"cat_{key}")
        sel_sub = st.selectbox("Podkategoria", CATEGORIES[sel_cat],
                               index=CATEGORIES[sel_cat].index(sugg[1]) if sugg[1] in CATEGORIES.get(sel_cat,[]) else 0,
                               key=f"sub_{key}")

        for i in idxs:
            cat.assign(key, sel_cat, sel_sub)

    st.markdown("---")
    st.success("Grupy oznaczone – możesz teraz skorygować tabelę poniżej.")

    # 6.5) Finalna tabela + keys_list
    keys_list = []
    for _, row in df.iterrows():
        acct = row['Nr rachunku']
        key = str(acct) if pd.notna(acct) else str(row['Description'])
        keys_list.append(key)

    df['category']   = [cat.map.get(k,("", ""))[0] for k in keys_list]
    df['subcategory']= [cat.map.get(k,("", ""))[1] for k in keys_list]
    final = df[['Date','Description','Tytuł','Amount','Kwota blokady','category','subcategory']]

    edited = st.data_editor(
        final,
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
        hide_index=True,
        use_container_width=True
    )

    # 6.6) Zapis + opcjonalny push
    if st.button("💾 Zapisz i eksportuj"):
        for idx, row in enumerate(edited.itertuples(index=False)):
            key = keys_list[idx]
            cat.assign(key, row.category, row.subcategory)
        cat.save()
        st.success("Zapisano assignments.csv")
        try:
            auto_git_commit()
            st.success("Wysłano assignments.csv do GitHuba")
        except Exception as e:
            st.warning(f"Błąd push: {e}")
        out = edited.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Pobierz wynik", data=out, file_name="wynik.csv")

if __name__=="__main__":
    main()
