import pandas as pd
import io
import streamlit as st
from pathlib import Path
from rapidfuzz import process, fuzz
import git  # tylko jeśli korzystasz z auto‑push
import os

# ---------------------------
# 1) TWOJA STRUKTURA KATEGORII
# ---------------------------
CATEGORIES = {
    'Przychody': ['Patryk', 'Jolka', 'Świadczenia', 'Inne'],
    'Rachunki': ['Prąd', 'Gaz', 'Woda', 'Odpady', 'Internet', 'Telefon', 'Subskrypcje', 'Przedszkole', 'Żłobek', 'Podatki'],
    'Transport': ['Paliwo', 'Ubezpieczenie', 'Parking', 'Przeglądy'],
    'Kredyty': ['Hipoteka', 'Samochód', 'TV+Dyson'],
    'Jedzenie': ['Zakupy Spożywcze'],
    'Zdrowie': ['Apteka', 'Lekarz', 'Kosmetyki', 'Fryzjer'],
    'Odzież': ['Ubrania', 'Buty'],
    'Dom i Ogród': ['Dom', 'Ogród', 'Zwierzęta'],
    'Inne': ['Prezenty', 'Rozrywka', 'Hobby', 'Edukacja'],
    'Oszczędności': ['Poduszka bezpieczeństwa', 'Fundusz celowy', 'Inwestycje'],
    'Nadpłata Długów': ['Hipoteka', 'Samochód', 'TV+Dyson'],
    'Wakacje': ['Wakacje'],
    'Gotówka': ['Wpłata', 'Wypłata']
}

ASSIGNMENTS_FILE = Path("assignments.csv")

# ---------------------------------
# 2) KLASA DO ZARZĄDZANIA ASSIGNMENT
# ---------------------------------
class Categorizer:
    def __init__(self):
        # wczytaj istniejące przypisania
        self.map = {}
        if ASSIGNMENTS_FILE.exists():
            try:
                df = pd.read_csv(ASSIGNMENTS_FILE)
                self.map = {
                    desc: (row['category'], row['subcategory'])
                    for _, row in df.iterrows()
                    for desc in [row['description']]
                }
            except Exception:
                st.warning("Plik assignments.csv istnieje, ale jest uszkodzony lub pusty.")

    def suggest(self, desc: str):
        if not self.map:
            return None
        best, score, _ = process.extractOne(desc, list(self.map.keys()), scorer=fuzz.token_sort_ratio)
        if score > 80:
            return self.map[best]
        return None

    def assign(self, desc: str, cat: str, sub: str):
        self.map[desc] = (cat, sub)

    def save(self):
        df = pd.DataFrame([
            {"description": desc, "category": cat, "subcategory": sub}
            for desc, (cat, sub) in self.map.items()
        ])
        df.to_csv(ASSIGNMENTS_FILE, index=False)

# ----------------------------------------------------
# 3) FUNKCJA DO AUTO‑PUSH (opcjonalnie: GitPython + Secrets)
# ----------------------------------------------------
def auto_git_commit():
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["GITHUB_REPO"]
    author = st.secrets["GITHUB_AUTHOR"]
    repo_url = f"https://{token}@github.com/{repo_name}.git"

    if not Path(".git").exists():
        git.Repo.clone_from(repo_url, ".", branch="main")
        repo = git.Repo(".")
    else:
        repo = git.Repo(".")

    # nadpisz origin z tokenem
    if "origin" not in [r.name for r in repo.remotes]:
        repo.create_remote("origin", repo_url)
    else:
        repo.remotes.origin.set_url(repo_url)

    repo.index.add([str(ASSIGNMENTS_FILE)])
    if repo.is_dirty():
        name, email = author.replace(">", "").split(" <")
        repo.index.commit("Automatyczny zapis assignments.csv", author=git.Actor(name, email))
        repo.remotes.origin.push()

# ----------------------------------------
# 4) POMOCNICZA FUNKCJA: wczytywanie CSV
# ----------------------------------------
def load_bank_csv(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    for enc, sep in [('cp1250',';'),('utf-8',';'),('utf-8',',')]:
        try:
            text = raw.decode(enc, errors='ignore').splitlines()
            # znajdź pierwszy wiersz, który zawiera „Data” i „Kwota”
            idx = next(i for i, line in enumerate(text) if 'Data' in line and 'Kwota' in line)
            data = '\n'.join(text[idx:])
            df = pd.read_csv(io.StringIO(data), sep=sep, decimal=',')
            return df
        except Exception:
            continue
    raise ValueError("Nie udało się wczytać pliku CSV.")

# -------------------------
# 5) GŁÓWNA FUNKCJA STREAMLIT
# -------------------------
def main():
    st.title("🗂 Kategoryzator transakcji bankowych")

    # 5.1) Wczytaj przypisania
    cat = Categorizer()

    # 5.2) Upload pliku bankowego
    uploaded = st.file_uploader("Wybierz CSV z banku", type=["csv"])
    if not uploaded:
        st.info("Wczytaj plik, aby zacząć kategoryzować.")
        return

    # 5.3) Parsuj i mapuj kolumny
    try:
        df_raw = load_bank_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        return

    # przytnij i mapuj
    df_raw = df_raw.loc[:, df_raw.columns.notna()]
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df_raw.rename(columns={
        'Data transakcji': 'Date',
        'Dane kontrahenta': 'Description',
        'Tytuł': 'Tytuł',
        'Nr rachunku': 'Nr rachunku',
        'Kwota transakcji (waluta rachunku)': 'Amount',
        'Kwota blokady/zwolnienie blokady': 'Kwota blokady'
    }, inplace=True)

    # wybierz kolumny potrzebne
    cols = ['Date','Description','Tytuł','Nr rachunku','Amount','Kwota blokady']
    df = df_raw[cols].copy()

    # 5.4) Podziel na grupy po podobnym opisie
    descriptions = df['Description'].fillna('').unique().tolist()
    to_process = set(descriptions)
    groups = []
    while to_process:
        desc = to_process.pop()
        # znajdź wszystkie podobne opisy
        matches = [d for d in descriptions
                   if fuzz.token_sort_ratio(desc, d) > 80]
        # usuń z to_process
        for m in matches:
            to_process.discard(m)
        groups.append(matches)

    # 5.5) Dla każdej grupy: pytanie o kategorię
    st.markdown("### Przypisz kategorie dla wykrytych grup transakcji")
    for group in groups:
        # sprawdź, czy już przypisane
        if all(g in cat.map for g in group):
            continue
        # zaproponuj kategorię na podstawie pierwszego w grupie
        sugg = cat.suggest(group[0]) or ("", "")
        col1, col2 = st.columns([2,1])
        with col1:
            st.write(f"**Opis:** `{group[0]}`  _(oraz {len(group)-1} podobnych)_")
        with col2:
            sel_cat = st.selectbox("Kategoria", options=list(CATEGORIES.keys()),
                                   index=list(CATEGORIES.keys()).index(sugg[0]) if sugg[0] in CATEGORIES else 0,
                                   key="cat_"+group[0])
            sel_sub = st.selectbox("Podkategoria",
                                   options=CATEGORIES[sel_cat],
                                   index=CATEGORIES[sel_cat].index(sugg[1]) if sugg[1] in CATEGORIES[sel_cat] else 0,
                                   key="sub_"+group[0])
        # zapisz dla całej grupy
        for g in group:
            cat.assign(g, sel_cat, sel_sub)

    st.markdown("---")
    st.success("Wszystkie grupy mają teraz przypisane kategorie. Możesz jeszcze skorygować pojedyncze transakcje poniżej.")

    # 5.6) Przypisz kategorie do df i pokaż edytor
    df['category'] = df['Description'].apply(lambda d: cat.map.get(d, ("", ""))[0])
    df['subcategory'] = df['Description'].apply(lambda d: cat.map.get(d, ("", ""))[1])

    edited = st.data_editor(
        df,
        column_config={
            'Date': st.column_config.Column("Data"),
            'Description': st.column_config.Column("Opis"),
            'Tytuł': st.column_config.Column("Tytuł"),
            'Nr rachunku': st.column_config.Column("Rachunek"),
            'Amount': st.column_config.NumberColumn("Kwota", format="%.2f"),
            'Kwota blokady': st.column_config.NumberColumn("Blokada", format="%.2f"),
            'category': st.column_config.SelectboxColumn("Kategoria", options=list(CATEGORIES.keys())),
            'subcategory': st.column_config.SelectboxColumn("Podkategoria", options=[s for subs in CATEGORIES.values() for s in subs])
        },
        hide_index=True,
        use_container_width=True
    )

    # 5.7) Zapis i push
    if st.button("💾 Zapisz i wyeksportuj"):
        # uaktualnij mapę
        for _, row in edited.iterrows():
            cat.assign(row['Description'], row['category'], row['subcategory'])
        # zapisz assignments.csv
        cat.save()
        st.success("Zapisano przypisania do assignments.csv")
        # auto‑push
        try:
            auto_git_commit()
            st.success("Wysłano assignments.csv do GitHub")
        except Exception as e:
            st.warning(f"Push nieudany: {e}")
        # pobierz wynik
        out = edited.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Pobierz wynikowy CSV", data=out, file_name="wynik.csv")

if __name__ == "__main__":
    main()
