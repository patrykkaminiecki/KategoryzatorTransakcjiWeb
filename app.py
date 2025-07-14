import pandas as pd
import io
import streamlit as st
from pathlib import Path
from rapidfuzz import process, fuzz
import git     # tylko jeśli używasz auto‑push
import os

# ------------------------
# 1) DEFINICJA KATEGORII
# ------------------------
CATEGORIES = {
    'Przychody': ['Patryk', 'Jolka', 'Świadczenia', 'Inne'],
    'Rachunki': ['Prąd', 'Gaz', 'Woda', 'Odpady', 'Internet', 'Telefon',
                 'Subskrypcje', 'Przedszkole', 'Żłobek', 'Podatki'],
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

# ------------------------------------
# 2) KLASA DO ZARZĄDZANIA PRZYPISANIAMI
# ------------------------------------
class Categorizer:
    def __init__(self):
        self.map = {}
        if ASSIGNMENTS_FILE.exists():
            try:
                df = pd.read_csv(ASSIGNMENTS_FILE)
                for _, row in df.iterrows():
                    desc = row['description']
                    self.map[desc] = (row['category'], row['subcategory'])
            except Exception:
                st.warning("Plik assignments.csv istnieje, ale jest uszkodzony lub pusty.")

    def suggest(self, desc: str):
        if not self.map:
            return None
        best, score, _ = process.extractOne(desc, list(self.map.keys()), scorer=fuzz.token_sort_ratio)
        return self.map[best] if score > 80 else None

    def assign(self, key: str, cat: str, sub: str):
        self.map[key] = (cat, sub)

    def save(self):
        df = pd.DataFrame([
            {"description": key, "category": cat, "subcategory": sub}
            for key, (cat, sub) in self.map.items()
        ])
        df.to_csv(ASSIGNMENTS_FILE, index=False)

# -----------------------------------------------------
# 3) OPCJONALNIE: AUTO‑PUSH DO GITHUB (GitPython + SECRETS)
# -----------------------------------------------------
def auto_git_commit():
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["GITHUB_REPO"]
    author = st.secrets["GITHUB_AUTHOR"]
    repo_url = f"https://{token}@github.com/{repo_name}.git"

    # Klon lub otwórz repo
    if not Path(".git").exists():
        git.Repo.clone_from(repo_url, ".", branch="main")
        repo = git.Repo(".")
    else:
        repo = git.Repo(".")

    # Nadpisz origin
    if "origin" not in [r.name for r in repo.remotes]:
        repo.create_remote("origin", repo_url)
    else:
        repo.remotes.origin.set_url(repo_url)

    # Dodaj i push
    repo.index.add([str(ASSIGNMENTS_FILE)])
    if repo.is_dirty():
        name, email = author.replace(">", "").split(" <")
        repo.index.commit("Automatyczny zapis assignments.csv", author=git.Actor(name, email))
        repo.remotes.origin.push()

# ------------------------------------
# 4) FUNKCJA WCZYTANIA CSV Z BANKU
# ------------------------------------
def load_bank_csv(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    for enc, sep in [('cp1250',';'),('utf-8',';'),('utf-8',',')]:
        try:
            text = raw.decode(enc, errors='ignore').splitlines()
            idx = next(i for i, line in enumerate(text) if 'Data' in line and 'Kwota' in line)
            data = '\n'.join(text[idx:])
            df = pd.read_csv(io.StringIO(data), sep=sep, decimal=',')
            return df
        except Exception:
            continue
    raise ValueError("Nie udało się wczytać pliku CSV.")

# --------------------------
# 5) GŁÓWNA FUNKCJA STREAMLIT
# --------------------------
def main():
    st.title("🗂 Kategoryzator transakcji bankowych")

    # 5.1) Inicjuj Categorizer
    cat = Categorizer()

    # 5.2) Upload pliku
    uploaded = st.file_uploader("Wybierz plik CSV z banku", type=["csv"])
    if not uploaded:
        st.info("Wczytaj plik, żeby rozpocząć.")
        return

    # 5.3) Wczytaj i zmapuj
    try:
        df_raw = load_bank_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        return

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

    cols = ['Date','Description','Tytuł','Nr rachunku','Amount','Kwota blokady']
    df = df_raw[cols].copy()

    # 5.4) Utwórz composite key i grupuj
    df['_group_key'] = df['Description'].fillna('') + '|' + df['Nr rachunku'].fillna('')
    keys = df['_group_key'].unique().tolist()
    to_process = set(keys)
    groups = []
    while to_process:
        k = to_process.pop()
        matches = [x for x in keys if fuzz.token_sort_ratio(k, x) > 80]
        for m in matches:
            to_process.discard(m)
        groups.append(matches)

    # 5.5) Bulk‑assign: dla każdej grupy pytaj o kategorię
    st.markdown("### Przypisz kategorię do każdej grupy transakcji")
    for group in groups:
        if all(g in cat.map for g in group):
            continue
        desc0, acct0 = group[0].split('|')
        sugg = cat.suggest(group[0]) or ("","")
        col1, col2 = st.columns([3,2])
        with col1:
            st.write(f"**{desc0}** _(rachunek: {acct0 or '—'})_ + {len(group)-1} podobnych")
        with col2:
            sel_cat = st.selectbox("Kategoria", list(CATEGORIES.keys()),
                                   index=list(CATEGORIES.keys()).index(sugg[0]) if sugg[0] in CATEGORIES else 0,
                                   key="cat_"+group[0])
            sel_sub = st.selectbox("Podkategoria", CATEGORIES[sel_cat],
                                   index=CATEGORIES[sel_cat].index(sugg[1]) if sugg[1] in CATEGORIES.get(sel_cat,[]) else 0,
                                   key="sub_"+group[0])
        for g in group:
            cat.assign(g, sel_cat, sel_sub)

    st.markdown("---")
    st.success("Grupy oznaczone – możesz skorygować poszczególne transakcje.")

    # 5.6) Przygotuj finalną tabelę (bez Nr rachunku)
    df['category'] = df['_group_key'].apply(lambda k: cat.map.get(k,('',''))[0])
    df['subcategory'] = df['_group_key'].apply(lambda k: cat.map.get(k,('',''))[1])
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

    # 5.7) Zapis i (opcjonalnie) push
    if st.button("💾 Zapisz i wyeksportuj"):
        # uaktualnij mapę na podstawie edycji
        for _, row in edited.iterrows():
            # odtwórz klucz: description|rachunek
            # pobierz rachunek z df (według Description)
            acct = df.loc[df['Description']==row['Description'], 'Nr rachunku'].iloc[0]
            key = f"{row['Description']}|{acct}"
            cat.assign(key, row['category'], row['subcategory'])
        cat.save()
        st.success("Zapisano assignments.csv")

        # opcjonalnie: push do GitHuba
        try:
            auto_git_commit()
            st.success("Wysłano assignments.csv do GitHuba")
        except Exception as e:
            st.warning(f"Push nieudany: {e}")

        # pobierz finalny CSV
        out = edited.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Pobierz wynikowy CSV", data=out, file_name="wynik.csv")

if __name__ == "__main__":
    main()
