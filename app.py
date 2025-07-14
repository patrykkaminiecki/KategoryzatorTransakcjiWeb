import pandas as pd
import io
import streamlit as st
from pathlib import Path
from rapidfuzz import process, fuzz
import git  # opcjonalnie, jeśli auto‐push

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
                    self.map[str(row['description'])] = (row['category'], row['subcategory'])
            except Exception:
                st.warning("Plik assignments.csv istnieje, ale jest uszkodzony lub pusty.")

    def suggest(self, key: str, amount: float):
        # 1. historia
        if key in self.map:
            return self.map[key]
        # 2. fuzzy
        if self.map:
            best, score, _ = process.extractOne(key, list(self.map.keys()), scorer=fuzz.token_sort_ratio)
            if score > 80:
                return self.map[best]
        # 3. domyślne wg kwoty
        if amount >= 0:
            return ('Przychody', 'Inne')
        return None

    def assign(self, key: str, cat: str, sub: str):
        self.map[key] = (cat, sub)

    def save(self):
        pd.DataFrame([
            {"description": k, "category": c, "subcategory": s}
            for k, (c, s) in self.map.items()
        ]).to_csv(ASSIGNMENTS_FILE, index=False)

# -----------------------------------------------------
# 3) OPCJONALNIE: AUTO‑PUSH DO GITHUB (GitPython + SECRETS)
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
# 4) FUNKCJA WCZYTANIA CSV Z BANKU
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
# 5) GŁÓWNA FUNKCJA STREAMLIT
# --------------------------
def main():
    st.title("🗂 Kategoryzator transakcji bankowych")
    cat = Categorizer()

    uploaded = st.file_uploader("Wybierz plik CSV z banku", type=["csv"])
    if not uploaded:
        st.info("Wczytaj plik, aby rozpocząć.")
        return

    try:
        df_raw = load_bank_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        return

    # mapowanie kolumn
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

    # 5.4) Przygotuj grupy:
    # a) transakcje z rachunkiem → grupuj po numerze
    acct_numbers = df['Nr rachunku'].dropna().unique().tolist()
    acct_groups = [df.index[df['Nr rachunku']==acct].tolist() for acct in acct_numbers]
    # b) transakcje bez rachunku → indywidualnie po opisie
    no_acct_idxs = df.index[df['Nr rachunku'].isna()].tolist()
    desc_groups = [[idx] for idx in no_acct_idxs]

    groups = acct_groups + desc_groups

    # 5.5) Bulk‑assign dla każdej grupy
    st.markdown("### Przypisz kategorię do każdej nieoznaczonej grupy")
    for idxs in groups:
        # klucz: albo rachunek, albo opis pierwszego wiersza
        first = idxs[0]
        acct = df.loc[first, 'Nr rachunku']
        key = str(acct) if pd.notna(acct) else str(df.loc[first, 'Description'])
        if key in cat.map:
            continue

        # pokaż przykłady z grupy
        descs = df.loc[idxs, 'Description'].unique().tolist()
        titles = df.loc[idxs, 'Tytuł'].unique().tolist()
        amount = df.loc[first, 'Amount']
        st.write(f"**Klucz:** {key}")
        st.write(f"- Opisy: {', '.join(descs[:3])}{'...' if len(descs)>3 else ''}")
        st.write(f"- Tytuły: {', '.join(titles[:3])}{'...' if len(titles)>3 else ''}")
        st.write(f"- Kwota przykładowa: {amount:.2f}")

        sugg = cat.suggest(key, amount) or ("", "")
        sel_cat = st.selectbox("Kategoria", list(CATEGORIES.keys()),
                               index=list(CATEGORIES.keys()).index(sugg[0]) if sugg[0] in CATEGORIES else 0,
                               key=f"cat_{key}")
        sel_sub = st.selectbox("Podkategoria", CATEGORIES[sel_cat],
                               index=CATEGORIES[sel_cat].index(sugg[1]) if sugg[1] in CATEGORIES.get(sel_cat,[]) else 0,
                               key=f"sub_{key}")

        # zapisz dla całej grupy
        for idx in idxs:
            cat.assign(key, sel_cat, sel_sub)

    st.markdown("---")
    st.success("Grupy oznaczone – teraz skoryguj pojedyncze wiersze w tabeli.")

    # 5.6) Finalna tabela + keys_list
    keys_list = []
    for i, row in df.iterrows():
        acct = row['Nr rachunku']
        key = str(acct) if pd.notna(acct) else str(row['Description'])
        keys_list.append(key)

    df['category'] = [cat.map.get(k, ("", ""))[0] for k in keys_list]
    df['subcategory'] = [cat.map.get(k, ("", ""))[1] for k in keys_list]
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
            'subcategory': st.column_config.SelectboxColumn(
                "Podkategoria",
                options=[s for subs in CATEGORIES.values() for s in subs]
            )
        },
        hide_index=True,
        use_container_width=True
    )

    # 5.7) Zapis + opcjonalny push
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
            st.warning(f"Push nieudany: {e}")
        out = edited.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Pobierz wynikowy CSV", data=out, file_name="wynik.csv")

if __name__ == "__main__":
    main()
