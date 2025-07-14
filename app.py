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
                    key = row['description']
                    self.map[key] = (row['category'], row['subcategory'])
            except Exception:
                st.warning("Plik assignments.csv istnieje, ale jest uszkodzony lub pusty.")

    def suggest(self, key: str):
        if not self.map:
            return None
        best, score, _ = process.extractOne(key, list(self.map.keys()), scorer=fuzz.token_sort_ratio)
        return self.map[best] if score > 80 else None

    def assign(self, key: str, cat: str, sub: str):
        self.map[key] = (cat, sub)

    def save(self):
        df = pd.DataFrame([
            {"description": k, "category": c, "subcategory": s}
            for k, (c, s) in self.map.items()
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

    if not Path(".git").exists():
        git.Repo.clone_from(repo_url, ".", branch="main")
        repo = git.Repo(".")
    else:
        repo = git.Repo(".")

    if "origin" not in [r.name for r in repo.remotes]:
        repo.create_remote("origin", repo_url)
    else:
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
        st.info("Wczytaj plik, żeby rozpocząć.")
        return

    try:
        df_raw = load_bank_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        return

    # Mapowanie i przycięcie
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

    # 5.4) Grupowanie po numerze rachunku
    acct_numbers = df['Nr rachunku'].dropna().unique().tolist()
    groups = [df.index[df['Nr rachunku']==acct].tolist() for acct in acct_numbers if (df['Nr rachunku']==acct).sum()>1]

    # 5.5) Bulk‐assign dla każdej grupy
    st.markdown("### Przypisz kategorię do każdej grupy transakcji (po rachunku)")
    for idxs in groups:
        acct = df.loc[idxs[0], 'Nr rachunku']
        key = str(acct)
        if key in cat.map:
            continue

        descs = df.loc[idxs, 'Description'].unique().tolist()
        titles = df.loc[idxs, 'Tytuł'].unique().tolist()
        st.write(f"**Rachunek:** {acct}")
        st.write(f"- Przykładowe opisy: {', '.join(descs[:3])}{'...' if len(descs)>3 else ''}")
        st.write(f"- Przykładowe tytuły: {', '.join(titles[:3])}{'...' if len(titles)>3 else ''}")

        sugg = cat.suggest(key) or ("","")
        sel_cat = st.selectbox("Kategoria", list(CATEGORIES.keys()),
                               index=list(CATEGORIES.keys()).index(sugg[0]) if sugg[0] in CATEGORIES else 0,
                               key=f"cat_{acct}")
        sel_sub = st.selectbox("Podkategoria", CATEGORIES[sel_cat],
                               index=CATEGORIES[sel_cat].index(sugg[1]) if sugg[1] in CATEGORIES.get(sel_cat,[]) else 0,
                               key=f"sub_{acct}")

        for idx in idxs:
            cat.assign(key, sel_cat, sel_sub)

    st.markdown("---")
    st.success("Grupy oznaczone – możesz teraz skorygować pojedyncze transakcje.")

    # 5.6) Finalna tabela (bez Nr rachunku)
    df['category'] = df['Nr rachunku'].astype(str).apply(lambda k: cat.map.get(k,('',''))[0])
    df['subcategory'] = df['Nr rachunku'].astype(str).apply(lambda k: cat.map.get(k,('',''))[1])
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
        # Aktualizacja mapy z edycji
        for _, row in edited.iterrows():
            acct = df.loc[df['Description']==row['Description'], 'Nr rachunku'].iloc[0]
            key = str(acct)
            cat.assign(key, row['category'], row['subcategory'])
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
