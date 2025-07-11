import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path
import streamlit as st
import io
import os
import git  # GitPython

st.set_page_config(layout="wide")

CATEGORIES = {
    'Przychód': ['Apteka'],
    'Rachunki': ['Buty'],
    'Subskrypcje': ['Disney'],
    'Kredyty': ['Dom'],
    'Jedzenie': ['Gaz'],
    'Wydatki': ['Gmina Kolbudy'],
    'Wakacje': ['Inne'],
    'Oszczędności': [
        'Internet','Jedzenie','Jolka','Nadpłata Kredytu','Netflix',
        'Odpady','Ogród','Oszczędności','Paliwo','Parking','Patryk',
        'Pies','Prąd','Przedszkole','Rozrywka','Samochód',
        'Telefon','TV + Dyson','Ubrania','Wakacje','Woda','Żłobek'
    ]
}

ASSIGNMENTS_FILE = Path("assignments.csv")

class Categorizer:
    def __init__(self):
        self.assignments = {}
        if ASSIGNMENTS_FILE.exists():
            try:
                df = pd.read_csv(ASSIGNMENTS_FILE)
                if not df.empty and {'description', 'category', 'subcategory'}.issubset(df.columns):
                    self.assignments = dict(zip(df['description'], zip(df['category'], df['subcategory'])))
                else:
                    st.warning("Plik assignments.csv istnieje, ale jest pusty lub nieprawidłowy.")
            except pd.errors.EmptyDataError:
                st.warning("Plik assignments.csv jest pusty.")

    def save(self):
        if not self.assignments:
            st.warning("Brak przypisań do zapisania.")
            return
        df = pd.DataFrame([
            {"description": desc, "category": cat, "subcategory": sub}
            for desc, (cat, sub) in self.assignments.items()
        ])
        df.to_csv(ASSIGNMENTS_FILE, index=False)

    def suggest(self, description):
        if not self.assignments:
            return None
        best, score, _ = process.extractOne(description, list(self.assignments.keys()), scorer=fuzz.token_sort_ratio)
        return self.assignments[best] if score > 80 else None

    def categorize(self, df):
        df['category'] = None
        df['subcategory'] = None
        for i, row in df.iterrows():
            desc = str(row.get('Description', '')).strip()
            if desc in self.assignments:
                df.at[i, 'category'], df.at[i, 'subcategory'] = self.assignments[desc]
            else:
                suggestion = self.suggest(desc)
                if suggestion:
                    df.at[i, 'category'], df.at[i, 'subcategory'] = suggestion
        return df

    def update_from_dataframe(self, df):
        for _, row in df.iterrows():
            desc = str(row['Description']).strip()
            cat = row['category']
            sub = row['subcategory']
            if desc and cat and sub:
                self.assignments[desc] = (cat, sub)

def auto_git_commit():
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["GITHUB_REPO"]
    author = st.secrets["GITHUB_AUTHOR"]
    repo_url = f"https://{token}@github.com/{repo_name}.git"

    if not Path(".git").exists():
        repo = git.Repo.clone_from(repo_url, ".", branch="main")
    else:
        repo = git.Repo(".")

    # Ustawienie zdalnego repozytorium z uwierzytelnieniem
    if "origin" not in [remote.name for remote in repo.remotes]:
        repo.create_remote("origin", repo_url)
    else:
        repo.remote("origin").set_url(repo_url)

    repo.git.add("assignments.csv")

    if repo.is_dirty():
        author_name, author_email = author.replace(">", "").split(" <")
        repo.index.commit(
            "Automatyczny zapis assignments.csv z aplikacji Streamlit",
            author=git.Actor(author_name, author_email)
        )
        repo.remote(name="origin").push()

def format_pln(amount):
    if pd.isna(amount) or str(amount).strip() == "" or \
       str(amount).replace(",", ".").strip() in ["0", "0.0"]:
        return ""
    try:
        return f"{float(amount):,.2f} PLN".replace(",", " ").replace(".", ",")
    except Exception:
        return ""

def main():
    st.title("📂 Kategoryzator transakcji bankowych (GitHub Sync)")
    uploaded = st.file_uploader("Wybierz plik CSV z banku", type=["csv"])
    if not uploaded:
        return

    raw = uploaded.getvalue()
    df = None
    for enc, sep in [('cp1250',';'),('utf-8',';'),('utf-8',',')]:
        try:
            text = raw.decode(enc, errors='ignore')
            lines = text.splitlines()
            header_i = next(i for i, line in enumerate(lines)
                            if 'Data transakcji' in line and 'Kwota' in line)
            data = '\n'.join(lines[header_i:])
            df = pd.read_csv(io.StringIO(data), sep=sep, decimal=',')
            break
        except Exception:
            continue

    if df is None:
        st.error("Nie udało się wczytać danych z pliku.")
        return

    df = df.loc[:, df.columns.notna()]
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        'Data transakcji': 'Date',
        'Dane kontrahenta': 'Description',
        'Tytuł': 'Tytuł',
        'Nr rachunku': 'Nr rachunku',
        'Kwota transakcji (waluta rachunku)': 'Amount',
        'Kwota blokady/zwolnienie blokady': 'Kwota blokady'
    }, inplace=True)

    # Usuwanie wierszy bez poprawnej daty
    import re
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    df = df[df['Date'].astype(str).str.match(date_pattern)]

    required = ['Date','Description','Tytuł','Nr rachunku','Amount','Kwota blokady']
    if not all(col in df.columns for col in required):
        st.error("Brakuje wymaganych kolumn w pliku.")
        return

    cat = Categorizer()
    df = df[required]
    df = cat.categorize(df)

    # Formatowanie walut
    df['Amount'] = df['Amount'].apply(format_pln)
    df['Kwota blokady'] = df['Kwota blokady'].apply(format_pln)

    # Edytor danych bez 'Nr rachunku'
    edited = st.data_editor(
        df[['Date','Description','Tytuł','Amount','Kwota blokady','category','subcategory']],
        column_config={
            'category': st.column_config.SelectboxColumn('Kategoria', options=list(CATEGORIES.keys())),
            'subcategory': st.column_config.SelectboxColumn(
                'Podkategoria',
                options=[sub for subs in CATEGORIES.values() for sub in subs]
            ),
        },
        hide_index=True,
        use_container_width=True
    )

    if st.button("💾 Zapisz przypisania i prześlij na GitHuba"):
        cat.update_from_dataframe(edited)
        cat.save()

        try:
            auto_git_commit()
            st.success("📤 Plik assignments.csv został zapisany i wysłany do GitHuba.")
        except Exception as e:
            st.warning(f"Nie udało się wykonać push: {e}")

        # Pobieranie wyniku
        csv = edited.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Pobierz wynikowy CSV", csv, file_name="wynik.csv", mime='text/csv')

if __name__ == '__main__':
    main()
