import sqlite3
import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path
import streamlit as st

# Definicja kategorii i podkategorii do wyboru
CATEGORIES = {
    'Przychód': ['Apteka'],
    'Rachunki': ['Buty'],
    'Subskrypcje': ['Disney'],
    'Kredyty': ['Dom'],
    'Jedzenie': ['Gaz'],
    'Wydatki': ['Gmina Kolbudy'],
    'Wakacje': ['Inne'],
    'Oszczędności': [
        'Internet', 'Jedzenie', 'Jolka', 'Nadpłata Kredytu', 'Netflix',
        'Odpady', 'Ogród', 'Oszczędności', 'Paliwo', 'Parking', 'Patryk',
        'Pies', 'Prąd', 'Przedszkole', 'Rozrywka', 'Samochód',
        'Telefon', 'TV + Dyson', 'Ubrania', 'Wakacje', 'Woda', 'Żłobek'
    ]
}

# Schemat bazy SQLite
DB_SCHEMA = '''
CREATE TABLE IF NOT EXISTS assignments (
    description TEXT PRIMARY KEY,
    category TEXT,
    subcategory TEXT
);
'''

class Categorizer:
    def __init__(self, db_path: Path):
        # Połączenie z bazą (plik w repozytorium lub pamięć)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute(DB_SCHEMA)
        self.conn.commit()

    def load_assignments(self):
        df = pd.read_sql_query("SELECT description, category, subcategory FROM assignments", self.conn)
        return dict(zip(df['description'], zip(df['category'], df['subcategory'])))

    def save_assignments(self, mappings: pd.DataFrame):
        cur = self.conn.cursor()
        for _, row in mappings.iterrows():
            cur.execute(
                "INSERT OR REPLACE INTO assignments(description, category, subcategory) VALUES (?, ?, ?)",
                (row['Description'], row['category'], row['subcategory'])
            )
        self.conn.commit()

    def suggest(self, description: str):
        assigned = self.load_assignments()
        if not assigned:
            return None
        best, score, _ = process.extractOne(description, list(assigned.keys()), scorer=fuzz.token_sort_ratio)
        return assigned[best] if score > 80 else None

    def categorize(self, df: pd.DataFrame) -> pd.DataFrame:
        assigned = self.load_assignments()
        df['category'] = None
        df['subcategory'] = None
        for idx, row in df.iterrows():
            desc = str(row.get('Description', '')).strip()
            if desc in assigned:
                df.at[idx, 'category'], df.at[idx, 'subcategory'] = assigned[desc]
            else:
                sug = self.suggest(desc)
                if sug:
                    df.at[idx, 'category'], df.at[idx, 'subcategory'] = sug
        return df

def main():
    st.title("Kategoryzator transakcji bankowych")
    st.markdown("Wczytaj plik CSV z transakcjami, nadaj kategorie i pobierz wynik.")

    # Ścieżka do pliku bazy SQLite (może być w pamięci lub na dysku)
    db_path_str = st.sidebar.text_input("Ścieżka do bazy SQLite", value="assignments.db")
    db_path = Path(db_path_str)
    cat = Categorizer(db_path)

    uploaded = st.file_uploader("Wybierz plik CSV", type=["csv"])
    if not uploaded:
        return

    # Próba wczytania w formacie polskim (średnik + cp1250), potem UTF-8
    try:
        df = pd.read_csv(uploaded, sep=';', encoding='cp1250')
    except Exception:
        try:
            df = pd.read_csv(uploaded, sep=',', encoding='utf-8')
        except Exception as e:
            st.error(f"Błąd podczas wczytywania pliku: {e}")
            return

    # Walidacja kolumn
    required = ['Date', 'Description', 'Amount']
    if not all(col in df.columns for col in required):
        st.error(f"Plik musi zawierać kolumny: {required}")
        return

    # Automatyczne kategoryzowanie
    df = cat.categorize(df)

    # Edycja przez użytkownika (dropdowny)
    edited = st.experimental_data_editor(
        df,
        column_config={
            'category': st.column_config.SelectboxColumn(
                'Kategoria', options=list(CATEGORIES.keys())
            ),
            'subcategory': st.column_config.SelectboxColumn(
                'Podkategoria',
                options=[sub for subs in CATEGORIES.values() for sub in subs]
            ),
        },
        use_container_width=True
    )

    # Zapis i pobranie
    if st.button("Zapisz i pobierz CSV"):
        cat.save_assignments(edited[['Description', 'category', 'subcategory']])
        csv = edited.to_csv(index=False).encode('utf-8')
        st.download_button("Pobierz wynikowy CSV", csv,
                          file_name="wynik.csv", mime='text/csv')

if __name__ == '__main__':
    main()
