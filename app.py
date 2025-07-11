import sqlite3
import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path
import streamlit as st
import io

# 1) DEFINICJA KATEGORII I PODKATEGORII
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

DB_SCHEMA = '''
CREATE TABLE IF NOT EXISTS assignments (
    description TEXT PRIMARY KEY,
    category TEXT,
    subcategory TEXT
);
'''

class Categorizer:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute(DB_SCHEMA)
        self.conn.commit()

    def load_assignments(self):
        df = pd.read_sql_query(
            "SELECT description, category, subcategory FROM assignments", self.conn
        )
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
        best, score, _ = process.extractOne(
            description, list(assigned.keys()), scorer=fuzz.token_sort_ratio
        )
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
    st.markdown("Wczytaj plik CSV z banku, przypisz kategorie i pobierz gotowe dane.")

    # 🔐 Utwórz katalog trwały, jeśli nie istnieje
    Path.home().joinpath(".streamlit").mkdir(parents=True, exist_ok=True)
    db_path = Path.home() / ".streamlit" / "assignments.db"
    cat = Categorizer(db_path)

    uploaded = st.file_uploader("Wybierz plik CSV", type=["csv"])
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
        st.error("Nie udało się wczytać tabeli transakcji. Sprawdź plik.")
        return

    # Przytnij i przemapuj kolumny
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

    # Sprawdź, czy wszystkie potrzebne kolumny są dostępne
    required = ['Date','Description','Tytuł','Nr rachunku','Amount','Kwota blokady']
    if not all(c in df.columns for c in required):
        st.error(f"Brakuje oczekiwanych kolumn: {required}")
        return

    # Automatyczna kategoryzacja
    df = cat.categorize(df)

    # Wybierz kolumny do wyświetlenia
    subset = ['Date','Description','Tytuł','Nr rachunku','Amount','Kwota blokady','category','subcategory']
    df = df[subset]

    # Edytor danych z dropdownami
    edited = st.data_editor(
        df,
        column_config={
            'Date': st.column_config.Column('Data'),
            'Description': st.column_config.Column('Dane kontrahenta'),
            'Tytuł': st.column_config.Column('Tytuł'),
            'Nr rachunku': st.column_config.Column('Nr rachunku'),
            'Amount': st.column_config.NumberColumn('Kwota', format='%.2f'),
            'Kwota blokady': st.column_config.NumberColumn('Kwota blokady', format='%.2f'),
            'category': st.column_config.SelectboxColumn('Kategoria', options=list(CATEGORIES.keys())),
            'subcategory': st.column_config.SelectboxColumn(
                'Podkategoria',
                options=[sub for subs in CATEGORIES.values() for sub in subs]
            ),
        },
        hide_index=True,
        use_container_width=True
    )

    # Zapis i pobranie wyników
    if st.button("Zapisz i pobierz CSV"):
        cat.save_assignments(edited[['Description','category','subcategory']])
        out = edited.to_csv(index=False).encode('utf-8')
        st.download_button("Pobierz wynikowy CSV", out, file_name="wynik.csv", mime='text/csv')

if __name__ == '__main__':
    main()
