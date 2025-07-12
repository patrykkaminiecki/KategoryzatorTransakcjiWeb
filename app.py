import streamlit as st
import pandas as pd
from github import Github, UnknownObjectException
import io

# --- Konfiguracja strony ---
st.set_page_config(
    page_title="Kategoryzator Transakcji",
    page_icon="💰",
    layout="wide"
)

# --- Kategorie i Podkategorie ---
# Słownik z predefiniowanymi kategoriami i podkategoriami
CATEGORIES = {
    "Przychody": ["Patryk", "Jolka", "Świadczenia", "Inne"],
    "Rachunki": ["Prąd", "Gaz", "Woda", "Odpady", "Internet", "Telefon", "Subskrypcje", "Przedszkole", "Żłobek", "Podatki"],
    "Transport": ["Paliwo", "Ubezpieczenie", "Parking", "Przeglądy"],
    "Kredyty": ["Hipoteka", "Samochód", "TV+Dyson"],
    "Jedzenie": ["Zakupy Spożywcze"],
    "Zdrowie": ["Apteka", "Lekarz", "Kosmetyki", "Fryzjer"],
    "Odzież": ["Ubrania", "Buty"],
    "Dom i Ogród": ["Dom", "Ogród", "Zwierzęta"],
    "Inne": ["Prezenty", "Rozrywka", "Hobby", "Edukacja"],
    "Oszczędności": ["Poduszka bezpieczeństwa", "Fundusz celowy", "Inwestycje"],
    "Nadpłata Długów": ["Hipoteka", "Samochód", "TV+Dyson"],
    "Wakacje": ["Wakacje"],
    "Gotówka": ["Wpłata", "Wypłata"]
}

# --- Funkcje do obsługi GitHub ---

@st.cache_resource
def get_github_repo():
    """Nawiązuje połączenie z repozytorium GitHub."""
    try:
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(st.secrets["GITHUB_REPO"])
        return repo
    except Exception as e:
        st.error(f"Błąd połączenia z GitHub: {e}")
        st.error("Upewnij się, że GITHUB_TOKEN i GITHUB_REPO są poprawnie skonfigurowane w secrets.toml.")
        return None

def get_assignments_from_github(repo):
    """Pobiera plik z przypisaniami kategorii z GitHub."""
    try:
        content = repo.get_contents("assignments.csv")
        csv_content = content.decoded_content.decode('utf-8')
        return pd.read_csv(io.StringIO(csv_content))
    except UnknownObjectException:
        # Plik nie istnieje, tworzymy pusty DataFrame
        return pd.DataFrame(columns=["key", "Kategoria", "Podkategoria"])
    except Exception as e:
        st.error(f"Nie udało się wczytać pliku 'assignments.csv': {e}")
        return pd.DataFrame(columns=["key", "Kategoria", "Podkategoria"])


def push_assignments_to_github(repo, df_assignments):
    """Wysyła zaktualizowany plik z przypisaniami na GitHub."""
    csv_buffer = io.StringIO()
    df_assignments.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    commit_message = "Aktualizacja przypisań kategorii przez aplikację Streamlit"

    try:
        contents = repo.get_contents("assignments.csv")
        repo.update_file(contents.path, commit_message, csv_content, contents.sha)
        st.success("✅ Pomyślnie zaktualizowano przypisania na GitHub!")
    except UnknownObjectException:
        repo.create_file("assignments.csv", commit_message, csv_content)
        st.success("✅ Pomyślnie utworzono i zapisano przypisania na GitHub!")
    except Exception as e:
        st.error(f"Błąd podczas zapisu na GitHub: {e}")

# --- Główna logika aplikacji ---

st.title("💰 Aplikacja do Kategoryzacji Transakcji Bankowych")
st.markdown("""
Wgraj plik CSV z historią transakcji, a aplikacja pomoże Ci je skategoryzować.
Twoje wybory są zapamiętywane i automatycznie stosowane w przyszłości.
""")

# Inicjalizacja repozytorium
repo = get_github_repo()

# Inicjalizacja stanu sesji, aby przechowywać dane między interakcjami
if 'assignments' not in st.session_state:
    if repo:
        st.session_state.assignments = get_assignments_from_github(repo)
    else:
        st.session_state.assignments = pd.DataFrame(columns=["key", "Kategoria", "Podkategoria"])

if 'transactions' not in st.session_state:
    st.session_state.transactions = None


# Krok 1: Wgranie pliku
uploaded_file = st.file_uploader(
    "Wybierz plik CSV z historią transakcji",
    type="csv"
)

if uploaded_file is not None:
    try:
        df_trans = pd.read_csv(
            uploaded_file,
            sep=';',
            encoding='cp1250',
            header=10
        )
        df_trans.columns = [
            col.replace('"', '').replace('³', 'ł').replace('æ', 'ę').replace('¿', 'ż')
               .replace('¹', 'ą').replace('œ', 'ś').replace('ó', 'ó')
               .replace('ä', 'a').replace('æ', 'ę').replace('Ÿ', 'Ż')
               .replace('Ó', 'Ó').replace('£', 'Ł').replace('ñ', 'ń')
               .replace('Ê', 'Ę').replace('ê', 'ę').replace('¿', 'ż')
               .replace('³', 'ł').replace('Œ', 'Ś').replace('ú', 'ó')
               .strip()
            for col in df_trans.columns
        ]
        st.success("Plik CSV wczytany pomyślnie!")
        st.write("Kolumny w pliku:", df_trans.columns)
        df_trans['key'] = (df_trans['Tytuł'].fillna('') + ' ' + df_trans['Dane kontrahenta'].fillna('')).str.lower().str.strip()
        df_merged = pd.merge(df_trans, st.session_state.assignments, on='key', how='left')
        st.session_state.transactions = df_merged
    except Exception as e:
        st.error(f"Błąd podczas przetwarzania pliku: {e}")
        st.warning("Upewnij się, że plik ma kodowanie 'cp1250', separator ';' oraz nagłówek w wierszu 11.")

# Jeśli transakcje są w stanie sesji, kontynuujemy pracę
if st.session_state.transactions is not None:
    df = st.session_state.transactions

    # Krok 2: Kategoryzacja nowych transakcji
    uncategorized_df = df[df['Kategoria'].isna()]
    unique_new_keys = uncategorized_df['key'].unique()

    if len(unique_new_keys) > 0:
        st.subheader("✍️ Nowe transakcje do skategoryzowania")
        st.info(f"Znaleziono {len(unique_new_keys)} unikalnych transakcji, które wymagają przypisania kategorii. Wypełnij poniższe pola, a Twój wybór zostanie zapamiętany.")

        # Używamy expandera, żeby nie zaśmiecać widoku
        with st.expander("Kliknij, aby przypisać kategorie"):
            new_assignments = {}
            for key in unique_new_keys:
                st.markdown(f"**Transakcja:** `{key}`")
                cols = st.columns(2)
                # Wybór kategorii głównej
                cat = cols[0].selectbox("Kategoria", options=list(CATEGORIES.keys()), key=f"cat_{key}", index=None, placeholder="Wybierz kategorię...")
                if cat:
                    # Wybór podkategorii na podstawie wybranej kategorii głównej
                    sub_cat = cols[1].selectbox("Podkategoria", options=CATEGORIES[cat], key=f"sub_{key}", index=None, placeholder="Wybierz podkategorię...")
                    if sub_cat:
                        new_assignments[key] = {'Kategoria': cat, 'Podkategoria': sub_cat}

            if st.button("Zapisz nowe kategorie", type="primary"):
                # Aktualizacja DataFrame z przypisaniami
                new_assignments_df = pd.DataFrame.from_dict(new_assignments, orient='index').reset_index().rename(columns={'index': 'key'})
                st.session_state.assignments = pd.concat([st.session_state.assignments, new_assignments_df], ignore_index=True).drop_duplicates(subset=['key'], keep='last')
                
                # Ponowne scalenie danych i odświeżenie widoku
                st.session_state.transactions = pd.merge(st.session_state.transactions.drop(columns=['Kategoria', 'Podkategoria']), st.session_state.assignments, on='key', how='left')
                st.rerun() # Odświeża aplikację, aby pokazać zaktualizowane dane

    # Krok 3: Wyświetlanie i edycja wszystkich transakcji
    st.subheader("📊 Twoje transakcje")
    st.markdown("Możesz ręcznie zmienić kategorię dla każdej transakcji poniżej. Zmiany zostaną uwzględnione przy ostatecznym zapisie.")

    # Lista kolumn do wyświetlenia
    display_columns = [
        'Data transakcji', 'Tytuł', 'Dane kontrahenta', 'Kwota transakcji (waluta rachunku)', 'Kategoria', 'Podkategoria'
    ]
    # Upewniamy się, że kolumny Kategoria i Podkategoria istnieją
    if 'Kategoria' not in df.columns:
        df['Kategoria'] = None
    if 'Podkategoria' not in df.columns:
        df['Podkategoria'] = None

    # Edytor danych - serce aplikacji
    edited_df = st.data_editor(
        df[display_columns],
        column_config={
            "Kategoria": st.column_config.SelectboxColumn(
                "Kategoria",
                options=list(CATEGORIES.keys()),
                required=True,
            ),
            "Podkategoria": st.column_config.SelectboxColumn(
                "Podkategoria",
                options=[sub for cat in CATEGORIES.values() for sub in cat], # Płaska lista wszystkich podkategorii
                required=True,
            ),
            "Kwota transakcji (waluta rachunku)": st.column_config.NumberColumn(
                "Kwota",
                format="%.2f PLN"
            )
        },
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic" # pozwala na dodawanie/usuwanie wierszy, choć tu głównie edytujemy
    )

    # Krok 4: Zapis zmian
    if st.button("💾 Zapisz wszystkie zmiany i wyślij do GitHub", disabled=(repo is None)):
        # Aktualizujemy główny zbiór przypisań na podstawie edycji w tabeli
        # Bierzemy tylko unikalne klucze i ich ostatnie przypisania z edytowanej tabeli
        final_assignments = edited_df[['Kategoria', 'Podkategoria']].copy()
        final_assignments['key'] = st.session_state.transactions['key'] # Dodajemy klucz z oryginalnego DF
        
        # Usuwamy puste i duplikaty, zachowując najnowsze przypisanie
        final_assignments = final_assignments.dropna(subset=['Kategoria', 'Podkategoria'])
        final_assignments = final_assignments.drop_duplicates(subset=['key'], keep='last')

        # Zapisujemy na GitHub
        push_assignments_to_github(repo, final_assignments)
        st.session_state.assignments = final_assignments # Aktualizujemy stan sesji
