import pandas as pd
import io, re
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# ------------------------
# 1) DEFINICJA KATEGORII
# ------------------------
CATEGORIES = {
    'Przychody': ['Patryk','Jolka','Świadczenia','Inne'],
    'Rachunki': ['Prąd','Gaz','Woda','Odpady','Internet','Telefon','Subskrypcje','Przedszkole','Żłobek','Podatki'],
    'Transport': ['Paliwo','Ubezpieczenie','Parking','Przeglądy'],
    'Kredyty': ['Hipoteka','Samochód','TV+Dyson','Gmina Kolbudy'],
    'Jedzenie': ['Zakupy Spożywcze'],
    'Zdrowie': ['Apteka','Lekarz','Kosmetyki','Fryzjer'],
    'Odzież': ['Ubrania','Buty'],
    'Dom i Ogród': ['Dom','Ogród','Zwierzęta'],
    'Inne': ['Prezenty','Rozrywka','Hobby','Edukacja'],
    'Oszczędności': [
        'Poduszka bezpieczeństwa - Lokata',
        'Poduszka bezpieczeństwa - Konto',
        'Poduszka bezpieczeństwa - Obligacje',
        'Fundusz celowy',
        'Inwestycje'
    ],
    'Nadpłata Długów': ['Hipoteka','Samochód','TV+Dyson','Gmina Kolbudy'],
    'Wakacje': ['Wakacje'],
    'Gotówka': ['Wpłata','Wypłata']
}
ASSIGNMENTS_FILE = Path("assignments.csv")
CATEGORY_PAIRS = [f"{cat} — {sub}" for cat, subs in CATEGORIES.items() for sub in subs]

# ------------------------
# 2) EMBEDDINGI
# ------------------------
@st.cache_resource(ttl=3600)
def get_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(ttl=3600)
def get_pair_embs():
    return get_embed_model().encode(CATEGORY_PAIRS, convert_to_numpy=True)

# ------------------------
# 3) CATEGORIZER
# ------------------------
def clean_desc(s):
    return re.sub(r'\s+', ' ', str(s)).strip()

class Categorizer:
    def __init__(self):
        self.map = {}
        if ASSIGNMENTS_FILE.exists():
            try:
                dfm = pd.read_csv(ASSIGNMENTS_FILE).drop_duplicates('description', keep='last')
                for _, r in dfm.iterrows():
                    self.map[clean_desc(r['description'])] = (r['category'], r['subcategory'])
            except:
                self.map = {}
    
    def suggest(self, key, amt):
        kc = clean_desc(key)
        if kc in self.map and self.map[kc][0]:
            return self.map[kc]
        
        # Ulepszone sugerowanie dla transakcji ujemnych
        if amt < 0:
            emb = get_embed_model().encode([key], convert_to_numpy=True)
            sims = cosine_similarity(emb, get_pair_embs())[0]
            idx, score = int(np.argmax(sims)), sims.max()
            if score > 0.4:  # Obniżony próg dla wydatków
                return tuple(CATEGORY_PAIRS[idx].split(" — "))
            return ('Inne', CATEGORIES['Inne'][0])
        else:
            return ('Przychody','Inne')
    
    def assign(self, key, cat, sub):
        kc = clean_desc(key)
        if not kc: return
        self.map[kc] = (cat, sub)
        ASSIGNMENTS_FILE.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame([{'description':k,'category':c,'subcategory':s} for k,(c,s) in self.map.items()]) \
          .to_csv(ASSIGNMENTS_FILE, index=False)

# ------------------------
# 4) WCZYTANIE CSV
# ------------------------
def load_bank_csv(u):
    raw = u.getvalue()
    for enc, sep in [('cp1250',';'),('utf-8',';'),('utf-8',',')]:
        try:
            txt = raw.decode(enc, errors='ignore').splitlines()
            header_idx = next((i for i, l in enumerate(txt) if any('data' in l.lower() and ('transakcji' in l.lower() or 'księgow' in l.lower())), None)
            
            if header_idx is None:
                continue
                
            # Poprawne wczytywanie nagłówków
            header = txt[header_idx].split(sep)
            df = pd.read_csv(io.StringIO("\n".join(txt[header_idx:])), 
                            sep=sep, 
                            decimal=',', 
                            thousands=' ',
                            header=0)
            
            # Poprawne mapowanie kolumn
            col_map = {}
            for col in df.columns:
                lc = col.lower()
                if 'data transakcji' in lc: col_map[col] = 'Date'
                elif 'data księgowania' in lc: col_map[col] = 'Accounting_Date'
                elif 'dane kontrahenta' in lc: col_map[col] = 'Counterparty'
                elif 'tytuł' in lc: col_map[col] = 'Title'
                elif 'nr rachunku' in lc: col_map[col] = 'Account_Number'
                elif 'kwota transakcji' in lc: col_map[col] = 'Amount'
                elif 'saldo po transakcji' in lc: col_map[col] = 'Balance_After'
            
            return df.rename(columns=col_map)
        except Exception as e:
            st.warning(f"Błąd przy wczytywaniu: {str(e)}")
    raise ValueError("Nie można wczytać CSV.")

# ------------------------
# 5) GŁÓWNA FUNKCJA
# ------------------------
def main():
    st.set_page_config(page_title="Kategoryzator", layout="wide")
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: #0e1117;
        color: #fafafa;
    }
    .stDataFrame, .stTable {
        background-color: #1e2130 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🗂 Kategoryzator transakcji + Raporty")
    cat = Categorizer()

    # 1) Wczytanie i przygotowanie df_full
    st.sidebar.header("Filtr dat")
    up = st.sidebar.file_uploader("CSV", type="csv")
    if not up:
        st.sidebar.info("Wczytaj plik."); 
        return
    
    with st.spinner("Wczytywanie danych..."):
        try:
            df0 = load_bank_csv(up)
        except Exception as e:
            st.error(f"Błąd: {str(e)}")
            return

    # Sprawdzenie wymaganych kolumn
    required_cols = ['Date', 'Amount']
    if not all(col in df0.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df0.columns]
        st.error(f"Brak wymaganych kolumn: {', '.join(missing)}")
        return

    # Przygotowanie danych
    df_full = df0.copy()
    df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce', dayfirst=True)
    df_full = df_full[df_full['Date'].notna()]
    
    # Tworzenie klucza do kategoryzacji
    desc_col = 'Counterparty' if 'Counterparty' in df_full.columns else 'Title'
    if desc_col not in df_full.columns:
        st.error("Brak kolumny z opisem transakcji")
        return
        
    df_full['key'] = (df_full[desc_col].astype(str) + '|' + 
                     df_full.get('Account_Number', '').astype(str)).map(clean_desc)
    
    # Przypisanie początkowych kategorii
    for key in df_full['key'].unique():
        if key not in cat.map:
            amt = df_full.loc[df_full['key'] == key, 'Amount'].iloc[0]
            cat.map[key] = cat.suggest(key, amt)
    
    df_full['category'] = df_full['key'].map(lambda k: cat.map.get(k, ("", ""))[0])
    df_full['subcategory'] = df_full['key'].map(lambda k: cat.map.get(k, ("", ""))[1])

    # 2) Filtr dat
    df = df_full.copy()
    mode = st.sidebar.radio("Tryb filtrowania", ["Zakres dat", "Pełny miesiąc"])
    
    if mode == "Zakres dat":
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        d0, d1 = st.sidebar.date_input("Zakres dat", [min_date, max_date], min_date, max_date)
        start = datetime.combine(d0, datetime.min.time())
        end = datetime.combine(d1, datetime.max.time())
        df = df[(df['Date'] >= start) & (df['Date'] <= end)]
    else:
        years = sorted(df['Date'].dt.year.unique())
        months = {
            1: 'Styczeń', 2: 'Luty', 3: 'Marzec', 4: 'Kwiecień', 5: 'Maj',
            6: 'Czerwiec', 7: 'Lipiec', 8: 'Sierpień', 9: 'Wrzesień',
            10: 'Październik', 11: 'Listopad', 12: 'Grudzień'
        }
        year = st.sidebar.selectbox("Rok", years, index=len(years)-1)
        month_name = st.sidebar.selectbox("Miesiąc", list(months.values()), index=6)
        month = next(k for k, v in months.items() if v == month_name)
        df = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)]

    # 3) Przypisanie kategorii
    st.markdown("#### Krok 1: Przypisz kategorie")
    keys_to_assign = [k for k in df['key'].unique() if not cat.map.get(k, ("", ""))[0]]
    
    if not keys_to_assign:
        st.info("Wszystkie transakcje mają już przypisane kategorie.")
    else:
        for key in keys_to_assign:
            amt = df.loc[df['key'] == key, 'Amount'].iloc[0]
            st.write(f"**{key}** – {amt:.2f} PLN")
            col1, col2 = st.columns(2)
            with col1:
                cat_val = st.selectbox("Kategoria", list(CATEGORIES.keys()),
                                      key=f"cat_{key}")
            with col2:
                sub_val = st.selectbox("Podkategoria", CATEGORIES[cat_val],
                                      key=f"sub_{key}")
            cat.assign(key, cat_val, sub_val)
            time.sleep(0.1)  # Zapobieganie błędom interfejsu

    # 4) Tabela transakcji
    st.markdown("## 🗃️ Tabela transakcji")
    df['category'] = df['key'].map(lambda k: cat.map.get(k, ("", ""))[0])
    df['subcategory'] = df['key'].map(lambda k: cat.map.get(k, ("", ""))[1])
    
    cols_to_show = ['Date', desc_col, 'Amount', 'category', 'subcategory']
    if 'Balance_After' in df.columns:
        cols_to_show.append('Balance_After')
    
    edited = st.data_editor(
        df[cols_to_show],
        column_config={
            'Date': st.column_config.DateColumn("Data", format="DD.MM.YYYY"),
            'Amount': st.column_config.NumberColumn("Kwota", format="%.2f PLN"),
            'Balance_After': st.column_config.NumberColumn("Saldo", format="%.2f PLN"),
            'category': st.column_config.SelectboxColumn("Kategoria", options=list(CATEGORIES.keys())),
            'subcategory': st.column_config.SelectboxColumn("Podkategoria", 
                options=[sub for subs in CATEGORIES.values() for sub in subs])
        },
        hide_index=True, 
        use_container_width=True,
        num_rows="fixed"
    )
    
    if st.button("💾 Zapisz zmiany"):
        for idx, row in edited.iterrows():
            key = df.iloc[idx]['key']
            cat.assign(key, row['category'], row['subcategory'])
        st.success("Zapisano zmiany kategorii")

    # 5) Raporty
    st.markdown("## 📊 Raporty podsumowujące")
    
    # Przygotowanie danych do raportów
    report_df = edited.copy()
    report_df['Kwota'] = report_df['Amount'].abs()
    report_df['Typ'] = report_df['Amount'].apply(lambda x: 'Przychód' if x > 0 else 'Wydatek')
    
    # Podsumowanie ogólne
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Podsumowanie finansowe")
        total_income = report_df[report_df['Typ'] == 'Przychód']['Kwota'].sum()
        total_expense = report_df[report_df['Typ'] == 'Wydatek']['Kwota'].sum()
        balance = total_income - total_expense
        
        st.metric("Przychody", f"{total_income:,.2f} PLN")
        st.metric("Wydatki", f"{total_expense:,.2f} PLN")
        st.metric("Bilans", f"{balance:,.2f} PLN", 
                 delta_color="inverse" if balance < 0 else "normal")

    with col2:
        st.markdown("### Oszczędności")
        savings = report_df[report_df['category'] == 'Oszczędności']
        if not savings.empty:
            total_savings = savings['Kwota'].sum()
            st.metric("Łączne oszczędności", f"{total_savings:,.2f} PLN")
            
            savings_by_type = savings.groupby('subcategory')['Kwota'].sum()
            fig = go.Figure(data=[go.Pie(
                labels=savings_by_type.index,
                values=savings_by_type.values,
                hole=0.4,
                marker_colors=px.colors.qualitative.Pastel
            )])
            fig.update_layout(showlegend=True, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brak transakcji oszczędnościowych")

    # Wykresy kategorii
    st.markdown("### Rozkład wydatków")
    expense_df = report_df[report_df['Typ'] == 'Wydatek']
    
    if not expense_df.empty:
        by_category = expense_df.groupby('category')['Kwota'].sum().reset_index()
        by_subcategory = expense_df.groupby(['category', 'subcategory'])['Kwota'].sum().reset_index()
        
        tab1, tab2 = st.tabs(["Według kategorii", "Według podkategorii"])
        
        with tab1:
            fig = px.bar(by_category, x='category', y='Kwota', 
                         text='Kwota', color='category',
                         labels={'Kwota': 'Suma wydatków', 'category': 'Kategoria'})
            fig.update_traces(texttemplate='%{text:,.2f} PLN', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.treemap(by_subcategory, 
                            path=['category', 'subcategory'], 
                            values='Kwota',
                            color='category')
            fig.update_traces(textinfo="label+value+percent parent")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Brak danych o wydatkach w wybranym okresie")

if __name__ == "__main__":
    main()
