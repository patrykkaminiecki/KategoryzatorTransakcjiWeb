import pandas as pd
import io, re
import streamlit as st
from pathlib import Path
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import requests
import base64
import json
import os

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
    'Oszczędności': [
        'Fundusz celowy',
        'Inwestycje',
        'Poduszka bezpieczeństwa - Konto',
        'Poduszka bezpieczeństwa - Lokata',
        'Poduszka bezpieczeństwa - Obligacje'
    ],
    'Nadpłata Długów': ['Hipoteka', 'Samochód', 'TV+Dyson', 'Gmina Kolbudy'],
    'Wakacje': ['Wakacje'],
    'Gotówka': ['Wpłata', 'Wypłata']
}

GITHUB_REPO = "patrykkaminiecki/kategoryzatortransakcjiweb"  # Zmień na swoje dane
GITHUB_FILE_PATH = "assignments.csv"
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")  # Token z Streamlit secrets

# Modele dystrybucji oszczędności
SAVINGS_MODELS = {
    "Zasada 50/30/20": {
        "description": "50% potrzeby, 30% rozrywka, 20% oszczędności",
        "distribution": {
            "Poduszka bezpieczeństwa - Konto": 50,
            "Inwestycje": 30,
            "Fundusz celowy": 20
        }
    },
    "Konserwatywny": {
        "description": "Priorytet bezpieczeństwa finansowego",
        "distribution": {
            "Poduszka bezpieczeństwa - Lokata": 40,
            "Poduszka bezpieczeństwa - Obligacje": 30,
            "Poduszka bezpieczeństwa - Konto": 20,
            "Fundusz celowy": 10
        }
    },
    "Agresywny": {
        "description": "Priorytet wzrostu kapitału",
        "distribution": {
            "Inwestycje": 60,
            "Fundusz celowy": 25,
            "Poduszka bezpieczeństwa - Konto": 15
        }
    }
}

ASSIGNMENTS_FILE = Path("assignments.csv")
CATEGORY_PAIRS = [f"{cat} — {sub}" for cat, subs in CATEGORIES.items() for sub in subs]

# ------------------------
# 2) EMBEDDINGI
# ------------------------
@st.cache_resource
def get_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def get_pair_embs():
    return get_embed_model().encode(CATEGORY_PAIRS, convert_to_numpy=True)

EMBED_MODEL = get_embed_model()
PAIR_EMBS = get_pair_embs()

def download_assignments_from_github():
    """Pobiera plik assignments.csv z GitHub"""
    if not GITHUB_TOKEN:
        return pd.DataFrame()
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.json()
            file_content = base64.b64decode(content['content']).decode('utf-8')
            return pd.read_csv(io.StringIO(file_content))
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Nie udało się pobrać assignments.csv z GitHub: {e}")
        return pd.DataFrame()

def upload_assignments_to_github(df):
    """Wysyła plik assignments.csv na GitHub"""
    if not GITHUB_TOKEN:
        st.error("Brak tokena GitHub")
        return False
    
    # Konwertuj DataFrame do CSV
    csv_content = df.to_csv(index=False)
    encoded_content = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    # Sprawdź czy plik już istnieje (potrzebujemy SHA)
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            sha = response.json()['sha']
        else:
            sha = None
    except:
        sha = None
    
    # Przygotuj dane do wysłania
    data = {
        "message": f"Update assignments.csv - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "content": encoded_content,
        "branch": "main"  # lub "master" jeśli używasz master
    }
    
    if sha:
        data["sha"] = sha
    
    try:
        response = requests.put(url, headers=headers, json=data)
        if response.status_code in [200, 201]:
            return True
        else:
            st.error(f"Błąd GitHub API: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Nie udało się wysłać pliku na GitHub: {e}")
        return False

# ------------------------
# 3) CATEGORIZER
# ------------------------
def clean_desc(s):
    return re.sub(r'\s+', ' ', str(s).replace("'", "").replace('"', "")).strip()

class Categorizer:
    def __init__(self):
        self.map = {}
        
        # Najpierw spróbuj pobrać z GitHub
        github_df = download_assignments_from_github()
        if not github_df.empty:
            github_df = github_df.drop_duplicates('description', keep='last')
            for _, r in github_df.iterrows():
                self.map[clean_desc(r['description'])] = (r['category'], r['subcategory'])
        
        # Jeśli nie ma w GitHub, sprawdź lokalnie
        elif ASSIGNMENTS_FILE.exists():
            df = pd.read_csv(ASSIGNMENTS_FILE).drop_duplicates('description', keep='last')
            for _, r in df.iterrows():
                self.map[clean_desc(r['description'])] = (r['category'], r['subcategory'])
    
    def suggest(self, key, amt):
        kc = clean_desc(key)
        if kc in self.map and self.map[kc][0]:
            return self.map[kc]
        emb = EMBED_MODEL.encode([key], convert_to_numpy=True)
        sims = cosine_similarity(emb, PAIR_EMBS)[0]
        idx, score = int(np.argmax(sims)), sims.max()
        if score > 0.5:
            return tuple(CATEGORY_PAIRS[idx].split(" — "))
        return ('Przychody','Inne') if amt>=0 else ('Inne',CATEGORIES['Inne'][0])
    
    def assign(self, key, cat, sub):
        kc = clean_desc(key)
        if not kc: return
        self.map[kc] = (cat, sub)
        
        # Zapisz lokalnie
        ASSIGNMENTS_FILE.parent.mkdir(exist_ok=True)
        df = pd.DataFrame([{"description":k,"category":c,"subcategory":s}
                          for k,(c,s) in self.map.items()])
        df.to_csv(ASSIGNMENTS_FILE, index=False)
        
        # Wyślij na GitHub
        if upload_assignments_to_github(df):
            st.success("✅ Zapisano assignments.csv lokalnie i na GitHub")
        else:
            st.warning("⚠️ Zapisano lokalnie, ale nie udało się wysłać na GitHub")

    def save_all_assignments(self):
        """Zapisuje wszystkie przypisania jednym ruchem bez pokazywania komunikatów"""
        ASSIGNMENTS_FILE.parent.mkdir(exist_ok=True)
        df = pd.DataFrame([{"description":k,"category":c,"subcategory":s}
                          for k,(c,s) in self.map.items()])
        df.to_csv(ASSIGNMENTS_FILE, index=False)
        return upload_assignments_to_github(df)

# ------------------------
# 4) WCZYTANIE CSV
# ------------------------
@st.cache_data
def load_bank_csv(u):
    raw = u.getvalue()
    for enc,sep in [('cp1250',';'),('utf-8',';'),('utf-8',',')]:
        try:
            lines=raw.decode(enc,errors='ignore').splitlines()
            idx=next(i for i,l in enumerate(lines) if 'Data' in l and 'Kwota' in l)
            return pd.read_csv(io.StringIO("\n".join(lines[idx:])), sep=sep, decimal=',')
        except:
            pass
    raise ValueError("Nie można wczytać CSV.")

# ------------------------
# 5) OBLICZENIE KWOTY EFEKTYWNEJ
# ------------------------
def calculate_effective_amount(row):
    """Zwraca Amount lub Kwota blokady - w zależności od tego, która nie jest 0"""
    amount = row.get('Amount', 0)
    blokada = row.get('Kwota blokady', 0)
    
    # Sprawdź które wartości nie są 0 lub NaN
    amount_valid = pd.notna(amount) and amount != 0
    blokada_valid = pd.notna(blokada) and blokada != 0
    
    if amount_valid and not blokada_valid:
        return amount
    elif blokada_valid and not amount_valid:
        return blokada
    elif amount_valid and blokada_valid:
        # Jeśli obie są niepuste, preferuj Amount
        return amount
    else:
        # Jeśli obie są 0 lub NaN, zwróć 0
        return 0

# ------------------------
# 6) FUNKCJE PROGNOZY
# ------------------------
def get_monthly_averages(df_full, months_back=3):
    """Oblicza średnie miesięczne wydatki na podstawie ostatnich miesięcy"""
    current_date = datetime.now()
    start_date = current_date - timedelta(days=months_back*30)
    
    recent_data = df_full[df_full['Date'] >= start_date]
    
    monthly_avg = recent_data.groupby(['category', 'subcategory']).agg({
        'Effective_Amount': 'mean'
    }).reset_index()
    
    return monthly_avg

def get_same_month_last_year(df_full, target_month, target_year):
    """Pobiera dane z tego samego miesiąca rok wcześniej"""
    last_year_data = df_full[
        (df_full['Date'].dt.month == target_month) & 
        (df_full['Date'].dt.year == target_year - 1)
    ]
    
    if last_year_data.empty:
        return pd.DataFrame()
    
    monthly_data = last_year_data.groupby(['category', 'subcategory']).agg({
        'Effective_Amount': 'sum'
    }).reset_index()
    
    return monthly_data

def get_recurring_expenses(df_full, months_back=6):
    """Identyfikuje powtarzające się wydatki na podstawie ostatnich miesięcy"""
    current_date = datetime.now()
    start_date = current_date - timedelta(days=months_back*30)
    
    recent_data = df_full[df_full['Date'] >= start_date]
    
    # Grupuj po miesiącach i kategorii
    monthly_groups = recent_data.groupby([
        recent_data['Date'].dt.to_period('M'),
        'category', 
        'subcategory'
    ]).agg({
        'Effective_Amount': 'sum'
    }).reset_index()
    
    # Znajdź wydatki które pojawiają się w większości miesięcy
    recurring = monthly_groups.groupby(['category', 'subcategory']).agg({
        'Effective_Amount': ['mean', 'count']
    }).reset_index()
    
    recurring.columns = ['category', 'subcategory', 'avg_amount', 'months_count']
    
    # Filtruj te które występują w co najmniej połowie miesięcy
    min_months = max(1, months_back // 2)
    recurring = recurring[recurring['months_count'] >= min_months]
    
    return recurring

def create_forecast(df_full, target_month, target_year):
    """Tworzy prognozę na podstawie różnych metod - bez oszczędności i nadpłat"""
    
    # Metoda 1: Średnia z ostatnich 3 miesięcy
    avg_3m = get_monthly_averages(df_full, 3)
    
    # Metoda 2: Dane z tego samego miesiąca rok wcześniej
    same_month_ly = get_same_month_last_year(df_full, target_month, target_year)
    
    # Metoda 3: Powtarzające się wydatki
    recurring = get_recurring_expenses(df_full)
    
    # Filtruj kategorie - bez oszczędności i nadpłat długów
    excluded_categories = ['Oszczędności', 'Nadpłata Długów']
    
    # Kombinuj prognozy - priorytet dla powtarzających się wydatków
    forecast = pd.DataFrame()
    
    # Rozpocznij od powtarzających się wydatków
    if not recurring.empty:
        recurring_filtered = recurring[~recurring['category'].isin(excluded_categories)]
        forecast = recurring_filtered[['category', 'subcategory', 'avg_amount']].copy()
        forecast = forecast.rename(columns={'avg_amount': 'predicted_amount'})
    
    # Dodaj z średniej 3-miesięcznej dla kategorii których nie ma
    if not avg_3m.empty:
        avg_3m_filtered = avg_3m[~avg_3m['category'].isin(excluded_categories)]
        for _, row in avg_3m_filtered.iterrows():
            exists = (
                (forecast['category'] == row['category']) & 
                (forecast['subcategory'] == row['subcategory'])
            ).any()
            
            if not exists:
                new_row = pd.DataFrame({
                    'category': [row['category']],
                    'subcategory': [row['subcategory']],
                    'predicted_amount': [row['Effective_Amount']]
                })
                forecast = pd.concat([forecast, new_row], ignore_index=True)
    
    # Dodaj z roku wcześniejszego dla kategorii których nie ma
    if not same_month_ly.empty:
        same_month_filtered = same_month_ly[~same_month_ly['category'].isin(excluded_categories)]
        for _, row in same_month_filtered.iterrows():
            exists = (
                (forecast['category'] == row['category']) & 
                (forecast['subcategory'] == row['subcategory'])
            ).any()
            
            if not exists:
                new_row = pd.DataFrame({
                    'category': [row['category']],
                    'subcategory': [row['subcategory']],
                    'predicted_amount': [row['Effective_Amount']]
                })
                forecast = pd.concat([forecast, new_row], ignore_index=True)
    
    return forecast

# ------------------------
# 7) GŁÓWNA FUNKCJA
# ------------------------
def main():
    st.set_page_config(page_title="Kategoryzator", layout="wide")
    st.markdown("""
    <style>
      body {background:#18191A;color:#fff;}
      .stApp, .stBlock{background:#18191A;}
      .stDataFrame, .stTable{background:#222!important;}
      .stMarkdown h1,h2,h3{color:#7fd8be;font-weight:bold;}
    </style>""", unsafe_allow_html=True)

    st.title("🏠 HomeFlow")
    cat = Categorizer()

    # --- Wczytanie pliku ---
    st.sidebar.header("Zakres analizy")
    up = st.sidebar.file_uploader("Import danych", type="csv")
    if not up:
        st.sidebar.info("Wczytaj plik CSV."); return
    try:
        df_raw = load_bank_csv(up)
    except Exception as e:
        st.error(str(e)); return
    if ASSIGNMENTS_FILE.exists():
        if st.button("🗑️ Usuń przypisania kategorii"):
            ASSIGNMENTS_FILE.unlink()
            st.success("Plik assignments.csv został usunięty.")
    # --- Przygotuj df_full (historyczne, do YTD i raportu) ---
    df_full = df_raw.copy()
    df_full.columns = [c.strip() for c in df_full.columns]
    df_full = df_full.rename(columns={
        'Data transakcji':'Date','Dane kontrahenta':'Description','Tytuł':'Tytuł',
        'Nr rachunku':'Nr rachunku','Kwota transakcji (waluta rachunku)':'Amount',
        'Kwota blokady/zwolnienie blokady':'Kwota blokady'
    })
    df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
    df_full = df_full[df_full['Date'].notna()]
    
    # Oblicz efektywną kwotę
    df_full['Effective_Amount'] = df_full.apply(calculate_effective_amount, axis=1)
    
    df_full['key'] = (df_full['Nr rachunku'].astype(str).fillna('') + '|' + df_full['Description']).map(clean_desc)
    df_full['category']    = df_full['key'].map(lambda k: cat.map.get(k,("",""))[0])
    df_full['subcategory'] = df_full['key'].map(lambda k: cat.map.get(k,("",""))[1])

    # --- Filtrowanie do głównego df ---
    df = df_full.copy()
    mode = st.sidebar.radio("Tryb filtrowania", ["Zakres dat","Pełny miesiąc"])
    if mode=="Zakres dat":
        mn,mx = df['Date'].min(), df['Date'].max()
        d0,d1 = st.sidebar.date_input("Zakres dat", [mn.date(), mx.date()], min_value=mn.date(), max_value=mx.date())
        start = datetime.combine(d0, datetime.min.time())
        end   = datetime.combine(d1, datetime.max.time())
        df = df[(df['Date']>=start)&(df['Date']<=end)]
    else:
        yrs = sorted(df['Date'].dt.year.unique())
        meses = {1:'Styczeń',2:'Luty',3:'Marzec',4:'Kwiecień',5:'Maj',
                 6:'Czerwiec',7:'Lipiec',8:'Sierpień',9:'Wrzesień',
                 10:'Październik',11:'Listopad',12:'Grudzień'}
        y = st.sidebar.selectbox("Rok", yrs, index=len(yrs)-1)
        mname = st.sidebar.selectbox("Miesiąc", list(meses.values()), index=6)
        m = {v:k for k,v in meses.items()}[mname]
        df = df[(df['Date'].dt.year==y)&(df['Date'].dt.month==m)]

        # --- Bulk‑assign ---
    # Sprawdź czy są nieprzypisane kategorie
    unassigned_keys = []
    for idxs in df.groupby('key').groups.values():
        key = df.loc[idxs[0],'key']
        if key not in cat.map or not cat.map[key][0]:
            unassigned_keys.append((key, idxs[0]))
    
    # Pokaż sekcję tylko jeśli są nieprzypisane kategorie
    if unassigned_keys:
        st.markdown("#### Krok 1: Przypisz kategorie")
        
        # Inicjalizacja session state dla indeksu bieżącej transakcji
        if 'current_assignment_index' not in st.session_state:
            st.session_state.current_assignment_index = 0
        
        # Sprawdź czy wszystkie transakcje zostały przetworzone
        if st.session_state.current_assignment_index >= len(unassigned_keys):
            st.success("✅ Wszystkie nowe transakcje zostały skategoryzowane!")
            st.session_state.current_assignment_index = 0  # Reset dla następnego razu
        else:
            # Pokaż progress bar
            progress = st.session_state.current_assignment_index / len(unassigned_keys)
            st.progress(progress, text=f"Postęp: {st.session_state.current_assignment_index + 1}/{len(unassigned_keys)}")
            
            # Pobierz bieżącą transakcję
            current_key, current_idx = unassigned_keys[st.session_state.current_assignment_index]
            current_amt = df.loc[current_idx,'Effective_Amount']
            
            # Pokaż dialog dla bieżącej transakcji
            with st.container():
                st.markdown("---")
                st.write(f"**Transakcja {st.session_state.current_assignment_index + 1}/{len(unassigned_keys)}:**")
                st.write(f"**Opis:** {current_key}")
                st.write(f"**Kwota:** {current_amt:.2f} PLN")
                
                # Pobierz sugestię dla bieżącej transakcji
                suggestion = cat.suggest(current_key, current_amt)
                
                # Selectboxy dla kategorii
                col1, col2 = st.columns(2)
                with col1:
                    selected_cat = st.selectbox(
                        "Kategoria", 
                        list(CATEGORIES.keys()), 
                        index=list(CATEGORIES.keys()).index(suggestion[0]), 
                        key=f"cat_popup_{st.session_state.current_assignment_index}"
                    )
                with col2:
                    selected_sub = st.selectbox(
                        "Podkategoria", 
                        CATEGORIES[selected_cat],
                        index=CATEGORIES[selected_cat].index(suggestion[1]) if suggestion[1] in CATEGORIES[selected_cat] else 0,
                        key=f"sub_popup_{st.session_state.current_assignment_index}"
                    )
                
                # Przyciski akcji
                col_skip, col_save, col_save_all = st.columns([1, 1, 1])
                
                with col_skip:
                    if st.button("⏭️ Pomiń", key=f"skip_{st.session_state.current_assignment_index}"):
                        st.session_state.current_assignment_index += 1
                        st.rerun()
                
                with col_save:
                    if st.button("💾 Zapisz i następna", key=f"save_next_{st.session_state.current_assignment_index}"):
                        # Zapisz kategorię dla bieżącej transakcji
                        cat.assign(current_key, selected_cat, selected_sub)
                        st.session_state.current_assignment_index += 1
                        st.success(f"Zapisano: {current_key} → {selected_cat} — {selected_sub}")
                        st.rerun()
                
                with col_save_all:
                    if st.button("💾 Zapisz pozostałe z sugestiami", key=f"save_all_{st.session_state.current_assignment_index}"):
                        # Zapisz bieżącą transakcję
                        cat.assign(current_key, selected_cat, selected_sub)
                        saved_count = 1
                        
                        # Zapisz wszystkie pozostałe z sugestiami
                        for i in range(st.session_state.current_assignment_index + 1, len(unassigned_keys)):
                            key, idx = unassigned_keys[i]
                            amt = df.loc[idx,'Effective_Amount']
                            auto_suggestion = cat.suggest(key, amt)
                            cat.assign(key, auto_suggestion[0], auto_suggestion[1])
                            saved_count += 1
                        
                        st.session_state.current_assignment_index = len(unassigned_keys)
                        st.success(f"Zapisano {saved_count} transakcji z automatycznymi sugestiami!")
                        st.rerun()
                
                st.markdown("---")
    else:
        st.success("✅ Wszystkie transakcje mają już przypisane kategorie!")

    # --- Tabela z dropdownami ---
    df['category']    = df['key'].map(lambda k: cat.map.get(k,("",""))[0])
    df['subcategory'] = df['key'].map(lambda k: cat.map.get(k,("",""))[1])
    final = df[['Date','Description','Tytuł','Effective_Amount','category','subcategory']].copy()
    final = final.rename(columns={'Effective_Amount': 'Kwota'})
    
    # Dodaj kolumny z oryginalnymi wartościami kategorii dla porównania
    original_categories = final[['category', 'subcategory']].copy()
    
    st.markdown("## 🗃️ Tabela Transakcji")
    edited = st.data_editor(
        final,
        column_config={
            'Date': st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
            'Kwota': st.column_config.NumberColumn("Kwota", format="%.2f"),
            'category': st.column_config.SelectboxColumn("Kategoria", options=list(CATEGORIES.keys())),
            'subcategory': st.column_config.SelectboxColumn("Podkategoria", options=[s for subs in CATEGORIES.values() for s in subs])
        },
        hide_index=True,
        use_container_width=True
    )
    
    if st.button("💾 Zapisz zmiany"):
        # Znajdź tylko rzeczywiście zmienione rekordy
        keys = df['key'].tolist()
        changes_made = False
        changed_count = 0
        
        for i, (original_row, edited_row) in enumerate(zip(original_categories.itertuples(index=False), edited.itertuples(index=False))):
            # Sprawdź czy kategoria lub podkategoria się zmieniła
            if original_row.category != edited_row.category or original_row.subcategory != edited_row.subcategory:
                desc = keys[i]
                # Aktualizuj tylko zmienione rekordy
                cat.map[desc] = (edited_row.category, edited_row.subcategory)
                changes_made = True
                changed_count += 1
        
        if changes_made:
            # Zapisz wszystkie przypisania jednym ruchem
            success = cat.save_all_assignments()
            if success:
                st.success(f"✅ Zapisano {changed_count} zmian lokalnie i na GitHub")
            else:
                st.warning(f"⚠️ Zapisano {changed_count} zmian lokalnie, ale nie udało się wysłać na GitHub")
        else:
            st.info("ℹ️ Nie wykryto żadnych zmian do zapisania")

    # --- Raport i YTD obok siebie ---
    colA, colB = st.columns(2, gap="medium")

    # RAPORT (na podstawie df filtrowanego)
    with colA:
        rt = df.copy()  # Używamy df (filtrowanego) zamiast df_full
        order = ['Przychody'] + sorted([c for c in CATEGORIES if c!='Przychody'])
        total = pd.DataFrame({'category':order,'sum':0.0,'count':0}).set_index('category')
        if not rt.empty:
            # Używamy Effective_Amount zamiast Amount
            inc = rt[(rt['category']=='Przychody')&(rt['Effective_Amount']>0)].groupby('category')['Effective_Amount'].agg(['sum','count'])
            exp = rt[(rt['category']!='Przychody')&(rt['Effective_Amount']<0)].groupby('category')['Effective_Amount'].agg(['sum','count'])
            total.update(pd.concat([inc,exp]))
        total = total.reset_index()
        total['count'] = total['count'].astype(int)
        total = total[total['count']>0]
        grouped = rt.groupby(['category','subcategory'])['Effective_Amount'].agg(['sum','count']).reset_index()

        st.markdown("## 📊 Podsumowanie")
        fmt = lambda v: f"{abs(v):,.2f}".replace(",", " ")
        for _,r in total.iterrows():
            # Oblicz rzeczywistą sumę dla kategorii z podkategorii
            real_sum = grouped[grouped['category']==r['category']]['sum'].sum()
            lbl = f"{r['category']} ({r['count']}) – {fmt(real_sum)}"
            with st.expander(lbl, expanded=False):
                subs = grouped[grouped['category']==r['category']]
                for __,s in subs.iterrows():
                    color = "green" if s['sum'] >= 0 else "red"
                    st.markdown(f"• **{s['subcategory']}** ({int(s['count'])}) – <span style='color:{color}'>{fmt(s['sum'])}</span>", unsafe_allow_html=True)

    # OSZCZĘDNOŚCI YTD (pełne dane - bez filtrowania) - POSORTOWANE A-Z
    with colB:
        st.markdown(f"## 💰 Twoje Oszczędności ({datetime.now().year})")
        ytd = df_full[(df_full['category']=='Oszczędności') & (df_full['Date'].dt.year==datetime.now().year)]
        total_ytd = ytd['Effective_Amount'].sum()
        st.markdown(f"**Łącznie: {abs(total_ytd):,.2f} zł**".replace(",", " "))
        sub = ytd.groupby('subcategory')['Effective_Amount'].sum().reset_index().sort_values('subcategory')  # Sortowanie A-Z
        for _,r in sub.iterrows():
            pct = (r['Effective_Amount']/total_ytd) if total_ytd else 0
            st.markdown(f"• **{r['subcategory']}** ({pct:.0%}) – {abs(r['Effective_Amount']):,.2f} zł".replace(",", " "))

    # --- SEKCJA SYMULACJI ---
    with st.expander("🔮 Symulacja przyszłego miesiąca", expanded=False):
    
        # Wybór miesiąca do prognozy
        col_month, col_year = st.columns(2)
        with col_month:
            next_month = datetime.now().month + 1 if datetime.now().month < 12 else 1
            month_names = ['Styczeń', 'Luty', 'Marzec', 'Kwiecień', 'Maj', 'Czerwiec',
                          'Lipiec', 'Sierpień', 'Wrzesień', 'Październik', 'Listopad', 'Grudzień']
            selected_month_name = st.selectbox("Miesiąc prognozy", month_names, index=next_month-1)
            selected_month = month_names.index(selected_month_name) + 1
        
        with col_year:
            next_year = datetime.now().year if datetime.now().month < 12 else datetime.now().year + 1
            selected_year = st.selectbox("Rok prognozy", [next_year, next_year + 1], index=0)
        
        # Generuj prognozę
        forecast = create_forecast(df_full, selected_month, selected_year)
        
        if not forecast.empty:
            st.markdown("### 📝 Edycja prognozy")
            st.markdown("Możesz zmodyfikować prognozowane wartości:")
            
            # Edytowalny data editor dla prognozy
            forecast_edited = st.data_editor(
                forecast,
                column_config={
                    'category': st.column_config.TextColumn("Kategoria", disabled=True),
                    'subcategory': st.column_config.TextColumn("Podkategoria", disabled=True),
                    'predicted_amount': st.column_config.NumberColumn("Prognozowana kwota", format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Oblicz prognozowane przychody i wydatki z edytowanej prognozy
            forecast_income = forecast_edited[forecast_edited['category'] == 'Przychody']['predicted_amount'].sum()
            forecast_expenses = forecast_edited[forecast_edited['category'] != 'Przychody']['predicted_amount'].sum()
            forecast_balance = forecast_income + forecast_expenses  # expenses są ujemne
            
            col_sim_a, col_sim_b = st.columns(2)
            
            with col_sim_a:
                st.markdown("### 📈 Prognoza finansowa")
                st.markdown(f"**Przychody:** {forecast_income:,.2f} zł".replace(",", " "))
                st.markdown(f"**Wydatki:** {abs(forecast_expenses):,.2f} zł".replace(",", " "))
                st.markdown(f"**Saldo:** {forecast_balance:,.2f} zł".replace(",", " "))
                
                # Szczegółowa prognoza po kategoriach
                st.markdown("### 📋 Szczegółowa prognoza")
                for category in sorted(forecast_edited['category'].unique()):
                    cat_data = forecast_edited[forecast_edited['category'] == category]
                    cat_total = cat_data['predicted_amount'].sum()
                    
                    with st.expander(f"{category} – {abs(cat_total):,.2f} zł".replace(",", " ")):
                        for _, row in cat_data.iterrows():
                            color = "green" if row['predicted_amount'] >= 0 else "red"
                            st.markdown(f"• **{row['subcategory']}** – <span style='color:{color}'>{abs(row['predicted_amount']):,.2f} zł</span>".replace(",", " "), unsafe_allow_html=True)
            
            with col_sim_b:
                st.markdown("### 💰 Dystrybucja nadwyżki")
                
                if forecast_balance > 0:
                    st.markdown(f"**Kwota do dystrybucji:** {forecast_balance:,.2f} zł".replace(",", " "))
                    
                    # Wybór podziału między oszczędności i nadpłaty
                    savings_percent = st.slider("% na oszczędności", 0, 100, 60)
                    debt_percent = 100 - savings_percent
                    
                    savings_amount = forecast_balance * savings_percent / 100
                    debt_amount = forecast_balance * debt_percent / 100
                    
                    st.markdown(f"**Oszczędności:** {savings_amount:,.2f} zł ({savings_percent}%)".replace(",", " "))
                    st.markdown(f"**Nadpłaty długów:** {debt_amount:,.2f} zł ({debt_percent}%)".replace(",", " "))
                    
                    # Wybór modelu dystrybucji oszczędności
                    model_choice = st.selectbox(
                        "Wybierz model dystrybucji oszczędności",
                        list(SAVINGS_MODELS.keys()) + ["Własny"]
                    )
                    
                    if model_choice != "Własny":
                        # Użyj predefiniowanego modelu
                        model = SAVINGS_MODELS[model_choice]
                        st.markdown(f"*{model['description']}*")
                        
                        st.markdown("**Dystrybucja oszczędności:**")
                        for subcategory, percentage in model['distribution'].items():
                            amount = savings_amount * percentage / 100
                            st.markdown(f"• **{subcategory}** ({percentage}%) – {amount:,.2f} zł".replace(",", " "))
                        
                        st.markdown("**Dystrybucja nadpłat długów:**")
                        debt_categories = CATEGORIES['Nadpłata Długów']
                        for debt in debt_categories:
                            debt_share = debt_amount / len(debt_categories)
                            st.markdown(f"• **{debt}** – {debt_share:,.2f} zł".replace(",", " "))
                    
                    else:
                        # Własny model - sliders
                        st.markdown("**Ustaw własne proporcje oszczędności:**")
                        
                        savings_subs = CATEGORIES['Oszczędności']
                        debt_subs = CATEGORIES['Nadpłata Długów']
                        
                        # Sliders dla oszczędności
                        savings_percentages = {}
                        debt_percentages = {}
                        
                        remaining_savings = 100
                        for i, sub in enumerate(savings_subs):
                            if i == len(savings_subs) - 1:
                                # Ostatnia kategoria dostaje resztę
                                savings_percentages[sub] = max(0, remaining_savings)
                            else:
                                max_val = remaining_savings
                                default_val = min(20, max_val)
                                if max_val > 0:
                                    pct = st.slider(f"{sub} (%)", 0, max_val, default_val, key=f"sav_{sub}")
                                    savings_percentages[sub] = pct
                                    remaining_savings -= pct
                                else:
                                    savings_percentages[sub] = 0
                        
                        st.markdown("**Ustaw własne proporcje nadpłat:**")
                        remaining_debt = 100
                        for i, sub in enumerate(debt_subs):
                            if i == len(debt_subs) - 1:
                                debt_percentages[sub] = max(0, remaining_debt)
                            else:
                                max_val = remaining_debt
                                default_val = min(25, max_val)
                                if max_val > 0:
                                    pct = st.slider(f"{sub} (%)", 0, max_val, default_val, key=f"debt_{sub}")
                                    debt_percentages[sub] = pct
                                    remaining_debt -= pct
                                else:
                                    debt_percentages[sub] = 0
                        
                        # Wyświetl dystrybucję
                        st.markdown("**Dystrybucja oszczędności:**")
                        for subcategory, percentage in savings_percentages.items():
                            amount = savings_amount * percentage / 100
                            st.markdown(f"• **{subcategory}** ({percentage}%) – {amount:,.2f} zł".replace(",", " "))
                        
                        st.markdown("**Dystrybucja nadpłat długów:**")
                        for subcategory, percentage in debt_percentages.items():
                            amount = debt_amount * percentage / 100
                            st.markdown(f"• **{subcategory}** ({percentage}%) – {amount:,.2f} zł".replace(",", " "))
                    
                else:
                    st.warning("Prognoza wskazuje na deficyt lub zerowe saldo - brak środków na oszczędności.")
        
        else:
            st.info("Brak wystarczających danych historycznych do utworzenia prognozy.")
    # --- DRILL‑DOWN wykresy kołowe ---
    with st.expander("📈 Twoje Finanse", expanded=False):
    
        # Utworzenie dwóch kolumn dla layoutu
        col_buttons, col_chart = st.columns([1, 3])
        
        # Przyciski kategorii w lewej kolumnie
        with col_buttons:
            if 'selected_cat' not in st.session_state:
                st.session_state['selected_cat'] = None
            st.markdown("**Wybierz kategorię:**")
            for cat_name in total['category']:
                if st.button(cat_name, key=f"btn_{cat_name}"):
                    st.session_state['selected_cat'] = cat_name
            if st.button("Resetuj wybór"):
                st.session_state['selected_cat'] = None
    
        # Wykresy w prawej kolumnie
        with col_chart:
            sel = st.session_state['selected_cat']
    
            # wykres kategorii
            tot = total.copy()
            colors = ["#2ca02c" if c=="Przychody" else "#d62728" for c in tot['category']]
            fig_cat = go.Figure(data=[go.Pie(
                labels=tot['category'], values=tot['sum'].abs(),
                marker=dict(colors=colors, line=dict(color='#111', width=3)),
                hole=0.3, domain=dict(x=[0.2,0.8], y=[0.2,0.8]),
                textposition='outside',
                texttemplate='<b>%{label}</b><br>%{percent:.0%}<br>%{value:,.2f} zł',
                textfont=dict(size=14, color='white'),
                pull=[0.02]*len(tot), hoverinfo='none'
            )])
            fig_cat.update_layout(height=450, showlegend=False,
                                  paper_bgcolor='#111', plot_bgcolor='#111', font_color='white',
                                  margin=dict(l=80,r=80,t=40,b=80))
            st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar":False})
    
            # wykres podkategorii
            if sel:
                sub = grouped[grouped['category']==sel].copy()
                title = f"Podkategorie: {sel}"
            else:
                sub = grouped.copy()
                title = "Podkategorie: wszystkie"
            fig_sub = go.Figure(data=[go.Pie(
                labels=sub['subcategory'], values=sub['sum'].abs(),
                marker=dict(line=dict(color='#111', width=2)),
                hole=0.3, domain=dict(x=[0.2,0.8], y=[0.2,0.8]),
                textposition='outside',
                texttemplate='<b>%{label}</b><br>%{percent:.0%}<br>%{value:,.2f} zł',
                textfont=dict(size=14, color='white'),
                pull=[0.02]*len(sub), hoverinfo='none'
            )])
            fig_sub.update_layout(title=title, height=450,
                                  showlegend=False,
                                  paper_bgcolor='#111', plot_bgcolor='#111', font_color='white',
                                  margin=dict(l=80,r=80,t=40,b=80))
            st.plotly_chart(fig_sub, use_container_width=True, config={"displayModeBar":False})
if __name__ == "__main__":
    main()
