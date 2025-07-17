import pandas as pd
import io, re
import streamlit as st
from pathlib import Path
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go

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
    'Oszczędności': ['Poduszka bezpieczeństwa', 'Fundusz celowy', 'Inwestycje', 'Wypłata z oszczędności'],
    'Nadpłata Długów': ['Hipoteka', 'Samochód', 'TV+Dyson', 'Gmina Kolbudy'],
    'Wakacje': ['Wakacje'],
    'Gotówka': ['Wpłata', 'Wypłata']
}
ASSIGNMENTS_FILE = Path("assignments.csv")
CATEGORY_PAIRS = [f"{cat} — {sub}" for cat, subs in CATEGORIES.items() for sub in subs]

# --------------------------------------------------
# 2) EMBEDDINGI
# --------------------------------------------------
@st.cache_resource
def get_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def get_pair_embs():
    return get_embed_model().encode(CATEGORY_PAIRS, convert_to_numpy=True)

EMBED_MODEL = get_embed_model()
PAIR_EMBS = get_pair_embs()

# ------------------------------------
# 3) CATEGORIZER
# ------------------------------------
def clean_desc(s):
    text = str(s).replace("'", "").replace('"', "")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class Categorizer:
    def __init__(self):
        self.map = {}
        if ASSIGNMENTS_FILE.exists():
            df = pd.read_csv(ASSIGNMENTS_FILE).drop_duplicates('description', keep='last')
            for _, row in df.iterrows():
                self.map[clean_desc(row['description'])] = (row['category'], row['subcategory'])

    def suggest(self, key: str, amount: float):
        kc = clean_desc(key)
        if kc in self.map and self.map[kc][0]:
            return self.map[kc]
        emb = EMBED_MODEL.encode([key], convert_to_numpy=True)
        sims = cosine_similarity(emb, PAIR_EMBS)[0]
        idx, score = int(np.argmax(sims)), sims.max()
        if score > 0.5:
            return tuple(CATEGORY_PAIRS[idx].split(" — "))
        return ('Przychody','Inne') if amount >= 0 else ('Inne', CATEGORIES['Inne'][0])

    def assign(self, key: str, cat: str, sub: str):
        kc = clean_desc(key)
        if not kc:
            return
        self.map[kc] = (cat, sub)
        ASSIGNMENTS_FILE.parent.mkdir(exist_ok=True)
        pd.DataFrame([
            {"description": k, "category": c, "subcategory": s}
            for k,(c,s) in self.map.items()
        ]).to_csv(ASSIGNMENTS_FILE, index=False)

# ------------------------------------
# 4) WCZYTANIE CSV
# ------------------------------------
@st.cache_data
def load_bank_csv(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    for enc, sep in [('cp1250',';'),('utf-8',';'),('utf-8',',')]:
        try:
            lines = raw.decode(enc, errors='ignore').splitlines()
            idx = next(i for i,l in enumerate(lines) if 'Data' in l and 'Kwota' in l)
            return pd.read_csv(io.StringIO("\n".join(lines[idx:])), sep=sep, decimal=',')
        except:
            pass
    raise ValueError("Nie udało się wczytać pliku CSV.")

# ------------------------------------
# 5) GŁÓWNA FUNKCJA
# ------------------------------------
def main():
    st.set_page_config(page_title="Kategoryzator Finansowy", layout="wide")
    st.markdown(
        """
        <style>
        body { background-color: #18191A; color: #fff; }
        .stApp { background-color: #18191A; }
        .block-container { padding-top: 1.5rem; }
        .stDataFrame, .stTable { background: #222 !important; }
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h1 { color: #7fd8be; font-weight: bold; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("🗂 Kategoryzator transakcji + Raporty")
    cat = Categorizer()

    # — sidebar: plik + filtr
    st.sidebar.header("Filtr dat")
    uploaded = st.sidebar.file_uploader("Wybierz plik CSV", type="csv")
    if not uploaded:
        st.sidebar.info("Wczytaj plik CSV, aby rozpocząć.")
        return

    try:
        df_raw = load_bank_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        return

    # — przygotowanie df
    cols = [c.strip() for c in df_raw.columns if c is not None]
    df = df_raw.copy(); df.columns = cols
    df = df.rename(columns={
        'Data transakcji':'Date','Dane kontrahenta':'Description','Tytuł':'Tytuł',
        'Nr rachunku':'Nr rachunku','Kwota transakcji (waluta rachunku)':'Amount',
        'Kwota blokady/zwolnienie blokady':'Kwota blokady'
    })[['Date','Description','Tytuł','Nr rachunku','Amount','Kwota blokady']]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].notna()]

    # — filtr dat
    mode = st.sidebar.radio("Tryb filtrowania", ["Zakres dat","Pełny miesiąc"])
    if mode == "Zakres dat":
        mn, mx = df['Date'].min(), df['Date'].max()
        start, end = st.sidebar.date_input("Zakres dat", [mn, mx], mn, mx)
        df = df[(df['Date']>=start)&(df['Date']<=end)]
    else:
        yrs = sorted(df['Date'].dt.year.unique())
        months = {1:'Styczeń',2:'Luty',3:'Marzec',4:'Kwiecień',5:'Maj',
                  6:'Czerwiec',7:'Lipiec',8:'Sierpień',9:'Wrzesień',
                  10:'Październik',11:'Listopad',12:'Grudzień'}
        y = st.sidebar.selectbox("Rok", yrs, index=len(yrs)-1)
        mname = st.sidebar.selectbox("Miesiąc", list(months.values()), index=6)
        m = {v:k for k,v in months.items()}[mname]
        df = df[(df['Date'].dt.year==y)&(df['Date'].dt.month==m)]

    # — bulk‑assign
    df['key'] = (df['Nr rachunku'].astype(str).fillna('') + '|' + df['Description']).map(clean_desc)
    for idxs in df.groupby('key').groups.values():
        key = df.loc[idxs[0],'key']
        if key in cat.map and cat.map[key][0]: continue
        amt = df.loc[idxs[0],'Amount']
        st.markdown(f"**{key}** – {amt:.2f} PLN")
        sugg = cat.suggest(key, amt)
        sc = st.selectbox("Kategoria", list(CATEGORIES.keys()), index=list(CATEGORIES.keys()).index(sugg[0]), key=f"cat_{key}")
        opts = CATEGORIES[sc]
        ss = st.selectbox("Podkategoria", opts, index=opts.index(sugg[1]) if sugg[1] in opts else 0, key=f"sub_{key}")
        cat.assign(key, sc, ss)

    st.markdown("---")
    st.success("Krok 1: assignments.csv zaktualizowany.")

    # — finalna tabela
    df['category']    = df['key'].map(lambda k: cat.map.get(k,("", ""))[0])
    df['subcategory'] = df['key'].map(lambda k: cat.map.get(k,("", ""))[1])
    final = df[['Date','Description','Tytuł','Amount','Kwota blokady','category','subcategory']]
    st.markdown("## 🗃️ Tabela transakcji")
    edited = st.data_editor(final, hide_index=True, use_container_width=True)

    if st.button("💾 Zapisz zmiany"):
        keys = df['key'].tolist()
        for i,row in enumerate(edited.itertuples(index=False)):
            cat.assign(keys[i], row.category, row.subcategory)
        st.success("Zapisano assignments.csv")

    # — przygotowanie raportu
    ed = edited.copy()
    all_order = ['Przychody'] + sorted([c for c in CATEGORIES if c!='Przychody'])
    total = pd.DataFrame({'category': all_order, 'sum':0.0,'count':0}).set_index('category')
    if not ed.empty:
        inc = ed[(ed['category']=='Przychody')&(ed['Amount']>0)].groupby('category')['Amount'].agg(['sum','count'])
        exp = ed[(ed['category']!='Przychody')&(ed['Amount']<0)].groupby('category')['Amount'].agg(['sum','count'])
        tot_act = pd.concat([inc,exp])
        total.update(tot_act)
    total = total.reset_index()
    total['count']=total['count'].astype(int)
    total = total[total['count']>0]
    grouped = ed.groupby(['category','subcategory'])['Amount'].agg(['sum','count']).reset_index()

    st.markdown("## 📊 Raport: ilość i suma wg kategorii")
    fmt=lambda v:f"{abs(v):,.2f}".replace(","," ").replace(".",",")

    for _,r in total.iterrows():
        with st.expander(f"{r['category']} ({r['count']}) – {fmt(r['sum'])}", expanded=False):
            subs = grouped[grouped['category']==r['category']]
            for __,s in subs.iterrows():
                st.markdown(f"• **{s['subcategory']}** ({s['count']}) – {fmt(s['sum'])}", unsafe_allow_html=True)

    # — wykresy Plotly
    st.markdown("## 📈 Wykresy: kategorie i podkategorie")
    total_sorted = total.copy().reset_index(drop=True)
    colors = ["#2ca02c" if c=="Przychody" else "#d62728" for c in total_sorted['category']]

    col1,col2 = st.columns([3,1], gap="medium")
    with col1:
        st.markdown("#### Podział kategorii")
        fig_cat = go.Figure(data=[go.Pie(
            labels=total_sorted['category'],
            values=total_sorted['sum'].abs(),
            marker=dict(colors=colors, line=dict(color='#111',width=3)),
            textposition='inside',
            insidetextorientation='radial',
            texttemplate='<b>%{label}</b><br>%{percent:.0%}<br>%{value:,.2f} zł',
            hoverinfo='none',
            textfont=dict(size=12),
            hole=.4,
            domain=dict(x=[0.1,0.9], y=[0.1,0.9])
        )])
        fig_cat.update_layout(
            height=350,
            showlegend=False,
            paper_bgcolor='#111',
            plot_bgcolor='#111',
            font_color='white',
            uniformtext_minsize=8, uniformtext_mode='hide',
            margin=dict(l=20,r=20,t=40,b=20)
        )
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar":False})

    with col2:
        st.markdown("#### Wybierz kategorię")
        if 'selected' not in st.session_state:
            st.session_state['selected']=None
        for c in total_sorted['category']:
            if st.button(c,key=f"btn_{c}",use_container_width=True):
                st.session_state['selected']=c
        if st.button("Wszystkie",key="btn_all",use_container_width=True):
            st.session_state['selected']=None

    selected = st.session_state['selected']
    if selected:
        st.markdown(f"### Szczegóły dla: {selected}")
        sub = grouped[grouped['category']==selected].copy()
        sub = sub.sort_values('sum',ascending=False)
        fig_sub = go.Figure(data=[go.Pie(
            labels=sub['subcategory'],
            values=sub['sum'].abs(),
            marker=dict(line=dict(color='#111',width=2)),
            textposition='inside',
            insidetextorientation='radial',
            texttemplate='<b>%{label}</b><br>%{percent:.0%}<br>%{value:,.2f} zł',
            hoverinfo='none',
            textfont=dict(size=12),
            hole=.4,
            domain=dict(x=[0.1,0.9],y=[0.1,0.9])
        )])
        fig_sub.update_layout(
            title=f"Podkategorie dla: {selected}",
            height=350,
            showlegend=False,
            paper_bgcolor='#111',
            plot_bgcolor='#111',
            font_color='white',
            uniformtext_minsize=8, uniformtext_mode='hide',
            margin=dict(l=20,r=20,t=40,b=20)
        )
        st.plotly_chart(fig_sub, use_container_width=True, config={"displayModeBar":False}, key="sub")

if __name__=="__main__":
    main()
