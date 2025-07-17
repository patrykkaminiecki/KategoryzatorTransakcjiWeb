import pandas as pd
import io, re
import streamlit as st
from pathlib import Path
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

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
        'Poduszka bezpieczeństwa - Lokata',
        'Poduszka bezpieczeństwa - Konto',
        'Poduszka bezpieczeństwa - Obligacje',
        'Fundusz celowy',
        'Inwestycje'
    ],
    'Nadpłata Długów': ['Hipoteka', 'Samochód', 'TV+Dyson', 'Gmina Kolbudy'],
    'Wakacje': ['Wakacje'],
    'Gotówka': ['Wpłata', 'Wypłata']
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

# ------------------------
# 3) CATEGORIZER
# ------------------------
def clean_desc(s):
    return re.sub(r'\s+', ' ', str(s).replace("'", "").replace('"', "")).strip()

class Categorizer:
    def __init__(self):
        self.map = {}
        if ASSIGNMENTS_FILE.exists():
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
        ASSIGNMENTS_FILE.parent.mkdir(exist_ok=True)
        pd.DataFrame([
            {"description": k,"category": c,"subcategory": s}
            for k,(c,s) in self.map.items()
        ]).to_csv(ASSIGNMENTS_FILE, index=False)

# ------------------------
# 4) WCZYTANIE CSV
# ------------------------
@st.cache_data
def load_bank_csv(uploaded):
    raw = uploaded.getvalue()
    for enc,sep in [('cp1250',';'),('utf-8',';'),('utf-8',',')]:
        try:
            lines = raw.decode(enc,errors='ignore').splitlines()
            idx = next(i for i,l in enumerate(lines) if 'Data' in l and 'Kwota' in l)
            return pd.read_csv(io.StringIO("\n".join(lines[idx:])), sep=sep, decimal=',')
        except:
            pass
    raise ValueError("Nie można wczytać CSV.")

# ------------------------
# 5) GŁÓWNA FUNKCJA
# ------------------------
def main():
    st.set_page_config(page_title="Kategoryzator", layout="wide")
    st.markdown("""
    <style>
      body {background:#18191A;color:#fff;}
      .stApp, .stBlock{background:#18191A;}
      .stDataFrame, .stTable{background:#222!important;}
      .stMarkdown h1,h2,h3{color:#7fd8be;font-weight:bold;}
    </style>""",unsafe_allow_html=True)

    st.title("🗂 Kategoryzator transakcji + Raporty")
    cat = Categorizer()

    # Wczytaj
    st.sidebar.header("Filtr dat")
    up = st.sidebar.file_uploader("CSV", type="csv")
    if not up:
        st.sidebar.info("Wczytaj plik."); return
    try: df_raw = load_bank_csv(up)
    except Exception as e:
        st.error(str(e)); return

    # Przygotuj df_full (bez filtra) do YTD
    df_full = df_raw.copy()
    df_full.columns = [c.strip() for c in df_full.columns]
    df_full = df_full.rename(columns={
        'Data transakcji':'Date','Dane kontrahenta':'Description','Tytuł':'Tytuł',
        'Nr rachunku':'Nr rachunku','Kwota transakcji (waluta rachunku)':'Amount',
        'Kwota blokady/zwolnienie blokady':'Kwota blokady'
    })
    df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
    df_full = df_full[df_full['Date'].notna()]
    df_full['key'] = (df_full['Nr rachunku'].astype(str).fillna('')+'|'+df_full['Description']).map(clean_desc)
    # Uzupełnij kategorie dla df_full
    df_full['category'] = df_full['key'].map(lambda k: cat.map.get(k,("",""))[0])
    df_full['subcategory']= df_full['key'].map(lambda k: cat.map.get(k,("",""))[1])

    # Teraz filtr dat oraz reszta dla głównego df
    df = df_full.copy()
    mode = st.sidebar.radio("Tryb filtrowania",["Zakres dat","Pełny miesiąc"])
    if mode=="Zakres dat":
        mn,mx=df['Date'].min(),df['Date'].max()
        start,end=st.sidebar.date_input("Zakres",[mn,mx],mn,mx)
        df = df[(df['Date']>=start)&(df['Date']<=end)]
    else:
        yrs=sorted(df['Date'].dt.year.unique())
        meses={1:'Styczeń',2:'Luty',3:'Marzec',4:'Kwiecień',5:'Maj',
              6:'Czerwiec',7:'Lipiec',8:'Sierpień',9:'Wrzesień',
              10:'Październik',11:'Listopad',12:'Grudzień'}
        y=st.sidebar.selectbox("Rok",yrs,index=len(yrs)-1)
        mname=st.sidebar.selectbox("Miesiąc",list(meses.values()),index=6)
        m={v:k for k,v in meses.items()}[mname]
        df=df[(df['Date'].dt.year==y)&(df['Date'].dt.month==m)]

    # Bulk-assign na df (nie dotyka df_full)
    st.markdown("#### Krok 1: Przypisz kategorie")
    for idxs in df.groupby('key').groups.values():
        key=df.loc[idxs[0],'key']
        if key in cat.map and cat.map[key][0]: continue
        amt=df.loc[idxs[0],'Amount']
        st.write(f"**{key}** – {amt:.2f} PLN")
        s=cat.suggest(key,amt)
        sc=st.selectbox("Kat",list(CATEGORIES),index=list(CATEGORIES).index(s[0]),key=f"c_{key}")
        ss=st.selectbox("Pod",CATEGORIES[sc],index=CATEGORIES[sc].index(s[1]) if s[1] in CATEGORIES[sc] else 0,key=f"s_{key}")
        cat.assign(key,sc,ss)

    st.markdown("---"); st.success("Zapis assignments.csv")

    # Finalna tabela edytowalna
    df['category']=df['key'].map(lambda k:cat.map.get(k,("",""))[0])
    df['subcategory']=df['key'].map(lambda k:cat.map.get(k,("",""))[1])
    final=df[['Date','Description','Tytuł','Amount','Kwota blokady','category','subcategory']]
    st.markdown("## 🗃 Tabela transakcji")
    edited = st.data_editor(final,use_container_width=True,hide_index=True)
    if st.button("💾 Zapisz zmiany"):
        keys=df['key'].tolist()
        for i,r in enumerate(edited.itertuples(False)):
            cat.assign(keys[i],r.category,r.subcategory)
        st.success("Zapisano")

    # Raport tekstowy i YTD w jednej linii
    colA,colB = st.columns(2)
    with colA:
        # Raport
        ed=edited.copy()
        order=['Przychody']+sorted([c for c in CATEGORIES if c!='Przychody'])
        tot=pd.DataFrame({'category':order,'sum':0.0,'count':0}).set_index('category')
        if not ed.empty:
            inc=ed[(ed.category=='Przychody')&(ed.Amount>0)].groupby('category')['Amount'].agg(['sum','count'])
            exp=ed[(ed.category!='Przychody')&(ed.Amount<0)].groupby('category')['Amount'].agg(['sum','count'])
            tot.update(pd.concat([inc,exp]))
        tot=tot.reset_index(); tot['count']=tot['count'].astype(int)
        tot=tot[tot['count']>0]
        grp=ed.groupby(['category','subcategory'])['Amount'].agg(['sum','count']).reset_index()

        st.markdown("## 📊 Raport: ilość i suma wg kategorii")
        fmt=lambda v:f"{abs(v):,.2f}".replace(","," ")
        for _,r in tot.iterrows():
            with st.expander(f"{r.category} ({r.count}) – {fmt(r.sum)}"):
                for _,s in grp[grp.category==r.category].iterrows():
                    st.markdown(f"• **{s.subcategory}** ({s.count}) – {fmt(s.sum)}",unsafe_allow_html=True)

    with colB:
        # Oszczędności YTD obok
        ytd=df_full.copy()
        ytd=ytd[(ytd.category=='Oszczędności')&(ytd['Date'].dt.year==datetime.now().year)]
        total_ytd=ytd.Amount.sum()
        st.markdown(f"## 💰 Oszczędności YTD")
        st.markdown(f"**Łącznie: {total_ytd:,.2f} zł**".replace(",", " "))
        sub=ytd.groupby('subcategory')['Amount'].sum().reset_index().sort_values('Amount',ascending=False)
        for _,r in sub.iterrows():
            pct = r.Amount/total_ytd if total_ytd else 0
            label=f"**{r.subcategory}**<br>{pct:.0%}<br>{r.Amount:,.2f} zł"
            with st.expander(label, expanded=False):
                st.write(f"- {r.subcategory}: {r.Amount:,.2f} zł ({pct:.0%})".replace(",", " "))

    # Wykresy kołowe – jak wcześniej...

if __name__=="__main__":
    main()
