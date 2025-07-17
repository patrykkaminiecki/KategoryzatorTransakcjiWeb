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
            {"description": k, "category": c, "subcategory": s}
            for k,(c,s) in self.map.items()
        ]).to_csv(ASSIGNMENTS_FILE, index=False)

# ------------------------
# 4) WCZYTANIE CSV
# ------------------------
@st.cache_data
def load_bank_csv(u):
    raw = u.getvalue()
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

    # 1) Wczytanie pliku
    st.sidebar.header("Filtr dat")
    up = st.sidebar.file_uploader("CSV banku", type="csv")
    if not up:
        st.sidebar.info("Wczytaj plik CSV."); return
    try:
        df_raw = load_bank_csv(up)
    except Exception as e:
        st.error(str(e)); return

    # 2) Przygotowanie df_full (do YTD)
    df_full = df_raw.copy()
    df_full.columns = [c.strip() for c in df_full.columns]
    df_full = df_full.rename(columns={
        'Data transakcji':'Date','Dane kontrahenta':'Description','Tytuł':'Tytuł',
        'Nr rachunku':'Nr rachunku','Kwota transakcji (waluta rachunku)':'Amount',
        'Kwota blokady/zwolnienie blokady':'Kwota blokady'
    })
    df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
    df_full = df_full[df_full['Date'].notna()]
    df_full['key']     = (df_full['Nr rachunku'].astype(str).fillna('') + '|' + df_full['Description']).map(clean_desc)
    df_full['category']= df_full['key'].map(lambda k: cat.map.get(k,("",""))[0])
    df_full['subcategory']= df_full['key'].map(lambda k: cat.map.get(k,("",""))[1])

    # 3) Filtr dat dla głównego df
    df = df_full.copy()
    mode = st.sidebar.radio("Tryb filtrowania", ["Zakres dat","Pełny miesiąc"])
    if mode=="Zakres dat":
        mn,mx = df['Date'].min(), df['Date'].max()
        start,end = st.sidebar.date_input("Zakres dat", [mn.date(), mx.date()], min_value=mn.date(), max_value=mx.date())
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt   = datetime.combine(end,   datetime.max.time())
        df = df[(df['Date']>=start_dt)&(df['Date']<=end_dt)]
    else:
        yrs=sorted(df['Date'].dt.year.unique())
        meses={1:'Styczeń',2:'Luty',3:'Marzec',4:'Kwiecień',5:'Maj',
               6:'Czerwiec',7:'Lipiec',8:'Sierpień',9:'Wrzesień',
               10:'Październik',11:'Listopad',12:'Grudzień'}
        y=st.sidebar.selectbox("Rok",yrs,index=len(yrs)-1)
        mname=st.sidebar.selectbox("Miesiąc",list(meses.values()),index=6)
        m={v:k for k,v in meses.items()}[mname]
        df=df[(df['Date'].dt.year==y)&(df['Date'].dt.month==m)]

    # 4) Bulk-assign
    st.markdown("#### Krok 1: Przypisz kategorie")
    for idxs in df.groupby('key').groups.values():
        key = df.loc[idxs[0],'key']
        if key in cat.map and cat.map[key][0]:
            continue
        amt = df.loc[idxs[0],'Amount']
        st.write(f"**{key}** – {amt:.2f} PLN")
        s = cat.suggest(key, amt)
        sc = st.selectbox("Kategoria", list(CATEGORIES.keys()), index=list(CATEGORIES.keys()).index(s[0]), key=f"c_{key}")
        ss = st.selectbox("Podkategoria", CATEGORIES[sc],
                          index=CATEGORIES[sc].index(s[1]) if s[1] in CATEGORIES[sc] else 0,
                          key=f"s_{key}")
        cat.assign(key, sc, ss)

    st.markdown("---"); st.success("Zapis assignments.csv")

    # 5) Edytowalna tabela
    df['category']    = df['key'].map(lambda k:cat.map.get(k,("",""))[0])
    df['subcategory'] = df['key'].map(lambda k:cat.map.get(k,("",""))[1])
    final = df[['Date','Description','Tytuł','Amount','Kwota blokady','category','subcategory']]
    st.markdown("## 🗃 Tabela transakcji")
    edited = st.data_editor(final, use_container_width=True, hide_index=True)
    if st.button("💾 Zapisz zmiany"):
        keys = df['key'].tolist()
        for i,r in enumerate(edited.itertuples(False)):
            cat.assign(keys[i],r.category,r.subcategory)
        st.success("Zapisano")

    # 6) Raport tekstowy + YTD obok
    colA,colB = st.columns(2, gap="medium")
    with colA:
        ed=edited.copy()
        order=['Przychody']+sorted([c for c in CATEGORIES if c!='Przychody'])
        total=pd.DataFrame({'category':order,'sum':0.0,'count':0}).set_index('category')
        if not ed.empty:
            inc=ed[(ed.category=='Przychody')&(ed.Amount>0)].groupby('category')['Amount'].agg(['sum','count'])
            exp=ed[(ed.category!='Przychody')&(ed.Amount<0)].groupby('category')['Amount'].agg(['sum','count'])
            total.update(pd.concat([inc,exp]))
        total=total.reset_index(); total['count']=total['count'].astype(int)
        total=total[total['count']>0]
        grouped=ed.groupby(['category','subcategory'])['Amount'].agg(['sum','count']).reset_index()
        st.markdown("## 📊 Raport: ilość i suma wg kategorii")
        fmt=lambda v: f"{abs(v):,.2f}".replace(",", " ")
        for _,r in total.iterrows():
            cat_label = r['category']
            cnt = r['count']
            ssum = fmt(r['sum'])
            with st.expander(f"{cat_label} ({cnt}) – {ssum}"):
                subs = grouped[grouped['category']==cat_label]
                for __,s in subs.iterrows():
                    st.markdown(f"• **{s['subcategory']}** ({int(s['count'])}) – {fmt(s['sum'])}", unsafe_allow_html=True)

    with colB:
        st.markdown(f"## 💰 Oszczędności YTD ({datetime.now().year})")
        ytd = df_full[(df_full['category']=='Oszczędności') & (df_full['Date'].dt.year==datetime.now().year)]
        total_ytd = ytd['Amount'].sum()
        st.markdown(f"**Łącznie: {total_ytd:,.2f} zł**".replace(",", " "))
        sub = ytd.groupby('subcategory')['Amount'].sum().reset_index().sort_values('Amount',ascending=False)
        for _,r in sub.iterrows():
            pct = (r['Amount']/total_ytd) if total_ytd else 0
            label = f"**{r['subcategory']}**\n{pct:.0%}\n{r['Amount']:,.2f} zł"
            with st.expander(label, expanded=False):
                st.write(f"- {r['subcategory']}: {r['Amount']:,.2f} zł ({pct:.0%})")

    # 7) Wykresy kołowe
    st.markdown("## 📈 Wykresy kołowe")
    # kategorie
    tot = total.reset_index(drop=True)
    colors = ["#2ca02c" if c=="Przychody" else "#d62728" for c in tot['category']]
    fig_cat = go.Figure(data=[go.Pie(
        labels=tot['category'], values=tot['sum'].abs(),
        marker=dict(colors=colors,line=dict(color='#111',width=3)),
        hole=0.3, domain=dict(x=[0.2,0.8],y=[0.2,0.8]),
        textposition='outside',
        texttemplate='<b>%{label}</b><br>%{percent:.0%}<br>%{value:,.2f} zł',
        textfont=dict(size=14,color='white'),
        pull=[0.02]*len(tot), hoverinfo='none'
    )])
    fig_cat.update_traces(textposition='outside')
    fig_cat.update_layout(height=450, showlegend=False,
                          paper_bgcolor='#111', plot_bgcolor='#111', font_color='white',
                          margin=dict(l=80,r=80,t=40,b=80))
    st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar":False})

    # podkategorie - cały zestaw
    fig_sub = go.Figure(data=[go.Pie(
        labels=grouped['subcategory'], values=grouped['sum'].abs(),
        marker=dict(line=dict(color='#111',width=2)),
        hole=0.3, domain=dict(x=[0.2,0.8],y=[0.2,0.8]),
        textposition='outside',
        texttemplate='<b>%{label}</b><br>%{percent:.0%}<br>%{value:,.2f} zł',
        textfont=dict(size=14,color='white'),
        pull=[0.02]*len(grouped), hoverinfo='none'
    )])
    fig_sub.update_traces(textposition='outside')
    fig_sub.update_layout(title="Podkategorie (całość)", height=450,
                          showlegend=False,
                          paper_bgcolor='#111', plot_bgcolor='#111', font_color='white',
                          margin=dict(l=80,r=80,t=40,b=80))
    st.plotly_chart(fig_sub, use_container_width=True, config={"displayModeBar":False})

if __name__=="__main__":
    main()
