import pandas as pd
import numpy as np
import sqlite3 
import torch
from transformers import pipeline



df = pd.read_csv("matkassen_data.csv")

# byta namna på kolumen kasstyp: kasstyp till kassatyp
def clean_kassatyp_column(df):
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.rename(columns={'kasstyp': 'kassatyp'})
    return df_cleaned
df = clean_kassatyp_column(df)

print(df["kassatyp"].unique())

# tar bort stor bokstav 
def clean_kassatyp(df):
    df_cleaned = df.copy()
    df_cleaned['kassatyp'] = df_cleaned['kassatyp'].str.lower()
    return df_cleaned
df = clean_kassatyp(df)
print(df["kassatyp"].unique())

# mapping för ordbetydelse i kassatyp, har gjort fritolkning på visa ord
def map_kassatyp(df):
    df_cleaned = df.copy()
    mapping = {
        'vegetarian': 'vegetariskkasse',
        'veggie': 'vegetariskkasse',
        'vegetarisk': 'vegetariskkasse',
        'familjekassen': 'familjekasse',
        'familj': 'familjekasse',
        'family': 'familjekasse',
        'klassisk': 'klassiskkasse',
        'classic': 'klassiskkasse',
        'standard': 'klassiskkasse',
        'veg': 'vegetariskkasse',
        'express': 'expresskasse',
        'snabb & enkel': 'expresskasse',
        '30-min': 'expresskasse',
        'snabb': 'expresskasse',
        'quick': 'expresskasse'
    }
    # Dessa två rader MÅSTE vara indragna för att tillhöra funktionen:
    df_cleaned['kassatyp'] = df_cleaned['kassatyp'].replace(mapping)
    return df_cleaned

# Nu kan du anropa den (längst ut till vänster):

df = map_kassatyp(df)
print(df["kassatyp"].unique())

# tar bort citattecken och gör dem till heltal i antal_portioner
def antal_portioner_nummer(df):
    df_cleaned = df.copy()
    ord_till_num = {'två': '2', 'fyra': '4', 'sex': '6'}
    
    # 1. Standardisera texten
    s = df_cleaned['antal_portioner'].astype(str).str.lower()
    
    # 2. Byt ut ord mot siffer-strängar
    for word, num in ord_till_num.items():
        s = s.replace(word, num)
    
    # 3. Extrahera siffror men BEHÅLL NaN (ingen fillna eller astype int)
    # Vi använder float för att kunna räkna korrekt trots tomma fält
    df_cleaned['antal_portioner'] = s.str.extract(r'(\d+)').astype(float)

    return df_cleaned


df = antal_portioner_nummer(df)
print(df["antal_portioner"])


def clean_leveransvecka(df):
    df_cleaned = df.copy()
    
    def format_week(val):
        val = str(val).lower().strip()
        if val == 'nan' or val == '':
            return None
            
        # 1. Extrahera alla siffror från strängen
        import re
        numbers = re.findall(r'\d+', val)
        
        if not numbers:
            return val
            
        # 2. Logik för att hitta vecka och år
        if len(numbers) == 1:
            # Bara ett tal (t.ex. '29' eller 'v29') -> Anta vecka och år 2024
            week = numbers[0].zfill(2)
            year = "2024"
        elif len(numbers) >= 2:
            # Två tal (t.ex. '2024' och '27')
            # Vi antar att det långa talet är år och det korta är vecka
            n1, n2 = numbers[0], numbers[1]
            year = n1 if len(n1) == 4 else n2
            week = n2 if len(n1) == 4 else n1
            week = week.zfill(2)
            
        return f"vecka {week} - {year}"

    df_cleaned['leveransvecka'] = df_cleaned['leveransvecka'].apply(format_week)
    return df_cleaned

# Anropa i din pipeline:
df = clean_leveransvecka(df)
print(sorted(df["leveransvecka"].dropna().unique()))
print(df["leveransvecka"])

# ändrar till datetime och skapar leveransvecka från datum
def clean_leveransdatum(df):
    df_clean = df.copy()
    df_clean["leveransdatum"] = pd.to_datetime(df_clean["leveransdatum"], errors='coerce')
    # ... din veckologik ...
    return df_clean

print(df["leveransdatum"].head(100))
# gör om potnr till ett format: 12345 i postnummer kolumnen samt Nan till okänt
def clean_postnummer(df):
    df_clean = df.copy()
    
    # 1. Rensa och formatera
    s = df_clean["postnummer"].astype(str).str.replace(r'SE-|S-| |-|\.', '', regex=True)
    s = s.str.slice(0, 5)
    
    # 2. Sista steget: Ersätt alla typer av tomrum/nan med en riktig sträng
    # Detta gör att kolumnens dtype blir en ren sträng (object)
    df_clean["postnummer"] = s.replace('nan', 'okänt').fillna('okänt')
    
    return df_clean

# NU KAN DU SKRIVA SÅ HÄR UTAN ATT DET KRASCHAR:
df = clean_postnummer(df)
print(df["postnummer"].head(1000))

## tar bort kr, SEK, :- och gör om till heltal i veckapris samt byter namn till "veckopris""
def clean_veckopris(df):
    df_clean = df.copy()
    df_clean = df_clean.rename(columns={'veckapris': 'veckopris'})
    
    # 1. Rensa text men behåll siffror och decimaltecken
    s = df_clean["veckopris"].astype(str).str.replace(r'[^\d,.]', '', regex=True)
    s = s.str.replace(',', '.')
    
    # 2. Konvertera till siffror. 'coerce' gör att skräp/tomma fält blir NaN (helt rätt!)
    # Vi tar bort .fillna(0.0) härifrån
    df_clean["veckopris"] = pd.to_numeric(s, errors='coerce')
    
    return df_clean
df = clean_veckopris(df)
# Anropa i din pipeline
print(df["veckopris"].head())

# gör om leveransstatus till boolean
def clean_leveransstatus_bool(df):
    df_clean = df.copy()
    
    # 1. Standardisera texten
    s = df_clean["leveransstatus"].astype(str).str.lower().str.strip()
    
    # 2. Definiera vad som är en lyckad leverans
    delivered_keywords = ['levererad', 'delivered', 'levered', 'ok', 'ja']
    
    # 3. Skapa en boolean-kolumn (True om levererad, annars False)
    df_clean["leveransstatus"] = s.isin(delivered_keywords)
    
    return df_clean

# Kom ihåg anropet!
df = clean_leveransstatus_bool(df)
print(df["leveransstatus"].head(20))

# byter dtype till daytimme för pren_startdatum
def clean_pren_startdatum(df):
    df_clean = df.copy()
    df_clean["pren_startdatum"] = pd.to_datetime(df_clean["pren_startdatum"], errors='coerce')
    return df_clean

# Anropa funktionen
df = clean_pren_startdatum(df)
print(df["pren_startdatum"].head(20))

# byter dtype i paus_från till till datetime
def clean_paus_fran_till(df):
    df_clean = df.copy()
    
    # 1. Konvertera till datetime med errors='coerce'
    df_clean["paus_från"] = pd.to_datetime(df_clean["paus_från"], errors='coerce')
    df_clean["paus_till"] = pd.to_datetime(df_clean["paus_till"], errors='coerce')
    
    return df_clean
df = clean_paus_fran_till(df)
print(df["paus_från"].head(20))
print(df["paus_till"].head(20))

# byter dtype i avslutad_datum till datetime
def clean_pren_avslutsdatum(df):
    df_clean = df.copy()
    
    # 1. Konvertera till datetime med errors='coerce'
    df_clean["pren_avslutsdatum"] = pd.to_datetime(df_clean["pren_avslutsdatum"], errors='coerce')
    
    return df_clean
df = clean_pren_avslutsdatum(df)
print(df["pren_avslutsdatum"].head(20))

# mapping för ordbetydelse i kostpreferens, och gjort nan till okänt
def clean_kostpreferens(df):
    df_clean = df.copy()
    
    # 1. Standardisera texten till små bokstäver först
    s = df_clean["kostpreferens"].fillna("okänt").astype(str).str.lower().str.strip()
    
    # 2. Mapping (Notera att alla 'nycklar' nu är små bokstäver eftersom s är .lower())
    mapping = {
        'laktosfri': 'laktosfri',
        'glutenfri': 'glutenfri',
        'nötfri': 'nötfri',
        'fläskfri': 'fläskfri',
        'gf': 'glutenfri',
        'ingen preferens': 'ingen preferens',
        'inga': 'ingen preferens',
        'unknown': 'ingen preferens',
        'standard': 'ingen preferens',
        'nut free': 'nötfri',
        'ingen fläsk': 'fläskfri',
        'normal': 'ingen preferens',
        'lactose free': 'laktosfri',
        'lf': 'laktosfri',
        'nf': 'nötfri',
        'gluten free': 'glutenfri',
        'nötter': 'nötfri',
        'pork free': 'fläskfri'
    } # Kommatecken tillagda mellan varje rad ovan!
    
    # 3. VIKTIGT: Denna rad måste vara indragen (ett tab-steg) för att tillhöra funktionen
    df_clean["kostpreferens"] = s.replace(mapping)
    
    return df_clean

# Anropa och se resultatet
df = clean_kostpreferens(df)
print(df["kostpreferens"].unique())


# fyller tomma värden i omdöme_text med "ingen kommentar"
def clean_omdome_text(df):
    df_clean = df.copy()
    
    # 1. Skapa engagemangs-feature (1=skrivit, 0=inte skrivit)
    # Vi använder namnet 'omdome_fylld' men det är samma logik som din
    df_clean['omdome_fylld'] = df_clean['omdöme_text'].notnull().astype(int)
    
    # 2. Städa original-texten
    df_clean["omdöme_text"] = df_clean["omdöme_text"].fillna("ingen kommentar").astype(str)
    
    # 3. Räkna ut längden på omdömet
    df_clean['omdome_langd'] = df_clean['omdöme_text'].apply(lambda x: len(x) if x != "ingen kommentar" else 0)
    
    return df_clean

# ANROP
df = clean_omdome_text(df)

# Kontrollera resultatet (visar de första raderna där det tidigare var tomt)
print(df[["omdöme_text"]].head(10))

def clean_omdömesdatum(df):
    # Skapar en kopia för att undvika varningar
    df_clean = df.copy()
    
    # Konverterar till datetime64[ns] som tillåter beräkningar
    df_clean['omdömesdatum'] = pd.to_datetime(df['omdömesdatum'], errors='coerce')
    
    return df_clean
df = clean_omdömesdatum(df)
print(df["omdömesdatum"].head(20))

# gör så att nan blir till numeric i omdömesbetyg
def clean_omdömesbetyg(df):
    df_clean = df.copy()
    
    # 1. Konvertera till siffror (float) för att kunna räkna
    # Vi använder errors='coerce' för att göra konstig text till NaN
    df_clean['omdömesbetyg'] = pd.to_numeric(df_clean['omdömesbetyg'], errors='coerce')
    
    # Vi hoppar över .fillna(0) för att inte sänka medelvärdet!
    
    return df_clean

# featuring engineering: 

def add_kunalder_dagar(df):
    df_feat = df.copy()
    
    # 1. Tvinga båda till datetime (detta löser TypeError-problemet)
    df_feat['leveransdatum'] = pd.to_datetime(df_feat['leveransdatum'], errors='coerce')
    df_feat['pren_startdatum'] = pd.to_datetime(df_feat['pren_startdatum'], errors='coerce')
    
    # 2. Utför beräkningen
    # .dt.days omvandlar tidsskillnaden till ett rent heltal (siffror)
    df_feat['feat_kundalder_dagar'] = (df_feat['leveransdatum'] - df_feat['pren_startdatum']).dt.days
    
    # 3. Fyll tomma värden (NaN) med 0 så att du kan räkna på kolumnen
    df_feat['feat_kundalder_dagar'] = df_feat['feat_kundalder_dagar'].fillna(0).astype(int)
    
    return df_feat

# ANROP
df = add_kunalder_dagar(df)
print(df["feat_kundalder_dagar"].head(20))

# hur länge en paus kunde har haft 

def feat_paus_langd(df):
    df_feat = df.copy()
    
    # 1. Räkna ut skillnaden mellan 'till' och 'från'
    # .dt.days gör om tidsskillnaden till ett rent heltal (antal dagar)
    df_feat['feat_paus_antal_dagar'] = (df_feat['paus_till'] - df_feat['paus_från']).dt.days
    
    # 2. Hantera de som inte har pausat (NaN) genom att sätta dem till 0 dagar
    df_feat['feat_paus_antal_dagar'] = df_feat['feat_paus_antal_dagar'].fillna(0).astype(int)
    
    # 3. Säkerhetskoll: Om 'paus_till' råkar vara före 'paus_från' sätter vi 0
    df_feat.loc[df_feat['feat_paus_antal_dagar'] < 0, 'feat_paus_antal_dagar'] = 0
    
    return df_feat

# ANROP (Kör efter clean_paus_fran_till)
df = feat_paus_langd(df)
print(df["feat_paus_antal_dagar"].head(20))

def feat_churn_status_avslutat(df):
    df_feat = df.copy()
    
    # 1. Kolla om det finns ett värde (notnull)
    # 2. .astype(int) gör om True till 1 och False till 0
    df_feat['har_avslutat'] = df_feat['pren_avslutsdatum'].notnull().astype(int)
    
    return df_feat

# ANROP (Viktigt: Kör clean_pren_avslutsdatum först!)
df = feat_churn_status_avslutat(df)

# Kolla fördelningen - hur många har slutat vs är kvar?
print(df['har_avslutat'].value_counts())

# pris per portion
def feat_pris_per_portion(df):
    df_feat = df.copy()
    
    # 1. Räkna ut pris per portion
    # Vi använder .replace(0, np.nan) på nämnaren för att undvika "division med noll"-fel
    df_feat['pris_per_portion'] = df_feat['veckopris'] / df_feat['antal_portioner'].replace(0, np.nan)
    
    # 2. (Valfritt) Avrunda till två decimaler för att det ska se snyggt ut
    df_feat['pris_per_portion'] = df_feat['pris_per_portion'].round(2)
    
    return df_feat

# ANROP (Viktigt: Kör clean_veckopris och antal_portioner_nummer FÖRST!)
df = clean_veckopris(df)
df = antal_portioner_nummer(df)
df = feat_pris_per_portion(df)

# Kontrollera resultatet
print(df[['kassatyp', 'veckopris', 'antal_portioner', 'pris_per_portion']].head(10))

# delara upp i region för att se vart i sverige kunder handlar

def add_region_feature(df):
    df_feat = df.copy()
    # Denna rad skapar kolumnen 'region' som nästa funktion behöver
    df_feat["region"] = df_feat["postnummer"].astype(str).str[:2]
    df_feat["region"] = df_feat["region"].replace('ok', 'okänt')
    return df_feat

def add_stadnamn_feature(df):
    df_feat = df.copy()
    stad_map = {
        '10': 'Storstockholm', '11': 'Storstockholm', '12': 'Storstockholm', 
        '13': 'Storstockholm', '14': 'Storstockholm', '15': 'Storstockholm', 
        '16': 'Storstockholm', '17': 'Storstockholm', '18': 'Storstockholm', '19': 'Storstockholm',
        '20': 'Malmö', '21': 'Malmö',
        '40': 'Göteborg', '41': 'Göteborg', '42': 'Göteborg', '43': 'Göteborg', '44': 'Göteborg'
    }
    # Här använder vi kolumnen 'region' som skapades ovan
    df_feat['stad_namn'] = df_feat['region'].map(stad_map)
    df_feat['stad_namn'] = df_feat['stad_namn'].fillna('Övriga Sverige')
    return df_feat

# --- KÖRNING (PIPELINE) ---

df = clean_postnummer(df)      # 1. Städa (t.ex. "S-112 34" -> "11234")
df = add_region_feature(df)    # 2. Extrahera (t.ex. "11234" -> "11")
df = add_stadnamn_feature(df)  # 3. Mappa (t.ex. "11" -> "Storstockholm")

# Nu kan du se resultatet av kedjereaktionen
print(df[['postnummer', 'region', 'stad_namn']].head(20))

# Sentimentanalys av omdöme_text


def add_kblab_sentiment(df):
    df_feat = df.copy()
    
    # 1. Kolla hårdvaran (din demo-logik)
    device = 0 if torch.cuda.is_available() else -1
    
    # 2. Ladda modellen med rätt device
    print(f"Laddar KBLab-modellen på {'GPU' if device == 0 else 'CPU'}...")
    classifier = pipeline(
        "sentiment-analysis",
        model="KBLab/robust-swedish-sentiment-multiclass",
        device=device
    )

    def get_sentiment(text):
        # Snabb-check: hoppa över tomma/korta texter
        if pd.isna(text) or text == "ingen kommentar" or len(str(text)) < 3:
            return "NEUTRAL", 0.5
        
        try:
            result = classifier(str(text))
            # Modellen returnerar en lista, t.ex. [{'label': 'POSITIVE', 'score': 0.99}]
            return result[0]['label'], result[0]['score']
        except:
            return "NEUTRAL", 0.5

    # 3. Kör analysen
    print("Analyserar sentiment på omdömen...")
    # Vi sparar resultaten i en temporär kolumn
    sentiment_results = df_feat['omdöme_text'].apply(get_sentiment)
    
    # 4. Dela upp resultaten i Label och Score
    df_feat['sentiment_label'] = sentiment_results.apply(lambda x: x[0])
    df_feat['sentiment_score'] = sentiment_results.apply(lambda x: x[1])
    
    # 5. Mapping till siffror (för att kunna räkna medelvärde)
    mapping = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
    df_feat['feat_sentiment_index'] = df_feat['sentiment_label'].map(mapping)
    
    return df_feat
df_test = add_kblab_sentiment(df.head(5))

# Visa resultatet
print(df_test[['omdöme_text', 'sentiment_label', 'feat_sentiment_index']])

#
def add_sentiment_features(df):
    df_feat = df.copy()
    
    # Förutsättning: Du har kört din add_kblab_sentiment(df) innan denna
    # så att kolumnerna 'sentiment_label' och 'feat_sentiment_index' finns.

    # --- 1. Sentiment-Betyg Gap (Viktigaste!) ---
    # Hittar kunder där texten och betyget inte stämmer överens.
    # Ex: Skriver "Jättegott" (1) men ger betyg 1.
    df_feat['feat_sentiment_mismatch'] = 0
    
    # Logik: Negativ text (-1) men betyg 4-5 ELLER Positiv text (1) men betyg 1-2
    mismatch_mask = ((df_feat['feat_sentiment_index'] == -1) & (df_feat['omdömesbetyg'] >= 4)) | \
                    ((df_feat['feat_sentiment_index'] == 1) & (df_feat['omdömesbetyg'] <= 2))
    
    df_feat.loc[mismatch_mask, 'feat_sentiment_mismatch'] = 1

    # --- 2. Extremt Sentiment (Styrka) ---
    # Hittar kunder som är väldigt tydliga i sin feedback (Score > 0.95)
    df_feat['feat_extreme_sentiment'] = 0
    df_feat.loc[df_feat['sentiment_score'] > 0.95, 'feat_extreme_sentiment'] = 1

    # --- 3. Negativ Trend-Varning ---
    # Skapar en ren flagga för negativa omdömen för enkel aggregering
    df_feat['feat_is_negative'] = (df_feat['feat_sentiment_index'] == -1).astype(int)

    return df_feat

df_small = df.head(15).copy()

# 2. Kör AI-analysen
df_small = add_kblab_sentiment(df_small)

# 3. Kör Feature Engineering
df_small = add_sentiment_features(df_small)

# 4. Printa resultatet
print(df_small[['omdöme_text', 'sentiment_label', 'feat_sentiment_mismatch']].head(15))

print(df.columns)
print(df.head(20))

print(df.columns)

# load till sql

def load_to_sqlite(df, db_name="matkassen_data.db", table_name="processed_data"):
    # 1. Skapa anslutning (filen skapas i din mapp)
    conn = sqlite3.connect(db_name)
    
    # 2. Spara DataFrame till SQL
    # if_exists='replace' gör att tabellen skrivs över varje gång du testar
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    # 3. VERIFIERING: Läs tillbaka antal rader (viktigt för uppgiften!)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    
    print(f"--- LOAD KLART ---")
    print(f"Data sparad i filen: {db_name}")
    print(f"Verifiering: {count} rader har laddats upp korrekt.")
    
    conn.close()

# ANROP
load_to_sqlite(df)

# test sql

df_small = df.head(15).copy()
df_small = add_kblab_sentiment(df_small)
df_small = add_sentiment_features(df_small)

# 2. KÖR LOAD (Spara df_small istället för df)
# Här skickar vi in df_small som nu innehåller 'stad_namn' och 'feat_sentiment_index'
load_to_sqlite(df_small)

conn = sqlite3.connect("matkassen_data.db")
df_kontroll = pd.read_sql("SELECT * FROM processed_data LIMIT 5", conn)
print("\n--- VERIFIERING: DATA LÄST FRÅN SQLITE ---")
print(df_kontroll[['leverans_id', 'stad_namn', 'feat_sentiment_index']])
conn.close()

# test valideringsdataset

print("\n" + "="*50)
print("KÖR PIPELINE PÅ VALIDERINGSDATA")
print("="*50)

# 1. Extract: Läs in den nya filen
df_val_raw = pd.read_csv("matkassen_validation.csv")

# 2. Transform: Kör alla dina funktioner (använd samma ordning som förut)
# Här använder vi din befintliga logik på den nya datan
df_val = clean_kassatyp_column(df_val_raw)
df_val = map_kassatyp(df_val)
df_val = antal_portioner_nummer(df_val)
df_val = clean_leveransdatum(df_val)
df_val = clean_postnummer(df_val)
df_val = clean_veckopris(df_val)
df_val = clean_omdome_text(df_val)
df_val = add_kunalder_dagar(df_val)
df_val = add_region_feature(df_val)
df_val = add_stadnamn_feature(df_val)

# Vi kör BERT-analysen på ett urval av valideringsdatan för att spara tid
df_val_test = add_kblab_sentiment(df_val.head(15))
df_val_test = add_sentiment_features(df_val_test)

# 3. Load: Spara till en separat tabell i databasen
load_to_sqlite(df_val_test, table_name="validation_results")

# 4. Verifiera resultatet
conn = sqlite3.connect("matkassen_data.db")
df_val_check = pd.read_sql("SELECT leverans_id, stad_namn, feat_sentiment_index FROM validation_results", conn)
print("\n--- RESULTAT FRÅN VALIDERINGSDATA ---")
print(df_val_check.head(10))
conn.close()






