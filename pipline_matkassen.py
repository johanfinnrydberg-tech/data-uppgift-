import pandas as pd
import numpy as np
import sqlite3 
import torch
from transformers import pipeline
device = 0 if torch.cuda.is_available() else -1
print(f"Förbereder KBLab-modellen på {'GPU' if device == 0 else 'CPU'}...")

classifier = pipeline(
    "sentiment-analysis",
    model="KBLab/robust-swedish-sentiment-multiclass",
    device=device
)

df = pd.read_csv("matkassen_data.csv")
df_validation = pd.read_csv("matkassen_validation.csv")

# Datatvätt: 

# 1. byta namna på kolumen kasstyp: kasstyp till kassatyp
def clean_kassatyp_column(df):
    
    df = df.rename(columns={'kasstyp': 'kassatyp'})
    return df

# 2. tar bort stor bokstav 
def clean_kassatyp(df):
    
    df['kassatyp'] = df['kassatyp'].str.lower()
    return df

# 3. mapping för ordbetydelse i kassatyp, har gjort fritolkning på visa ord
def map_kassatyp(df):
    
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
    
    df['kassatyp'] = df['kassatyp'].replace(mapping)
    return df

# 4. tar bort citattecken och gör dem till heltal i antal_portioner
def antal_portioner_nummer(df):
    
    ord_till_num = {'två': '2', 'fyra': '4', 'sex': '6'}
    
    # Standardisera texten
    s = df['antal_portioner'].astype(str).str.lower()
    
    # Byt ut ord mot siffer-strängar
    for word, num in ord_till_num.items():
        s = s.replace(word, num)
    
    # Extrahera siffror men behåller NaN (ingen fillna eller astype int)
    # Vi använder float för att kunna räkna korrekt trots tomma fält
    df['antal_portioner'] = s.str.extract(r'(\d+)').astype(float)

    return df

# 5. Standardisera Leveransvecka
def clean_leveransvecka(df):

    
    def format_week(val):
        val = str(val).lower().strip()
        if val == 'nan' or val == '':
            return None
            
        # Extrahera alla siffror från strängen
        import re
        numbers = re.findall(r'\d+', val)
        
        if not numbers:
            return val
            
        # Logik för att hitta vecka och år
        if len(numbers) == 1:
            # Bara ett tal (t.ex. '29' eller 'v29') -> Anta vecka och år 2024
            week = numbers[0].zfill(2)
            year = "2024"
        elif len(numbers) >= 2:
            
            # Vi antar att det långa talet är år och det korta är vecka
            n1, n2 = numbers[0], numbers[1]
            year = n1 if len(n1) == 4 else n2
            week = n2 if len(n1) == 4 else n1
            week = week.zfill(2)
            
        return f"vecka {week} - {year}"

    df['leveransvecka'] = df['leveransvecka'].apply(format_week)
    return df

# 6. ändrar till datetime och skapar leveransvecka från datum
def clean_leveransdatum(df):
    
    df["leveransdatum"] = pd.to_datetime(df["leveransdatum"], errors='coerce')
    # ... din veckologik ...
    return df

# 7. gör om postnr till ett format: 12345 i postnummer kolumnen samt Nan till okänt
def clean_postnummer(df):
    
    #  Rensa och formatera
    s = df["postnummer"].astype(str).str.replace(r'SE-|S-| |-|\.', '', regex=True)
    s = s.str.slice(0, 5)
    
    # Ersätt alla typer av tomrum/nan med en riktig sträng
    
    df["postnummer"] = s.replace('nan', 'okänt').fillna('okänt')
    
    return df

## 8. tar bort kr, SEK, :- och gör om till decimaler i veckapris samt byter namn till "veckopris""
def clean_veckopris(df):
    
    df = df.rename(columns={'veckapris': 'veckopris'})
    
    #  behåller siffror och decimaltecken
    s = df["veckopris"].astype(str).str.replace(r'[^\d,.]', '', regex=True)
    s = s.str.replace(',', '.')
    
    #  Konvertera till siffror. 
    # Vi tar bort .fillna(0.0) härifrån
    df["veckopris"] = pd.to_numeric(s, errors='coerce')
    
    return df

# 9. gör om leveransstatus till boolean
def clean_leveransstatus_bool(df):
    
    #  Standardisera texten
    s = df["leveransstatus"].astype(str).str.lower().str.strip()
    
    #  Definiera vad som är en lyckad leverans
    delivered_keywords = ['levererad', 'delivered', 'levered', 'ok', 'ja']
    
    #  Skapa en boolean-kolumn (True om levererad, annars False)
    df["leveransstatus"] = s.isin(delivered_keywords)
    
    return df

# 10. byter dtype till daytimme för pren_startdatum
def clean_pren_startdatum(df):
    
    df["pren_startdatum"] = pd.to_datetime(df["pren_startdatum"], errors='coerce')
    return df

# 11. byter dtype i paus_från och paus_till till datetime
def clean_paus_fran_till(df):

    # Konvertera till datetime med errors='coerce'
    df["paus_från"] = pd.to_datetime(df["paus_från"], errors='coerce')
    df["paus_till"] = pd.to_datetime(df["paus_till"], errors='coerce')
    
    return df

# 12. byter dtype i avslutad_datum till datetime
def clean_pren_avslutsdatum(df):
    
    # Konvertera till datetime med errors='coerce'
    df["pren_avslutsdatum"] = pd.to_datetime(df["pren_avslutsdatum"], errors='coerce')
    
    return df

# 12. mapping för ordbetydelse i kostpreferens, och gjort nan till okänt
def clean_kostpreferens(df):

    #  Standardisera texten till små bokstäver först
    s = df["kostpreferens"].fillna("okänt").astype(str).str.lower().str.strip()
    
    #  Mapping 
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
    } 
    
    
    df["kostpreferens"] = s.replace(mapping)
    
    return df

# 13. fyller tomma värden i omdöme_text med "ingen kommentar"
def clean_omdome_text(df):
    
    #  Skapar engagemangs-feature (1=skrivit, 0=inte skrivit)
    # Vi använder namnet 'omdome_fylld' 
    df['omdome_fylld'] = df['omdöme_text'].notnull().astype(int)
    
    #  Städar original-texten
    df["omdöme_text"] = df["omdöme_text"].fillna("ingen kommentar").astype(str)
    
    #  Räknar ut längden på omdömet
    df['omdome_langd'] = df['omdöme_text'].apply(lambda x: len(x) if x != "ingen kommentar" else 0)
    
    return df

# 14. gör om omdömesdatum till datetime64
def clean_omdömesdatum(df):
    
    # Konverterar till datetime64 som tillåter beräkningar
    df['omdömesdatum'] = pd.to_datetime(df['omdömesdatum'], errors='coerce')
    
    return df

# 15. gör så att nan blir till numeric i omdömesbetyg
def clean_omdömesbetyg(df):

    #  Konvertera till siffror (float) för att kunna räkna
    # Vi använder errors='coerce' för att göra konstig text till NaN
    df['omdömesbetyg'] = pd.to_numeric(df['omdömesbetyg'], errors='coerce')
    
    # Vi hoppar över .fillna(0) för att inte sänka medelvärdet!
    
    return df


# featuring engineering: 

# 1. Beräkning av kundlivslängd (Kundålder)
def add_kunalder_dagar(df):
    
    # Säkerställ att datumkolumnerna är i datetime-format
    df['leveransdatum'] = pd.to_datetime(df['leveransdatum'], errors='coerce')
    df['pren_startdatum'] = pd.to_datetime(df['pren_startdatum'], errors='coerce')
    
    #  Utför beräkning
    # .dt.days omvandlar tidsskillnaden till ett rent heltal (siffror)
    df['feat_kundalder_dagar'] = (df['leveransdatum'] - df['pren_startdatum']).dt.days
    
    #  Fyller tomma värden (NaN) med 0 så att vi kan räkna på kolumnen
    df['feat_kundalder_dagar'] = df['feat_kundalder_dagar'].fillna(0).astype(int)
    
    return df

# 2. hur länge en kund har haft paus

def feat_paus_langd(df):

    # Räknar ut skillnaden mellan 'till' och 'från'
    # .dt.days gör om tidsskillnaden till ett rent heltal (antal dagar)
    df['feat_paus_antal_dagar'] = (df['paus_till'] - df['paus_från']).dt.days
    
    # Hantera de som inte har pausat (NaN) genom att sätta dem till 0 dagar
    df['feat_paus_antal_dagar'] = df['feat_paus_antal_dagar'].fillna(0).astype(int)
    
    #  Säkerhetskoll: Om 'paus_till' råkar vara före 'paus_från' sätter vi 0
    df.loc[df['feat_paus_antal_dagar'] < 0, 'feat_paus_antal_dagar'] = 0
    
    return df

# 3. Identifiering av avslutade prenumerationer(dummy-variabel)
def feat_churn_status_avslutat(df):
    
    #  Kollar om det finns ett värde (notnull)
    #  .astype(int) gör om True till 1 och False till 0
    df['har_avslutat'] = df['pren_avslutsdatum'].notnull().astype(int)
    
    return df

# 4. pris per portion
def feat_pris_per_portion(df):
    
    #  Räknar ut pris per portion
    # Vi använder .replace(0, np.nan) på nämnaren för att undvika "division med noll"-fel
    df['pris_per_portion'] = df['veckopris'] / df['antal_portioner'].replace(0, np.nan)
    
    #  Avrundar till två decimaler för att det ska se snyggt ut
    df['pris_per_portion'] = df['pris_per_portion'].round(2)
    
    return df

# 5. delara upp i region för att se vart i sverige kunder handlar samt vilken stad

def add_region_feature(df):
    
    # Denna rad skapar kolumnen 'region' som nästa funktion behöver
    df["region"] = df["postnummer"].astype(str).str[:2]
    df["region"] = df["region"].replace('ok', 'okänt')
    return df

def add_stadnamn_feature(df):
    
    stad_map = {
        '10': 'Storstockholm', '11': 'Storstockholm', '12': 'Storstockholm', 
        '13': 'Storstockholm', '14': 'Storstockholm', '15': 'Storstockholm', 
        '16': 'Storstockholm', '17': 'Storstockholm', '18': 'Storstockholm', '19': 'Storstockholm',
        '20': 'Malmö', '21': 'Malmö',
        '40': 'Göteborg', '41': 'Göteborg', '42': 'Göteborg', '43': 'Göteborg', '44': 'Göteborg'
    }
    # Här använder vi kolumnen 'region' som skapades ovan
    df['stad_namn'] = df['region'].map(stad_map)
    df['stad_namn'] = df['stad_namn'].fillna('Övriga Sverige')
    return df

# 6. #Sentimentanalys av omdöme_text

def add_kblab_sentiment(df):
    
    

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

    #  Kör analysen
    
    # Vi sparar resultaten i en temporär kolumn
    sentiment_results = df['omdöme_text'].apply(get_sentiment)
    
    # 4. Dela upp resultaten i Label och Score
    df['sentiment_label'] = sentiment_results.apply(lambda x: x[0])
    df['sentiment_score'] = sentiment_results.apply(lambda x: x[1])
    
    # 5. Mapping till siffror (för att kunna räkna medelvärde)
    mapping = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
    df['feat_sentiment_index'] = df['sentiment_label'].map(mapping)
    
    return df

# skapar add_sentiment_behavior_features för att se mismatch mellan betyg och text
def add_sentiment_features(df):
    
    # Hittar kunder där texten och betyget inte stämmer överens.
    # Ex: Skriver "Jättegott" (1) men ger betyg 1.
    df['feat_sentiment_mismatch'] = 0
    
    # Logik: Negativ text (-1) men betyg 4-5 ELLER Positiv text (1) men betyg 1-2
    mismatch_mask = ((df['feat_sentiment_index'] == -1) & (df['omdömesbetyg'] >= 4)) | \
                    ((df['feat_sentiment_index'] == 1) & (df['omdömesbetyg'] <= 2))
    
    df.loc[mismatch_mask, 'feat_sentiment_mismatch'] = 1

    
    # Hittar kunder som är väldigt tydliga i sin feedback (Score > 0.95)
    df['feat_extreme_sentiment'] = 0
    df.loc[df['sentiment_score'] > 0.95, 'feat_extreme_sentiment'] = 1

    
    # Skapar en ren flagga för negativa omdömen för enkel aggregering
    df['feat_is_negative'] = (df['feat_sentiment_index'] == -1).astype(int)

    return df

# load till sql

def load_to_sqlite(df, db_name="matkassen_data.db", table_name="processed_data", method='append'):
    # 1. Skapa anslutning (filen skapas i din mapp)
    conn = sqlite3.connect(db_name)
    
    #  Spara DataFrame till S
    # if_exists='replace' gör att tabellen skrivs över varje gång jag testar
    df.to_sql(table_name, conn, if_exists=method, index=False)
    
    #  Läsr tillbaka antal rader 
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    
    print(f"--- LOAD KLART ---")
    print(f"Data sparad i filen: {db_name}")
    print(f"Verifiering: {count} rader har laddats upp korrekt.")
    
    conn.close()

#  anropar alla dina funktioner i den fulla pipeline-kedjan:

def transform_data(df):
    df_clean = df.copy()

    df_clean = clean_kassatyp_column(df_clean)
    df_clean = map_kassatyp(df_clean)
    df_clean = antal_portioner_nummer(df_clean)
    df_clean = clean_leveransvecka(df_clean)
    df_clean = clean_leveransdatum(df_clean)
    df_clean = clean_postnummer(df_clean)
    df_clean = clean_veckopris(df_clean)
    df_clean = clean_leveransstatus_bool(df_clean)
    df_clean = clean_pren_startdatum(df_clean)
    df_clean = clean_paus_fran_till(df_clean)
    df_clean = clean_pren_avslutsdatum(df_clean)
    df_clean = clean_kostpreferens(df_clean)
    df_clean = clean_omdome_text(df_clean)
    df_clean = clean_omdömesdatum(df_clean)
    df_clean = clean_omdömesbetyg(df_clean)
    df_clean = add_kunalder_dagar(df_clean)
    df_clean = feat_paus_langd(df_clean)
    df_clean = feat_churn_status_avslutat(df_clean)
    df_clean = feat_pris_per_portion(df_clean)
    df_clean = add_region_feature(df_clean)
    df_clean = add_stadnamn_feature(df_clean)
    df_clean = add_kblab_sentiment(df_clean)  
    df_clean = add_sentiment_features(df_clean)


    return df_clean

df_clean = transform_data(df)
df_clean_validation = transform_data(df_validation)
# Spara den färdiga träningsdatan till databasen
load_to_sqlite(df_clean, table_name="processed_training_data", method='replace')
load_to_sqlite(df_clean_validation, table_name="processed_training_data", method='append')
print("\n" + "="*50)
print("ETL-PIPELINE SLUTFÖRD. REDO FÖR ANALYS I JUPYTER")
print("="*50)






