import pandas as pd
import numpy as np
import sqlite3 

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
    s = df_cleaned['antal_portioner'].astype(str).str.lower()
    for word, num in ord_till_num.items():
        s = s.replace(word, num)
        df_cleaned['antal_portioner'] = s.str.extract('(\d+)').fillna(0).astype(float).astype(int)

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
    
    # 1. Gör om leveransdatum till riktiga datum
    df_clean["leveransdatum"] = pd.to_datetime(df_clean["leveransdatum"])
    
    # 2. Skapa en snygg leveransvecka direkt från datumet
    # .isocalendar().week ger veckonumret enligt internationell standard
    weeks = df_clean["leveransdatum"].dt.isocalendar().week.astype(str).str.zfill(2)
    years = df_clean["leveransdatum"].dt.year.astype(str)
    
    df_clean["leveransvecka"] = "vecka " + weeks + " - " + years
    
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
    # 1. Skapa kopia och byt namn från 'veckapris' till 'veckopris'
    df_clean = df.copy()
    df_clean = df_clean.rename(columns={'veckapris': 'veckopris'})
    
    # 2. Gör till sträng och rensa allt som inte är siffror eller komma
    # (Tar bort SEK, kr, :-, mellanslag etc.)
    s = df_clean["veckopris"].astype(str).str.replace(r'[^\d,]', '', regex=True)
    
    # 3. Byt komma mot punkt så Python förstår att det är ett tal
    s = s.str.replace(',', '.')
    
    # 4. Konvertera till siffror (float). Felaktiga värden blir 0.
    df_clean["veckopris"] = pd.to_numeric(s, errors='coerce').fillna(0)
    
    # 5. Gör om till heltal (int) så du slipper .0 på slutet
    df_clean["veckopris"] = df_clean["veckopris"].astype(int)
    
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



