import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('cat_data.xlsx', sheet_name='Data')
df = df.drop(columns=['Horodateur', 'Row.names', 'Plus'])
df = df.drop_duplicates()

df['Sexe'] = df['Sexe'].replace('F', 1)
df['Sexe'] = df['Sexe'].replace('M', 2)

df['Age'] = df['Age'].replace('Moinsde1', 1)
df['Age'] = df['Age'].replace('1a2', 2)
df['Age'] = df['Age'].replace('2a10', 3)
df['Age'] = df['Age'].replace('Plusde10', 4)

df['Race'] = df['Race'].replace('BEN', 1)
df['Race'] = df['Race'].replace('SBI', 2)
df['Race'] = df['Race'].replace('BRI', 3)
df['Race'] = df['Race'].replace('CHA', 4)
df['Race'] = df['Race'].replace('EUR', 5)
df['Race'] = df['Race'].replace('MCO', 6)
df['Race'] = df['Race'].replace('PER', 7)
df['Race'] = df['Race'].replace('RAG', 8)
df['Race'] = df['Race'].replace('SPH', 9)
df['Race'] = df['Race'].replace('ORI', 10)
df['Race'] = df['Race'].replace('TUV', 11)
df['Race'] = df['Race'].replace('Autre', 12)
df['Race'] = df['Race'].replace('NR', 13)
df['Race'] = df['Race'].replace('SAV', 14)

df['Nombre'] = df['Nombre'].replace('Plusde5', 6)

df['Logement'] = df['Logement'].replace('ASB', 1)
df['Logement'] = df['Logement'].replace('AAB', 2)
df['Logement'] = df['Logement'].replace('ML', 3)
df['Logement'] = df['Logement'].replace('MI', 4)

df['Zone'] = df['Zone'].replace('U', 1)
df['Zone'] = df['Zone'].replace('PU', 2)
df['Zone'] = df['Zone'].replace('R', 3)

df = df.replace('NSP', 0)

df = df.astype(int)

def inlocuieste_zero_cu_media(df):
    for col in df.columns:
        media_rotunjita = round(df[df[col] != 0][col].mean())
        df[col] = df[col].replace(0, media_rotunjita)
    return df

def numara_instante_rase(df):
    result = {}
    for val in range(1, 15):
        count = int((df['Race'] == val).sum())
        result[val] = count
    return result

def extrage_statistici(df, class_column):
    statistici = {}
    
    # Iterăm prin toate coloanele din dataframe
    for col in df.columns:
        if col != class_column:  # Ignorăm coloana cu clasele pentru prelucrarea generală
            # La nivelul întregului fișier
            valori_distincte = df[col].value_counts().to_dict()
            statistici[col] = {
                'total_valori_distincte': len(valori_distincte),
                'frecventa_valori_total': valori_distincte
            }
            
            # La nivelul fiecărei clase
            statistici[col]['frecventa_pe_clase'] = {}
            for clasa in df[class_column].unique():
                valori_pe_clasa = df[df[class_column] == clasa][col].value_counts().to_dict()
                statistici[col]['frecventa_pe_clase'][clasa] = valori_pe_clasa
    
    return statistici

def afiseaza_statistici(statistici):
    for col, data in statistici.items():
        print(f"\nAtribut: {col}")
        print(f"Numar total de valori distincte: {data['total_valori_distincte']}")
        print("Frecventa valorilor la nivel total:")
        for valoare, frecventa in data['frecventa_valori_total'].items():
            print(f"  Valoare {valoare}: {frecventa} aparitii")
        
        print("\nFrecventa valorilor pe clase:")
        for clasa, frecventa_clasa in data['frecventa_pe_clase'].items():
            print(f"  Clasa {clasa}:")
            for valoare, frecventa in frecventa_clasa.items():
                print(f"    Valoare {valoare}: {frecventa} aparitii")
        print("-" * 50)


df = inlocuieste_zero_cu_media(df)

distributie_rase = extrage_statistici(df, 'Race')
# afiseaza_statistici(distributie_rase)

def afiseaza_grafice(df):
    sns.set(style="whitegrid")
    
    for col in df.columns:
            plt.figure(figsize=(12, 5))
            
            # Histogramă pentru atribut
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], bins=10, kde=True)
            plt.title(f'Histogramă - {col}')
            plt.xlabel(col)
            plt.ylabel('Frecvență')
            
            # Boxplot pentru atribut
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot - {col}')
            plt.xlabel(col)
            
            # Afișăm graficele
            plt.tight_layout()
            plt.show()

afiseaza_grafice(df)