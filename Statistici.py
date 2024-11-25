import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Statistici:
    def __init__(self, df, true_labels=False):
        self.df = df
        self.labels = true_labels
        self.race_mapping = {
            1: 'BEN',
            2: 'SBI',
            3: 'BRI',
            4: 'CHA',
            5: 'EUR',
            6: 'MCO',
            7: 'PER',
            8: 'RAG',
            9: 'SPH',
            10: 'ORI',
            11: 'TUV',
            12: 'Autre',
            13: 'NR',
            14: 'SAV'
        }
    
    def numara_instante_rase(self, fisier=None):
        result = {}
        for val in range(1, 15):
            count = int((self.df['Race'] == val).sum())
            if self.labels:
                result[self.race_mapping.get(val, val)] = count
            else:
                result[val] = count
        if fisier:
            with open(fisier, 'w') as f:
                f.write(str(result))
        else:
            return result
    
    def extrage_statistici(self, class_column):
        statistici = {}
        for col in self.df.columns:
            if col != class_column:
                total_valori_distincte = self.df[col].nunique()
                frecventa_valori_total = self.df[col].value_counts().to_dict()
                frecventa_pe_clase = self.df.groupby(class_column)[col].value_counts().unstack(fill_value=0).to_dict()
                if self.labels:
                    frecventa_pe_clase = {self.race_mapping.get(k, k): v for k, v in frecventa_pe_clase.items()}
                statistici[col] = {
                    'total_valori_distincte': total_valori_distincte,
                    'frecventa_valori_total': frecventa_valori_total,
                    'frecventa_pe_clase': frecventa_pe_clase
                }
        return statistici
    
    def afiseaza_statistici(self, statistici):
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

    def afiseaza_grafice(self):
        sns.set(style="whitegrid")
        
        for col in self.df.columns:
                plt.figure(figsize=(12, 5))
                
                # Histogramă pentru atribut
                plt.subplot(1, 2, 1)
                sns.histplot(self.df[col], bins=10, kde=True)
                plt.title(f'Histogramă - {col}')
                plt.xlabel(col)
                plt.ylabel('Frecvență')

                # Boxplot pentru atribut
                plt.subplot(1, 2, 2)
                sns.boxplot(x=self.df[col])
                plt.title(f'Boxplot - {col}')
                plt.xlabel(col)

                # Afișăm graficele
                plt.tight_layout()
                plt.show()
    
    def afiseaza_corelatii(self):
        corr_matrix = self.df.corr()
        
        plt.figure(figsize=(20, 10))  
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        
        plt.show()