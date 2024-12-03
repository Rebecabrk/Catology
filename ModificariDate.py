import os
import pandas as pd

pd.set_option('future.no_silent_downcasting', True) # ca sa nu mai avem warning-uri

class ModificariDate:
    def __init__(self, smote=False):
        if not os.path.exists('cat_data_preprocesat.xlsx'):
            self.df = self.__citire_set_de_date__('cat_data.xlsx', sheet_name='Data')
            self.df = self.__codificare__(self.df)
            self.df = self.__inlocuieste_zero_cu_media__(self.df)
            self.__scriere_set_de_date__(self.df, 'cat_data_preprocesat.xlsx')
        else:
            self.df = self.__citire_set_de_date__('cat_data_preprocesat.xlsx', sheet_name='Sheet1')
       
        if smote:
            if not os.path.exists('cat_data_preprocesat_plus_smote.xlsx'):
                self.df_smote = self.__aplica_smote__(self.df)
                self.__scriere_set_de_date__(self.df_smote, 'cat_data_preprocesat_plus_smote.xlsx')
            else:
                self.df_smote = self.__citire_set_de_date__('cat_data_preprocesat_plus_smote.xlsx')               
    
    def __citire_set_de_date__(self, fisier, sheet_name=None):
        dict = pd.read_excel(fisier, sheet_name=sheet_name)
        df = pd.DataFrame(dict)
        if fisier == 'cat_data.xlsx':
            df = df.drop(columns=['Horodateur', 'Row.names', 'Plus'])
        df = df.drop_duplicates()
        return df
    
    def __codificare__(self, df):
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
        return df
    
    def __inlocuieste_zero_cu_media__(self, df):
        for col in df.columns:
            media_rotunjita = round(df[df[col] != 0][col].mean())
            df[col] = df[col].replace(0, media_rotunjita)
        return df
    
    def __scriere_set_de_date__(self, df, fisier):
        df.to_excel(fisier, index=False)
    
    def __aplica_smote__(self, df):
        pass