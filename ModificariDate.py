import os
import pandas as pd
import json

pd.set_option('future.no_silent_downcasting', True) # ca sa nu mai avem warning-uri

class ModificariDate:
    def __init__(self, smote=False):
        if not os.path.exists('cat_data_preprocesat.xlsx'):
            self.df = self.__citire_set_de_date__('cat_data.xlsx', sheet_name='Data')
            self.df = self.__codificare__(self.df)
            self.df = self.__inlocuieste_minus1_cu_media__(self.df)
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
    
    @staticmethod
    def extract_feature_names(df, print_dict=False):
        translation_dict = {
            'Sexe': 'Sex',
            'Age': 'Age',
            'Race': 'Breed',
            'Nombre': 'Number',
            'Logement': 'Housing',
            'Zone': 'Zone',
            'Ext': 'Ext',
            'Obs': 'Obs',
            'Timide': 'Shy',
            'Calme': 'Calm',
            'Effrayé': 'Scared',
            'Intelligent': 'Intelligent',
            'Vigilant': 'Vigilant',
            'Perséverant': 'Persistent',
            'Affectueux': 'Affectionate',
            'Amical': 'Friendly',
            'Solitaire': 'Solitary',
            'Brutal': 'Brutal',
            'Dominant': 'Dominant',
            'Agressif': 'Aggressive',
            'Impulsif': 'Impulsive',
            'Prévisible': 'Predictable',
            'Distrait': 'Distracted',
            'Abondance': 'Abundance',
            'PredOiseau': 'PredBirds',
            'PredMamm': 'PredMammals',
        }

        if print_dict:
            print("Translation dictionary:")
            for key, value in translation_dict.items():
                print(f"{key} -> {value}")
            print()
        return [translation_dict[col] for col in df.columns]

    
    def __codificare__(self, df):
        df['Sexe'] = df['Sexe'].replace('F', 0)
        df['Sexe'] = df['Sexe'].replace('M', 1)

        df['Age'] = df['Age'].replace('Moinsde1', 0)
        df['Age'] = df['Age'].replace('1a2', 1)
        df['Age'] = df['Age'].replace('2a10', 2)
        df['Age'] = df['Age'].replace('Plusde10', 3)

        df['Race'] = df['Race'].replace('BEN', 0)
        df['Race'] = df['Race'].replace('SBI', 1)
        df['Race'] = df['Race'].replace('BRI', 2)
        df['Race'] = df['Race'].replace('CHA', 3)
        df['Race'] = df['Race'].replace('EUR', 4)
        df['Race'] = df['Race'].replace('MCO', 5)
        df['Race'] = df['Race'].replace('PER', 6)
        df['Race'] = df['Race'].replace('RAG', 7)
        df['Race'] = df['Race'].replace('SPH', 8)
        df['Race'] = df['Race'].replace('ORI', 9)
        df['Race'] = df['Race'].replace('TUV', 10)
        df['Race'] = df['Race'].replace('Autre', 11)
        df['Race'] = df['Race'].replace('NR', 12)
        df['Race'] = df['Race'].replace('SAV', 13)

        df['Nombre'] = df['Nombre'].replace('Plusde5', 6)

        df['Logement'] = df['Logement'].replace('ASB', 0)
        df['Logement'] = df['Logement'].replace('AAB', 1)
        df['Logement'] = df['Logement'].replace('ML', 2)
        df['Logement'] = df['Logement'].replace('MI', 3)

        df['Zone'] = df['Zone'].replace('U', 0)
        df['Zone'] = df['Zone'].replace('PU', 1)
        df['Zone'] = df['Zone'].replace('R', 2)

        df = df.replace('NSP', -1)
        df = df.astype(int)

        race_codification = {
            0: 'BENGAL',
            1: 'BIRMAN',
            2: 'BRITISH_SHORTHAIR',
            3: 'CHARTREUX',
            4: 'EUROPEAN',
            5: 'MAINE_cOON',
            6: 'PERSIAN',
            7: 'RAGDOLL',
            8: 'SPHYNX',
            9: 'SIAMESE', #ORI.....
            10: 'TURKISH_ANGORA',
            11: 'OTHER',
            12: 'NO_BREED',
            13: 'SAVANNAH'
        }
        with open('race_codification.json', 'w') as json_file:
            json.dump(race_codification, json_file, indent=4)
        return df
    
    def __inlocuieste_minus1_cu_media__(self, df):
        for col in df.columns:
            media_rotunjita = round(df[df[col] != -1][col].mean())
            df[col] = df[col].replace(-1, media_rotunjita)
        return df
    
    def __scriere_set_de_date__(self, df, fisier):
        df.to_excel(fisier, index=False)
    
    def __aplica_smote__(self, df):
        pass