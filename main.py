import pandas as pd
import pprint
from OneHotEncoderPersonalizat import OneHotEncoderPersonalizat
from Statistici import Statistici

date_codificate = OneHotEncoderPersonalizat().df
statistici = Statistici(date_codificate, true_labels=True)
pprint.pprint(statistici.numara_instante_rase())
statistici.afiseaza_statistici(statistici.extrage_statistici('Race'))
statistici.afiseaza_grafice()