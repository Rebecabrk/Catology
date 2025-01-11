import pandas as pd
import pprint
from ModificariDate import ModificariDate
from Statistici import Statistici
from ReteaNeuronala import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np

modificari = ModificariDate()
date_codificate = modificari.df
statistici = Statistici(date_codificate, true_labels=True)
feature_names  = modificari.extract_feature_names(date_codificate)
# pprint.pprint(statistici.numara_instante_rase())
# statistici.afiseaza_statistici(statistici.extrage_statistici('Race'))
# statistici.afiseaza_grafice()
# statistici.afiseaza_corelatii()

p = Perceptron(date_codificate)
p.set_feature_names(feature_names)
W1, b1, W2, b2, mean_losses = p.antreneaza()
p.ploteaza_loss(mean_losses)

train_predictions = p.predict(p.train_x, W1, b1, W2, b2)
test_predictions = p.predict(p.test_x, W1, b1, W2, b2)

train_predictions = np.argmax(train_predictions, axis=1)
test_predictions = np.argmax(test_predictions, axis=1)

p.save_model('model.pkl')

p.vizualizeaza_puncte_eronate(p.test_x, p.test_y, test_predictions)

print(f'Training Accuracy: {accuracy_score(np.argmax(p.train_y, axis=1), train_predictions) * 100:.2f}%')
print(f'Test Accuracy: {accuracy_score(p.test_y, test_predictions) * 100:.2f}%')