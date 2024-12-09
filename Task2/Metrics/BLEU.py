import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Exemple de données : remplacez-les par vos propres données
bleu_scores = [5.326,5.487,6,6.179,6.25,6.32,6.36,6.6,6.68,6.74,6.81,6.86,6.9,6.94,6.97,7,7.14,7.19,7.25,7.28,7.30,7.32,7.34,7.35,8.5,8.8,9,9.05,9.16]  # 
epochs = [i+1 for i in range(len(bleu_scores)-5)]+[28,30,35,37,40]  # Numéros d'epochs


# Création des points d'interpolation
x_smooth = np.linspace(min(epochs), max(epochs), 300)  # Plus de points pour lisser
spl = make_interp_spline(epochs, bleu_scores, k=3)  # k=3 pour une courbe lisse (cubic spline)
y_smooth = spl(x_smooth)

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(x_smooth, y_smooth, color='b', label='BLEU Score')

# Personnalisation du graphique
plt.title('BLEU Score', fontsize=16)
plt.xlabel('100k Rows of Training', fontsize=14)
plt.ylabel('BLEU Score', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.xticks(epochs)
plt.yticks(fontsize=12)
plt.tight_layout()

# Affichage
plt.show()

