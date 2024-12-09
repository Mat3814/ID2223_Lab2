import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Exemple de données : remplacez-les par vos propres données
loss = [1.639,1.623,1.614,1.564,1.52,1.506,1.496,1.491,1.447,1.437,1.427,1.418,1.41,1.403,1.397,1.393,1.389,1.346,1.343,1.339,1.335,1.331,1.327,1.332,1.2,1,0.85,0.8,0.7579]
epochs = [i+1 for i in range(len(loss)-5)]+[28,30,35,37,40]  # Numéros d'epochs


# Création des points d'interpolation
x_smooth = np.linspace(min(epochs), max(epochs), 300)  # Plus de points pour lisser
spl = make_interp_spline(epochs, loss, k=3)  # k=3 pour une courbe lisse (cubic spline)
y_smooth = spl(x_smooth)

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(x_smooth, y_smooth, color='b', label='LOSS')

# Personnalisation du graphique
plt.title('LOSS', fontsize=16)
plt.xlabel('100k Rows of Training', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.xticks(epochs)
plt.yticks(fontsize=12)
plt.tight_layout()

# Affichage
plt.show()

