import numpy as np
from ReadingMnist import *
class Reseau2neurone_RELU :
    def __init__(self, nb_couche, neurones_couche, taux_apprentissage): #nb couche désigne le nb de couche
        #neurones_couche est une liste dont l'indice désigne la couche (0 étant la couche d'entrée)
        #et donc le nombre correspondant désigne le nombre de neurones dans la couche
        #le taux d'apprentissage est un paramètre qui permet de réguler la grandeur dans laquelle on va
        #modifier les poids et biais après chaque backward. Un taux d'apprentissage trop élevé rend difficile
        #la minimisation de l'erreur, et un taux d'apprentissage trop faible la rend trop longue. On peut
        #commencer par 0,05 ou 0,1, puis on le baissera si jamais c'est trop imprécis
        self.nb_couche=nb_couche
        self.neurones_couche=neurones_couche
        self.taux_apprentissage=taux_apprentissage
        self.sommes={} #il s'agit d'un dictionnaire dont les indices (1,...,nb_couche) désignent la couche
        #et dont l'élément correspondant est une liste des sommes entre biais et sorties des neurones
        #précédents, pour chaque neurone de la couche actuelle
        self.activation={} #pareil que précédemment, sauf qu'on initialise à 0 avec les valeurs d'entrées
        #et que c'est après la fonction d'activation (g(somme), où g est la fonction d'activation)
        self.gradients={}#cela désigne les gradients de chaque couche, c'est-à-dire un vecteur dans lequel
        #on a l'effet d'une modification marginale de chaque poids de la couche sur l'erreur totale
        self.erreurs={}#ici, on enregistre les erreurs pour chaque neurone par couche

        #ensuite, on met en place les biais et poids, en les initialisant aléatoirement, et on les place
        #dans un dictionnaire qu'on va remplir par des matrices pour chaque "couche" de poids et biais

        #on les met sous la forme suivante : en ligne, on a l'ensemble des poids qui relient de chaque
        #neurone de la couche d'avant vers un neurone de la couche d'après, + le biais. A partir
        #de ces matrices, nous pourrons ensuite calculer les fonctions d'activation et les mettre dans
        #le dico correspondant, en faisant la forward propagation

        self.reseau_poids={}#le dico des matrices de biais, poids
        for couche in range(1,self.nb_couche):
            nb_de_colonnes=neurones_couche[couche-1]+1 #il s'agit des neurones reliant à un neurone de la couche suivante
            #plus le biais
            nb_de_lignes=self.neurones_couche[couche]
            self.reseau_poids[couche]=np.random.randn(nb_de_lignes,nb_de_colonnes)*np.sqrt(2.0 / neurones_couche[couche-1]) #matrice d'éléments aléatoires
            #on utilise l'initialisation de Kaiming He, jugée plus adaptée aux fonctions d'activation de type ReLU
            self.reseau_poids[couche][:, -1] = 0
            #on initialise le biais à 0
    def forward_propagation(self,image):
        self.activation[0]=image #on suppose qu'on aura au préalable découpé l'image et mis dans une liste
        #les éléments correspondants à chaque pixel
        for couche in range(1, self.nb_couche):
            activation_avec_biais = np.append(self.activation[couche - 1], 1)
            self.sommes[couche] = np.dot(self.reseau_poids[couche], activation_avec_biais)  # on conserve dans
            # le dictionnaire les valeurs de chaque somme (seront utilisées dans la descente du gradient)
            if couche==self.nb_couche-1:
                exp_z = np.exp(self.sommes[couche])  # Stabilité numérique
                self.activation[couche] = exp_z / np.sum(exp_z)
            else:
                self.activation[couche] = np.where(self.sommes[couche] > 0, self.sommes[couche],0)  # pareil pour post-activation
        # après la forward, on a donc obtenu chaque output de chaque couche, dont celle de sortie

    #Pour la backward, on utilise comme fonction de coût Somme((y-y_attendu)^2), de dérivée partielle
    #2(yi-yi_attendu)
    def backward_propagation(self,label): #on va d'abord commencer à calculer les erreurs pour les neurones
        #de la couche de sortie. On utilise pour cela, la dérivée de la fonction d'activation
        #on utilise la formule erreur_sortie_i=2(yi-yi_attendu)*g'(z_sortie_i) où z est la somme avant activation
        vecteur_attendu=np.zeros(self.neurones_couche[-1]) #il s'agit d'un vecteur uniquement composé
        #de 0 sauf pour le label attendu où on a un 1
        vecteur_attendu[label]=1
        L = self.nb_couche
        self.erreurs[L-1]=self.activation[L-1] - vecteur_attendu#on a trouvé l'erreur de la couche de sortie, puis on va la projeter
        #vers les couches inférieures, avec la formule mathématique correspondante (on a utilisé softmax (fonction d'activation de la
        #dernière couche) combinée à fonction entropie croisée (calcul du coût))
        for couche in range(self.nb_couche-2,0,-1):
            poids_sans_biais = self.reseau_poids[couche + 1][:, :-1]
            erreur_propagee = np.dot(poids_sans_biais.T, self.erreurs[couche + 1]) #calcul matriciel afin d'
            #appliquer la formule de l'erreur propagée à tous les poids
            self.erreurs[couche] = erreur_propagee * np.where(self.sommes[couche] > 0, 1, 0)
        for couche in range(self.nb_couche-1, 0, -1):
            activation_avec_biais = np.append(self.activation[couche-1], 1)  # ajoute le biais
            gradient = np.outer(self.erreurs[couche],activation_avec_biais)  # calcule de gradiant pour tous les poids de la couche
            #en effet la dérivée partielle de l'erreur globale par rapport à un poids Wij=XiEi
            #et on a aussi, en effet, dL/dbj = ei
            self.reseau_poids[couche] = self.reseau_poids[couche] - self.taux_apprentissage * gradient


#### ENTRAINEMENT DU MODEL ####
def entrainement(nb_essais,nb_couche,neurones_couche,taux_apprentissage):
    meilleur_modele={}
        #{"Reseau":None,"sommes":None,"activation":None,"meilleure_precision":None,
                   #  "evolution_perte_moyenne":None,"evolution_precision":None}
    #on ne retiendra que ce qui est utilisé dans la forward (afin de prédire)
    meilleure_precision=0
    for i in range(nb_essais):
        reseau = Reseau2neurone_RELU(nb_couche, neurones_couche, taux_apprentissage)
        images = MnistDataloader()
        (x_train, y_train), (x_test, y_test) = images.load_data()
        nb_it = len(x_train)
        cpt = 0
        evolution_precision = np.array([])
        evolution_perte_moyenne = np.array([])
        rang = 0
        total_loss = 0
        for i in range(nb_it):
            picture = np.array(x_train[i]) / 255.0
            picture_a = picture.reshape(len(x_train[i]), -1)
            reseau.forward_propagation(picture_a)
            prediction = reseau.activation[reseau.nb_couche - 1]
            pred = np.argmax(prediction)
            perte = 0.5 * np.sum((prediction - [1 if j == y_train[i] else 0 for j in range(10)]) ** 2)
            total_loss += perte
            result = np.argmax([1 if j == y_train[i] else 0 for j in range(10)])
            if pred == result:
                cpt += 1
            reseau.backward_propagation(y_train[i])
            rang += 1
            evolution_precision = np.append(evolution_precision, cpt / rang)
            evolution_perte_moyenne = np.append(evolution_perte_moyenne, total_loss / rang)
        precision = cpt / nb_it
        if precision>meilleure_precision:
            meilleur_modele["meilleure_precision"]=precision
            meilleur_modele["Reseau"]=reseau.reseau_poids
            meilleur_modele["evolution_precision"]=evolution_precision
            meilleur_modele["evolution_perte_moyenne"]=evolution_perte_moyenne
            meilleur_modele["sommes"]=reseau.sommes
            meilleur_modele["activation"]=reseau.activation
            meilleur_modele["erreurs"]=reseau.erreurs

    return meilleur_modele

def entrainement_consecutif(nb_essais,nb_couche,neurones_couche,taux_apprentissage):
    meilleur_modele={}
        #{"Reseau":None,"sommes":None,"activation":None,"meilleure_precision":None,
                   #  "evolution_perte_moyenne":None,"evolution_precision":None}
    #on ne retiendra que ce qui est utilisé dans la forward (afin de prédire)
    meilleure_precision=0
    reseau = Reseau2neurone_RELU(nb_couche, neurones_couche, taux_apprentissage)
    images = MnistDataloader()
    (x_train, y_train), (x_test, y_test) = images.load_data()
    nb_it = len(x_train)
    cpt = 0
    evolution_precision = np.array([])
    evolution_perte_moyenne = np.array([])
    rang = 0
    total_loss = 0
    for i in range(nb_it):
        picture = np.array(x_train[i]) / 255.0
        picture_a = picture.reshape(len(x_train[i]), -1)
        reseau.forward_propagation(picture_a)
        prediction = reseau.activation[reseau.nb_couche - 1]
        pred = np.argmax(prediction)
        perte = 0.5 * np.sum((prediction - [1 if j == y_train[i] else 0 for j in range(10)]) ** 2)
        total_loss += perte
        result = np.argmax([1 if j == y_train[i] else 0 for j in range(10)])
        if pred == result:
            cpt += 1
        reseau.backward_propagation(y_train[i])
        rang += 1
        evolution_precision = np.append(evolution_precision, cpt / rang)
        evolution_perte_moyenne = np.append(evolution_perte_moyenne, total_loss / rang)
    precision = cpt / nb_it
    meilleur_modele["meilleure_precision"] = precision
    meilleur_modele["Reseau"] = reseau.reseau_poids
    meilleur_modele["evolution_precision"] = evolution_precision
    meilleur_modele["evolution_perte_moyenne"] = evolution_perte_moyenne
    meilleur_modele["sommes"] = reseau.sommes
    meilleur_modele["activation"] = reseau.activation
    meilleur_modele["erreurs"] = reseau.erreurs
    # Visualisation
    plt.figure(figsize=(15, 6))
    # x = les étapes (1, 2, 3, ...)
    x = range(1, len(evolution_precision) + 1)

    # Scatter plot + ligne pour voir l'évolution
    plt.plot(x, evolution_precision, linewidth=1, alpha=0.7, color='blue', label='Précision')
    plt.plot(x, evolution_perte_moyenne, linewidth=2, alpha=0.7, color='black', label='Perte moyenne')
    plt.scatter(x, evolution_precision, s=2, alpha=0.3, color='red')

    plt.title(f"Résultat du 1er entrainement sur ({len(evolution_precision)} images)")
    plt.xlabel("Image traitée")
    plt.ylabel("Précision cumulée")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Ajouter la précision finale comme ligne horizontale
    plt.axhline(y=precision, color='green', linestyle='--', alpha=0.5,
                label=f'Précision finale: {precision:.4f}')
    plt.legend()
    print("1ère simulation effectuée")
    for j in range(2,nb_essais+1):
        images = MnistDataloader()
        (x_train, y_train), (x_test, y_test) = images.load_data()
        nb_it = len(x_train)
        cpt = 0
        evolution_precision = np.array([])
        evolution_perte_moyenne = np.array([])
        rang = 0
        total_loss = 0
        for i in range(nb_it):
            picture = np.array(x_train[i]) / 255.0
            picture_a = picture.reshape(len(x_train[i]), -1)
            reseau.forward_propagation(picture_a)
            prediction = reseau.activation[reseau.nb_couche - 1]
            pred = np.argmax(prediction)
            perte = 0.5 * np.sum((prediction - [1 if j == y_train[i] else 0 for j in range(10)]) ** 2)
            total_loss += perte
            result = np.argmax([1 if j == y_train[i] else 0 for j in range(10)])
            if pred == result:
                cpt += 1
            reseau.backward_propagation(y_train[i])
            rang += 1
            evolution_precision = np.append(evolution_precision, cpt / rang)
            evolution_perte_moyenne = np.append(evolution_perte_moyenne, total_loss / rang)
        precision = cpt / nb_it
        meilleur_modele["meilleure_precision"]=precision
        meilleur_modele["Reseau"]=reseau.reseau_poids
        meilleur_modele["evolution_precision"]=evolution_precision
        meilleur_modele["evolution_perte_moyenne"]=evolution_perte_moyenne
        meilleur_modele["sommes"]=reseau.sommes
        meilleur_modele["activation"]=reseau.activation
        meilleur_modele["erreurs"]=reseau.erreurs
        # Visualisation
        plt.figure(figsize=(15, 6))
        # x = les étapes (1, 2, 3, ...)
        x = range(1, len(evolution_precision) + 1)

        # Scatter plot + ligne pour voir l'évolution
        plt.plot(x, evolution_precision, linewidth=1, alpha=0.7, color='blue', label='Précision')
        plt.plot(x, evolution_perte_moyenne, linewidth=2, alpha=0.7, color='black', label='Perte moyenne')
        plt.scatter(x, evolution_precision, s=2, alpha=0.3, color='red')

        plt.title(f"Résultat du {j}ème entrainement sur ({len(evolution_precision)} images)")
        plt.xlabel("Image traitée")
        plt.ylabel("Précision cumulée")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Ajouter la précision finale comme ligne horizontale
        plt.axhline(y=precision, color='green', linestyle='--', alpha=0.5,
                    label=f'Précision finale: {precision:.4f}')
        plt.legend()
        print(f"{j}eme simulation effectuée")
        reseau.taux_apprentissage = reseau.taux_apprentissage / 10  # on diminue le taux d'apprentissage apr_s chaque itération
    return meilleur_modele




nb_couche = 3
neurones_couche = [784, 128, 10]
taux_apprentissage = 0.01
meilleur_modele=entrainement_consecutif(1,nb_couche, neurones_couche, taux_apprentissage)

#TEST
images = MnistDataloader()
(x_train, y_train), (x_test, y_test) = images.load_data()
nb_it = len(x_test)
cpt = 0
evolution_precision = np.array([])
evolution_perte_moyenne = np.array([])
rang = 0
total_loss = 0
reseau = Reseau2neurone_RELU(nb_couche, neurones_couche, taux_apprentissage)
reseau.reseau_poids=meilleur_modele["Reseau"]
reseau.sommes=meilleur_modele["sommes"]
reseau.activation=meilleur_modele["activation"]
for i in range(nb_it):
    picture = np.array(x_test[i]) / 255.0
    picture_a = picture.reshape(len(x_test[i]), -1)
    reseau.forward_propagation(picture_a)
    prediction = reseau.activation[reseau.nb_couche - 1]
    pred = np.argmax(prediction)
    perte = 0.5 * np.sum((prediction - [1 if j == y_test[i] else 0 for j in range(10)]) ** 2)
    total_loss += perte
    result = np.argmax([1 if j == y_test[i] else 0 for j in range(10)])
    if pred == result:
        cpt += 1
    rang += 1
    evolution_precision = np.append(evolution_precision, cpt / rang)
    evolution_perte_moyenne = np.append(evolution_perte_moyenne, total_loss / rang)
precision = cpt / nb_it


#### VISUALISATION ####
plt.figure(figsize=(15, 6))

x = range(1, len(evolution_precision) + 1)

plt.plot(x, evolution_precision, linewidth=1, alpha=0.7, color='blue', label='Précision')
plt.plot(x, evolution_perte_moyenne, linewidth=2, alpha=0.7, color='black', label='Perte moyenne')
plt.scatter(x, evolution_precision, s=2, alpha=0.3, color='red')

plt.title(f"Résultat des tests sur ({len(evolution_precision)} images)")
plt.xlabel("Image traitée")
plt.ylabel("Précision cumulée")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

plt.axhline(y=precision, color='green', linestyle='--', alpha=0.5,
            label=f'Précision finale: {precision:.4f}')
plt.legend()

plt.tight_layout()

print(f'Précision finale du test : {precision}, Paramètres utilisés : learning rate de {taux_apprentissage}, nombre de neurones (par couche):{neurones_couche}')
plt.show()
