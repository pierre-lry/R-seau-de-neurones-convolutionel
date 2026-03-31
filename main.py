import numpy as np
from perceptron import *
#Pour jeudi : A. Dataset images (xtrain,xlabel,ytrain,ylabel)
#B. forward
class CNN:
    def __init__(self,image,label,nb_couches,couches,taille_filtre,nb_couches_mlp,neurones_couche,learning_rate):
        self.image=image #tuple : (largeur,hauteur,canaux)
        self.label=label #label correspondant à l'image étudiée
        self.nb_couches=nb_couches #chiffre
        self.couches=couches #liste avec nb de filtres dans la couche
        self.taille_filtre=taille_filtre
        self.filtres={}
        for couche in range(0,nb_couches):
            for filtre in range(0,self.couches[couche]):
                self.filtres[couche][filtre]=np.random.randn(self.taille_filtre[couche][0],self.taille_filtre[couche][1])*np.sqrt(2.0 /(self.taille_filtre[couche][0]*self.taille_filtre[couche][1]))
        self.biais={}#dico avec en clé s'il s'agit d'un biais pour les filtres ou d'un biais pour la couche de densification
        for couche in range(0,nb_couches):
            self.biais[couche]=np.full((self.taille_filtre[couche][0],self.filtres[couche][1]),1)
        self.reseau=Reseau2neurone_RELU(nb_couches_mlp,neurones_couche,learning_rate)
    def reLU(self,x):
        return np.where(x>0,x,0)
    def dim_image(self, image): #fonction pour donner la dim de l'image sous la forme (H,W,D)
        shape = image.shape
        if len(shape) == 2 :
            H,W = shape
            D = 1
        else :
            H,W,D = shape
        return H,W,D
    def dim_filtre(self,filtre):
        H,W=filtre.shape
        return H,W
    def nb_iteration(self, dim_image, dim_filtre, pas): #le pas peut être différent entre décalage colonne
        #et décalage ligne
        cpt_H = 0
        cpt_W = 0
        while dim_image[0] - pas[0]*cpt_H >= dim_filtre[0] :
            cpt_H+=1
        while dim_image[1] - pas[1]*cpt_W >= dim_filtre[1] :
            cpt_W+=1
        return cpt_H,cpt_W

    def padding(self,image,valeur_padding,h_padding,w_padding):
        '''
        :param image --> matrice considérée,
        valeur_padding --> la valeur des pixels qu'on ajoute,
        h_padding --> hauteur du padding
        w_padding --> largeur du padding
        :
        :return:
        '''
        padded_image = np.pad(image, ((h_padding,h_padding), (w_padding,w_padding), (0,0)), mode='constant', constant_values=valeur_padding)
        return padded_image
    def convolution(self,image,filtres,pas, biais): #renvoie un np array de matrices de dimension qui dépend du pas et du padding. Les nb de matrices correspond aux nb de filtres (canaux)
        dimension_image=self.dim_image(image)
        taille_filtres=self.dim_filtre(filtres)
        iteration_H, iteration_W = self.nb_iteration(dimension_image, taille_filtres, pas)
        resultat = np.zeros((len(filtres), iteration_H, iteration_W))
        for num_filtre in range(len(filtres)):
            resultat_intermediaire = np.full((iteration_H, iteration_W), 0)
            for num_canal in range(dimension_image[2]): #on applique le filtre à chaque couche
                resultat_intermediaire2 = np.full((iteration_H, iteration_W), 0)
                for decalage_ligne in range(iteration_H):
                    for decalage_colonne in range(iteration_W):
                        kH, kW = filtres[num_filtre][num_canal].shape
                        zone = image[decalage_ligne * pas: decalage_ligne * pas + kH,decalage_colonne * pas: decalage_colonne * pas + kW,num_canal]
                        resultat_intermediaire2[decalage_ligne][decalage_colonne] = np.sum(zone*filtres[num_filtre][num_canal])
                resultat_intermediaire += resultat_intermediaire2
            resultat_intermediaire += biais[num_filtre]
            resultat[num_filtre] = resultat_intermediaire
        return resultat
    def maxPooling(self,liste_image,dimension):
        H,W=dimension
        resultat=[]
        for i in range(liste_image.shape[0]):
            iteration_H, iteration_W = self.nb_iteration(self.dim_image(liste_image[i]), (H, W), (H, W))
            pooling = np.zeros((iteration_H, iteration_W))
            for j in range(iteration_W):
                for k in range(iteration_H):
                    debut_H=H*k
                    debut_W=W*j
                    pooling[k,j]=np.max(liste_image[i][debut_H:debut_H+H,debut_W:debut_W+W])
            resultat.append(pooling)
        return resultat

    def averagePooling(self,liste_image,dimension):
        H, W = dimension
        resultat = []
        for i in range(liste_image.shape[0]):
            iteration_H, iteration_W = self.nb_iteration(self.dim_image(liste_image[i]), (H, W), (H, W))
            pooling = np.zeros((iteration_H, iteration_W))
            for j in range(iteration_W):
                for k in range(iteration_H):
                    debut_H = H * k
                    debut_W = W * j
                    pooling[k, j] = np.mean(liste_image[i][debut_H:debut_H + H, debut_W:debut_W + W])
            resultat.append(pooling)
        return resultat

    def dense(self,liste_image):
        return liste_image.flatten()

    def forward(self,nb_convolutions,pas,type='max'): #assemble les étapes
        image=self.image
        biais=self.biais
        for i in range(nb_convolutions):
            filtres=self.filtres[i]
            liste_image=self.convolution(image,filtres,pas[i],biais[i])
            liste_image=self.reLU(liste_image)
            if type=='max':
                liste_image=self.maxPooling(liste_image,pas)
            else:
                liste_image=self.averagePooling(liste_image,pas)
            image=liste_image
        image=self.dense(image)
        self.reseau.forward_propagation(image)

    def backward(self):
        self.reseau.backward_propagation()
