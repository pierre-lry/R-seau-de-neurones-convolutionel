import numpy as np
class CNN:
    def __init__(self,image,label,nb_couches,couches,filtres):
        self.image=image #tuple : (largeur,hauteur,canaux)
        self.label=label #label correspondant à l'image étudiée
        self.nb_couches=nb_couches #chiffre
        self.couches=couches
        self.filtres=filtres
    def reLU(self,x):
        return np.where(x>0,x,0)
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x))
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
    def convolution(self,image,filtres,nb_filtres,pas): #renvoie une liste de matrices de dimension qui dépend du pas et du padding. Les nb de matrices correspond aux nb de filtres (canaux)
        dimension_image=self.dim_image(image)
        taille_filtres=self.dim_filtre(filtres)
        resultat=np.array([])
        for i in range(nb_filtres):
            iteration_H, iteration_W = self.nb_iteration(dimension_image, taille_filtres, pas)
            resultat_intermediaire = np.full((iteration_H, iteration_W), 0)
            for j in range(dimension_image[2]): #on applique le filtre à chaque couche
                resultat_intermediaire2 = np.full((iteration_H, iteration_W), 0)
                for k in range(iteration_W):
                    for l in range(iteration_H):
                        #
            resultat=np.append(resultat,resultat_intermediaire)
        return resultat


        #return image
    def maxPooling(self,liste_image,dimension):
        H,W=dimension
        for i in range(len(liste_image)):
            iteration_H,iteration_W=self.nb_iteration(self.dim_image(liste_image[i]), (H,W),W)
            for i in range(iteration_W):
        pass
    def averagePooling(self):
        pass
    def dense(self):
        pass
    def forward(self): #assemble les étapes
        pass
    def backward(self):
        pass