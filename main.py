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
    def padding(self,image,dimension_image,valeur_padding,h_padding,w_padding):
        '''
        :param image --> matrice considérée,
        dimension_image --> (H,W,D)
        valeur_padding --> la valeur des pixels qu'on ajoute,
        h_padding --> hauteur du padding
        w_padding --> largeur du padding
        :
        :return:
        '''
        for i in range(dimension_image[2]):
            nouvelle_image=np.full((h_padding+dimension_image[0],w_padding+dimension_image[1],3),valeur_padding)
            nouvelle_image[][]
        pass
    def convolution(self,image,filtres,nb_filtres,taille_filtres,pas): '''
    renvoie une liste de matrices de dimension qui dépend du pas et du padding. 
    Les nb de matrices correspond aux nb de filtres (canaux)'''
        return image
    def maxPooling(self,image):
        pass
    def averagePooling(self):
        pass
    def dense(self):
        pass
    def forward(self): #assemble les étapes
        pass
    def backward(self):
        pass

