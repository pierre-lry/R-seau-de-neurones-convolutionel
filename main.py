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
    def convolution(self,image,filtres,pas, biais): #renvoie une liste de matrices de dimension qui dépend du pas et du padding. Les nb de matrices correspond aux nb de filtres (canaux)
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
        for i in range(len(liste_image)):
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
        for i in range(len(liste_image)):
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
        liste_image=np.asarray(liste_image)
        applati=liste_image.flatten()

        pass
    def forward(self): #assemble les étapes
        pass
    def backward(self):
        pass