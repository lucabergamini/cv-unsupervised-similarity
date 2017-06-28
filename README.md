# cv-unsupervised-similarity
Similarity between images using CV techniques 

# Peoples:
## Group 1:
Bergamini Luca-> Hist-Color with EMD,BOW, general framework
Ballotta Diego-> Performances Test, SIFT
## Group 2:
Della Casa Venturelli Gabriele-> Deep features, GUI
Pini Stefano-> Resnet with training and A LOT of debugging 


## TODO:

* Write Python 3-Compatible Code (Fottiti Stefano)
* HIST con EMD
* HOG
* SIFT
* reti siamesi?
* il testing va fatto rispetto alla cartella di train!

## TASKS:

- [x] hist
- [x] EMD
- [x] SIFT
- [x] BOW
- [x] HOG
- [ ] Siamesi
- [x] caricamento immagini
- [x] memorizzazione vettori features

## RESULTS:

* BOW con SIFT fa schifo
* BOW con SIFT e hist HSV calcolato su 5 parti dell'immagine (Graziano) con distanza EMD fa ancora piu schifo
* BOW con SIFT e hist BGR calcolato su 5 parti dell'immagine (Graziano) con distanza EMD non va proprio(sempre stessa immagine)
* il mio nome vicino ad "ancora più schifo" mi fa sentire bene <3
* VGG16 usando layer 4096 fc2 funziona su alcuni animali e su qualche cibo.
* VGG16 e le altre usando layer classificazione deve essere rivisto, perchè il metodo di confronto è importante, adesso usa le 5 piu probabili e da priorita a chi ha quelle per la distanza (controllate la mia implementazione!!)


## EMD

ha il vantaggio di poter specificare una matrice di distanza, che però va costruita in numpy e deve essere simmetrica (ovviamente). Ho usato una intera incrementale, forse una gaussiana sarebbe meglio.

## HOG

non sono ancora riuscito a farmi dare un feature vector lungo sempre uguale, comunque non sono invarianti ad un cazzo (scale rotation ma penso nemmeno alla traslazione).

## SIFT

molto buono perchè invariante, possiamo usarle cosi come sono o come BOW(gia wrappato)

## SIAMESI

le fa Stefano perchè le sa fare e a lui le reti funzionano

# SCORE SUL TEST:

hist_sift
+-----------------+----------------+----------------+----------------+----------------+----------------+  
|       food      |    animals     |   landscapes   |     tools      |     people     |      mean      |  
+-----------------+----------------+----------------+----------------+----------------+----------------+  
| 0.0878605308014 | 0.164083371295 | 0.126272359989 | 0.438322553161 | 0.250859078948 | 0.213479578839 |  
+-----------------+----------------+----------------+----------------+----------------+----------------+  

hist_color
+----------------+----------------+---------------+----------------+----------------+----------------+  
|      food      |    animals     |   landscapes  |     tools      |     people     |      mean      |  
+----------------+----------------+---------------+----------------+----------------+----------------+  
| 0.143403576659 | 0.277448268447 | 0.23695548543 | 0.377946901448 | 0.336654897138 | 0.274481825825 |  
+----------------+----------------+---------------+----------------+----------------+----------------+  

resnet50
+----------------+----------------+----------------+----------------+----------------+----------------+  
|      food      |    animals     |   landscapes   |     tools      |     people     |      mean      |  
+----------------+----------------+----------------+----------------+----------------+----------------+  
| 0.091203043676 | 0.188071925427 | 0.495663092757 | 0.745786574354 | 0.191117913789 | 0.342368510001 |  
+----------------+----------------+----------------+----------------+----------------+----------------+  

vgg16
+-----------------+----------------+----------------+----------------+----------------+----------------+  
|       food      |    animals     |   landscapes   |     tools      |     people     |      mean      |  
+-----------------+----------------+----------------+----------------+----------------+----------------+  
| 0.0600104905354 | 0.106363490736 | 0.392611619228 | 0.838423583397 | 0.141175937488 | 0.307717024277 |  
+-----------------+----------------+----------------+----------------+----------------+----------------+  

vgg19
+----------------+-----------------+----------------+----------------+---------------+----------------+  
|      food      |     animals     |   landscapes   |     tools      |     people    |      mean      |  
+----------------+-----------------+----------------+----------------+---------------+----------------+  
| 0.051083604418 | 0.0975266787908 | 0.238417574148 | 0.847449502092 | 0.13640612418 | 0.274176696726 |  
+----------------+-----------------+----------------+----------------+---------------+----------------+  

resnet50_cl
+-----------------+----------------+----------------+----------------+----------------+----------------+  
|       food      |    animals     |   landscapes   |     tools      |     people     |      mean      |  
+-----------------+----------------+----------------+----------------+----------------+----------------+  
| 0.0685309248777 | 0.134480418002 | 0.318105766591 | 0.561624410588 | 0.254405685134 | 0.267429441039 |  
+-----------------+----------------+----------------+----------------+----------------+----------------+  

vgg16_cl
+-----------------+----------------+---------------+----------------+----------------+----------------+  
|       food      |    animals     |   landscapes  |     tools      |     people     |      mean      |  
+-----------------+----------------+---------------+----------------+----------------+----------------+  
| 0.0670523716014 | 0.147221098782 | 0.29984618808 | 0.587401746843 | 0.244579498749 | 0.269220180811 |  
+-----------------+----------------+---------------+----------------+----------------+----------------+  

vgg19_cl
+-----------------+---------------+----------------+----------------+----------------+--------------+  
|       food      |    animals    |   landscapes   |     tools      |     people     |     mean     |  
+-----------------+---------------+----------------+----------------+----------------+--------------+  
| 0.0702165774838 | 0.14848225021 | 0.290013365565 | 0.575979632169 | 0.245296066075 | 0.2659975783 |  
+-----------------+---------------+----------------+----------------+----------------+--------------+  

inception_resnet_v2
+----------------+----------------+----------------+----------------+----------------+----------------+  
|      food      |    animals     |   landscapes   |     tools      |     people     |      mean      |  
+----------------+----------------+----------------+----------------+----------------+----------------+  
| 0.144206114565 | 0.221810640932 | 0.544063105228 | 0.639256231591 | 0.230850733601 | 0.356037365184 |  
+----------------+----------------+----------------+----------------+----------------+----------------+  

inception_resnet_v2_cl
+----------------+----------------+----------------+----------------+----------------+----------------+  
|      food      |    animals     |   landscapes   |     tools      |     people     |      mean      |  
+----------------+----------------+----------------+----------------+----------------+----------------+  
| 0.165390432657 | 0.299849711045 | 0.621758373424 | 0.593101833448 | 0.251146697422 | 0.386249409599 |  
+----------------+----------------+----------------+----------------+----------------+----------------+  




