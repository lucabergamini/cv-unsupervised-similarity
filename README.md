# cv-unsupervised-similarity
Similarity between images using CV techniques 

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
- [x] caricamento immagini
- [x] memorizzazione vettori features

## RESULTS:

* BOW con SIFT fa schifo
* BOW con SIFT e hist HSV calcolato su 5 parti dell'immagine (Graziano) con distanza EMD fa ancora piu schifo
* BOW con SIFT e hist BGR calcolato su 5 parti dell'immagine (Graziano) con distanza EMD non va proprio(sempre stessa immagine)
* il mio nome vicino ad "ancora più schifo" mi fa sentire bene <3
* VGG16 usando layer 4096 fc2 funziona su alcuni animali e su qualche cibo. 
* RESNSET funziona peggio di VGG16 (1 anno di ricerca sulle skip connection per fare merda)
* VGG16 e le altre usando layer classificazione deve essere rivisto, perchè il metodo di confronto è importante, adesso usa le 5 piu probabili e da priorita a chi ha quelle per la distanza (suona bene ma la mia implementazione fa schifo)


## EMD

ha il vantaggio di poter specificare una matrice di distanza, che però va costruita in numpy e deve essere simmetrica (ovviamente).

## HOG

non sono ancora riuscito a farmi dare un feature vector lungo sempre uguale, comunque non sono invarianti ad un cazzo (scale rotation ma penso nemmeno alla traslazione)

## SIFT

molto buono perchè invariante, possiamo usarle cosi come sono o come BOW(gia wrappato)

## SIAMESI

le fa Stefano perchè le sa fare e a lui le reti funzionano

# SCORE SUL TEST:

## hist_sift
+----------------+----------------+---------------+----------------+----------------+---------------+  
|      food      |    animals     |   landscapes  |     tools      |     people     |      mean     |  
+----------------+----------------+---------------+----------------+----------------+---------------+  
| 0.389304152637 | 0.578878741497 | 0.46228531746 | 0.355361503576 | 0.392057823129 | 0.43557750766 |  
+----------------+----------------+---------------+----------------+----------------+---------------+  

## hist_color
+----------------+----------------+---------------+----------------+----------------+----------------+  
|      food      |    animals     |   landscapes  |     tools      |     people     |      mean      |  
+----------------+----------------+---------------+----------------+----------------+----------------+  
| 0.496529123434 | 0.551726927438 | 0.45982244898 | 0.390832025118 | 0.529864058957 | 0.485754916785 |  
+----------------+----------------+---------------+----------------+----------------+----------------+  

## resnet50
+---------------+----------------+----------------+----------------+----------------+----------------+  
|      food     |    animals     |   landscapes   |     tools      |     people     |      mean      |  
+---------------+----------------+----------------+----------------+----------------+----------------+  
| 0.68323896562 | 0.827984353741 | 0.732389795918 | 0.630774961999 | 0.569891439909 | 0.688855903438 |  
+---------------+----------------+----------------+----------------+----------------+----------------+  

## vgg16
+----------------+---------------+----------------+----------------+----------------+----------------+  
|      food      |    animals    |   landscapes   |     tools      |     people     |      mean      |  
+----------------+---------------+----------------+----------------+----------------+----------------+  
| 0.647984378937 | 0.73294829932 | 0.689745975057 | 0.630802746007 | 0.510466156463 | 0.642389511157 |  
+----------------+---------------+----------------+----------------+----------------+----------------+  

## vgg19
+----------------+---------------+----------------+----------------+----------------+----------------+  
|      food      |    animals    |   landscapes   |     tools      |     people     |      mean      |  
+----------------+---------------+----------------+----------------+----------------+----------------+  
| 0.647984378937 | 0.73294829932 | 0.689745975057 | 0.630802746007 | 0.510466156463 | 0.642389511157 |  
+----------------+---------------+----------------+----------------+----------------+----------------+  

## resnet50_cl
+----------------+----------------+----------------+----------------+---------------+----------------+  
|      food      |    animals     |   landscapes   |     tools      |     people    |      mean      |  
+----------------+----------------+----------------+----------------+---------------+----------------+  
| 0.548377997664 | 0.633779024943 | 0.626408333333 | 0.526619633201 | 0.41195515873 | 0.549428029574 |  
+----------------+----------------+----------------+----------------+---------------+----------------+  

## vgg16_cl
+----------------+----------------+----------------+----------------+----------------+----------------+  
|      food      |    animals     |   landscapes   |     tools      |     people     |      mean      |  
+----------------+----------------+----------------+----------------+----------------+----------------+  
| 0.623865812318 | 0.660673866213 | 0.637927210884 | 0.516051369266 | 0.456924206349 | 0.579088493006 |  
+----------------+----------------+----------------+----------------+----------------+----------------+  

## vgg19_cl
+----------------+----------------+----------------+----------------+----------------+----------------+  
|      food      |    animals     |   landscapes   |     tools      |     people     |      mean      |  
+----------------+----------------+----------------+----------------+----------------+----------------+  
| 0.623865812318 | 0.660673866213 | 0.637927210884 | 0.516051369266 | 0.456924206349 | 0.579088493006 |  
+----------------+----------------+----------------+----------------+----------------+----------------+  


