# cv-unsupervised-similarity
Similarity between images using CV techniques 

## TODO:

* Write Python 3-Compatible Code (Fottiti Stefano)
* HIST con EMD
* HOG
* SIFT
* reti siamesi?

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
