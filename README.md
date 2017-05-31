# cv-unsupervised-similarity
Similarity between images using CV techniques 

## TODO:

* Write Python 3-Compatible Code
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
- [ ] caricamento immagini
- [ ] memorizzazione vettori features
- [ ] normalizzazione immagini?

## RESULTS:

* BOW con SIFT fa schifo
* BOW con SIFT e hist BGR o HSV calcolato su 5 parti dell'immagine (Graziano) con distanza EMD fa ancora piu schifo
## EMD

ha il vantaggio di poter specificare una matrice di distanza, che però va costruita in numpy e deve essere simmetrica (ovviamente).

## HOG

non sono ancora riuscito a farmi dare un feature vector lungo sempre uguale, comunque non sono invarianti ad un cazzo (scale rotation ma penso nemmeno alla traslazione)

## SIFT

molto buono perchè invariante, possiamo usarle cosi come sono o come BOW(gia wrappato)

## SIAMESI

le fa Stefano perchè le sa fare e a lui le reti funzionano
