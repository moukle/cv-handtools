# Freie Aufgabe

Frank Neumann (), Moritz Klein ()

## Idee

1. Wir haben uns als Ziel genommen, die Werkzeuge mittels eines Neuronalen Netzes zu klassifizieren.
Dazu möchten wir auf ein bereits trainiertes Netz zurückgreifen. Ein Mögliches ist das [vgg16](https://neurohive.io/en/popular-networks/vgg16/).
Wie genau wir es anpassen müssen, sodass es unsere Werkzeuge klassifiziert ist derzeit noch unsicher.

2. Eine mögliche Erweiterung ist es, dass wir ein Bild mit mehreren Werkzeugen erhalten und ein bestimmtes suchen / markieren.
Dazu wird das Bild mit einer Maske abgetastet. Der jeweilige Bereich wird dann an das zuvor beschriebene NN gegeben und klassifiziert.
Wir merken uns die Wahrscheinlichkeit der jeweiligen Bereiche und markieren am Ende die Maske mit der höchsten Wahrscheinlichkeit.

3. Auf die gleiche Weise wäre es möglich, alle Werkzeuge im Bild zu klassifizieren


## Ausarbeitung
### Datenbeschaffung
- hammer, plane, wrench: [scraper][2] auf imagenet

``` sh
$ id="n03481172" # sysnet_id for hammer
$ imagenetscraper $id dataset/unsorted/hammer --size 128,128 # download datset
```

- background: [uiuc][3]
``` sh
# set same size for UIUC images
mkdir $folder_unsorted/background
for file in $folder_path/uiuc_texture/*.jpg; do
    fileName=$(basename "$file")
    convert $file -resize $sizepx\x$sizepx! $folder_unsorted/background/$fileName
done
```

- in train/val/test datansatz teilen
``` sh
$ split_folders dataset/unsorted/ --output dataset/split --ratio .8 .1 .1 
```

- Datengrößen:
``` sh
# TRAIN SIZE [.8]
$ ls -l dataset/split/train/* | wc -l
3667

# VAL SIZE [.1]
$ ls -l dataset/split/val/* | wc -l
468

# TEST SIZE [.1]
$ ls -l dataset/split/test/* | wc -l
470
```

### Netzwerk - VGG16
- vortrainiertes netzwerk
- empfehlung kommilitone, sehr gut abgeschnitten [[1]]
- baselayer freezen, final layer definieren und trainieren

``` 
Epoch 1/8
3656/3656 [==============================] - 11s 3ms/step - loss: 4.0973 - acc: 0.7281 - val_loss: 3.4089 - val_acc: 0.7877
...
Epoch 8/8
3656/3656 [==============================] - 8s 2ms/step - loss: 3.4101 - acc: 0.7875 - val_loss: 2.5783 - val_acc: 0.8381

testset acc: 0.8431372549019608
trainingstime on gpu ~ 70sec
```


### Werkzeuge finden
1. ansatz: ganzes bild gerastert: sehr oft ausgeschlagen, wenn auch nur teil des werkzeugs zu sehen (mit hoher confidence auch bei sigmoid - kein rausfiltern moeglich)
[imgs/Figure_1.png]

2. ansatz: edges im bild detekten, counturen erkennen, boundingboxes um contouren legen und den bereich ans netzwerk fuettern zum klassifizieren
[imgs/labeledImage.jpg]

# Referenzen
[1]: https://neurohive.io/en/popular-networks/vgg16/ "VGG16 – Convolutional Network for Classification and Detection"
[2]: https://github.com/spinda/imagenetscraper "imagenetscraper: Bulk-download thumbnails from ImageNet synsets"
[3]: http://slazebni.cs.illinois.edu/research/uiuc_texture_dataset.zip "UIUC texture dataset"