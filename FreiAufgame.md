# Freie Aufgabe
*Frank Neumann (), Moritz Klein ()*

**Table of Contents**
- [Freie Aufgabe](#Freie-Aufgabe)
  - [Idee](#Idee)
  - [Ausarbeitung](#Ausarbeitung)
    - [Datenbeschaffung](#Datenbeschaffung)
      - [Datengrößen](#Datengr%C3%B6%C3%9Fen)
    - [Netzwerk - VGG16](#Netzwerk---VGG16)
    - [Werkzeuge finden](#Werkzeuge-finden)
    - [Ergebnis](#Ergebnis)
  - [Referenzen](#Referenzen)


---

## Idee
1. Wir haben uns als Ziel genommen, die Werkzeuge mittels eines Neuronalen Netzes zu klassifizieren.
Dazu möchten wir auf ein bereits trainiertes Netz zurückgreifen. Ein Mögliches ist das [vgg16](https://neurohive.io/en/popular-networks/vgg16/).
Wie genau wir es anpassen müssen, sodass es unsere Werkzeuge klassifiziert, ist derzeit noch unsicher.

2. Eine mögliche Erweiterung ist es, dass wir ein Bild mit mehreren Werkzeugen erhalten und ein bestimmtes suchen / markieren.
Dazu wird das Bild mit einer Maske abgetastet. Der jeweilige Bereich wird dann an das zuvor beschriebene NN gegeben und klassifiziert.
Wir merken uns die Wahrscheinlichkeit der jeweiligen Bereiche und markieren am Ende die Maske mit der höchsten Wahrscheinlichkeit.

3. Auf die gleiche Weise wäre es möglich, alle Werkzeuge im Bild zu klassifizieren

---

## Ausarbeitung

[![Code](https://img.icons8.com/color/48/000000/gitlab.png)](https://code.fbi.h-da.de/istmoklei/cv-ss19/tree/master/ToolClassification)

### Datenbeschaffung
- Hammer, Plane, Wrench: [Scraper][2] auf ImageNet Datenbank
``` sh
$ id="n03481172" # sysnet_id for hammer
$ imagenetscraper $id dataset/unsorted/hammer --size 128,128 # download datset
```

- Background: [UIUC Texturen][3]
``` sh
# set same size for UIUC images
mkdir $folder_unsorted/background
for file in $folder_path/uiuc_texture/*.jpg; do
    fileName=$(basename "$file")
    convert $file -resize $sizepx\x$sizepx! $folder_unsorted/background/$fileName
done
```


#### Datengrößen
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
- vortrainiertes Netzwerk
- Empfehlung Kommilitone, sehr gut abgeschnitten [[1]]
- Baselayer freezen, Final Layer definieren und trainieren

``` 
Epoch 1/8

```


### Werkzeuge finden
1. Ansatz: ganzes Bild gerastert: sehr oft ausgeschlagen, wenn auch nur Teil des Werkzeugs zu sehen (mit hoher confidence auch bei Sigmoid - kein rausfiltern moeglich)

![gerastertes Bild](https://i.imgur.com/1hvV565.png)


2. Ansatz: Edges im Bild detektieren, Counturen erkennen, Boundingboxes um Contouren legen und den Bereich ans Netzwerk fuettern zum klassifizieren

![gelabeltes Bild](https://i.imgur.com/ZScg4p8.jpg)


### Ergebnis
[fertiger Ansatz][]


## Referenzen
[1]: https://neurohive.io/en/popular-networks/vgg16/ "VGG16 – Convolutional Network for Classification and Detection"
[2]: https://github.com/spinda/imagenetscraper "imagenetscraper: Bulk-download thumbnails from ImageNet synsets"
[3]: http://slazebni.cs.illinois.edu/research/uiuc_texture_dataset.zip "UIUC texture dataset"
[4]: https://github.com/OlafenwaMoses/ImageAI "ImageAI"