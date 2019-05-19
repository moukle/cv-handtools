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