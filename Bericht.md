# Bericht
Frank Neumann (), Moritz Klein ()

---

## Praktikum 1
### 0. Tensorflow & OpenCV
``` sh
$ sudo pacman -S ...
# tensorflow, opencv, ipython, jupyter
```

Visualize: TensorBoard
Abstraction: Keras, TF-Slim

### 1.1. Code nachvollziehen

### 1.2. padding='SAME'
Sonst würden die Bilder zu klein: jede Seite -2 Pixel.

### 1.3. Aktivierungsfunktion
Die Aktivierungsfunktion gibt an, wann *gefeuert* wird.

### 1.4 ReLu -> Sigmoid
- ReLu [ max(0,v) == [0,v]]
```
step 0, training accuracy 0.10000000149011612
step 100, training accuracy 0.8799999952316284
step 200, training accuracy 0.8199999928474426
step 300, training accuracy 0.8999999761581421
step 400, training accuracy 0.9399999976158142
test accuracy: 0.9354000091552734
```

- Sigmoid [ 1 / (1 + e^(-av)) == [0,1]]
```
step 0, training accuracy 0.07999999821186066
step 100, training accuracy 0.30000001192092896
step 200, training accuracy 0.10000000149011612
step 300, training accuracy 0.07999999821186066
step 400, training accuracy 0.23999999463558197
test accuracy: 0.29429998993873596
```

Advantage:
Sigmoid: not blowing up activation
Relu: not vanishing gradient
Relu: More computationally efficient to compute than Sigmoid like functions since Relu just needs to pick max(0,x) and not perform expensive exponential operations as in Sigmoids
Relu: In practice, networks with Relu tend to show better convergence performance than sigmoid.

Disadvantage:
Sigmoid: tend to vanish gradient (cause there is a mechanism to reduce the gradient as "a" increase, where "a" is the input of a sigmoid function. Gradient of Sigmoid: S′(a)=S(a)(1−S(a)). When "a" grows to infinite large , S′(a)=S(a)(1−S(a))=1×(1−1)=0).
Relu: tend to blow up activation (there is no mechanism to constrain the output of the neuron, as "a" itself is the output)
Relu: Dying Relu problem - if too many activations get below zero then most of the units(neurons) in network with Relu will simply output zero, in other words, die and thereby prohibiting learning.(This can be handled, to some extent, by using Leaky-Relu instead.)

### 1.5. Pooling
Technically, pooling means reducing the size of the data with some local aggregation function, typically within each feature map.

The reasoning behind this is both technical and more theoretical. The technical aspect is that pooling reduces the size of the data to be processed downstream. This can drastically reduce the number of overall parameters in the model, especially if we use fully connected layers after the convolutional ones.

The more theoretical reason for applying pooling is that we would like our computed features not to care about small changes in position in an image. For instance, a feature looking for eyes in the top-right part of an image should not change too much if we move the camera a bit to the right when taking the picture, moving the eyes slightly to the center of the image. Aggregating the “eye-detector feature” spatially allows the model to overcome such spatial variability between images, capturing some form of invariance as discussed at the beginning of this chapter.

ksize = abtastgroesse
stride = schiebegroesse

### 1.6. Overfitting
Overfitting occurs when a rule (for instance, a classifier) is computed in a way that explains the training set, but with poor generalization to unseen data.

### 1.7. Dropout Test
```
```

### 1.8. Dropout
Dropout “turns off” a random preset fraction of the units in a layer, by setting their values to zero during training. These dropped-out neurons are random—different for each computation—forcing the network to learn a representation that will work even after the dropout. This process is often thought of as training an “ensemble” of multiple networks, thereby increasing generalization.

### 1.9. CIFAR10
[CIFAR10 Python Download](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

### 1.10. Daten visualisieren
``` python
def display_cifar(images, labels, size):
    n = len(images)
    dimension = ceil(sqrt(size))
    plt.figure()
    for i in range(size):
        plt.subplot(dimension, dimension, i+1)
        plt.axis('off')
        rand = np.random.choice(n)
        label = np.argmax(labels[rand])
        image = images[rand]
        plt.title(label)
        plt.imshow(image)
    plt.show()
```

### 1.11. False Positive
```
Actual Class | Predicted Class
1               0                   = False Negative
1               1                   = True Positive
0               0                   = True Negative
0               1                   = False Positive
```


### 1.12. False Positives Klassifizieren?
``` matlab
fp = sum(Y == 0 && pred)
```

### 1.13. Skizze RNN / CNN (Frameworks)

### 1.14. Kontrast
```
Alt:
Accuracy: 10.62%
Accuracy: 42.09%
Accuracy: 48.17%

Neu:
(3x3 grid)
Accuracy: 9.07%
Accuracy: 41.17%
Accuracy: 48.46%

(8x8 grid)
Accuracy: 10.53%
Accuracy: 39.89%
Accuracy: 45.72%
```

---

## Praktikum 2 - Einarbeitung in OpenCV
### Einleitung
- Aufgabe / Ziel
- CV Funktionen

### Konzept
#### Linien aus Bild extrahieren
1. Bild schwarzweiß
2. Schwarzweiß binarisieren
    - Wie? Adaptiv nur bedingt gute Ergebnisse
3. Noise entfernen
    - Opening/Closing, Blur
4. Edges mithilfe von Canny detektieren
5. Linien mit `HoughLines` (Probabilistic) bestimmt
    - `HoughLinesP` gibt die Extremwerte der Linie zurück (P1, P2)

> bin.jpg, edges.jpg, lines.jpg

#### Längste Linie finden
1. Über alle Linien iterieren
2. Euklidischen Abstand bestimmen
    $$Länge = \sqrt{(x1 -x2)^2 + (y1 - y2)^2}$$
3. Längste Linie merken und returnen

> longest.jpg

#### Pixel zu Milimeter
1. Bild mit Referenzobjekt erstellen (Objekt hebt sich farblich ab; Länge bekannt)
2. Aus Bild nur Objekt maskieren
3. Aus maskiertem Bild längste Linie ermitteln
4. Pixel zu Milimeter verhältnis bestimmen: `ref_mm_length / longest_line_px`
5. Längste Linie des Werkstücks mit Ratio multiplizieren

> ref.jpg

### Probleme
- Händische Binarisierung
- Werkstück kleiner als Referenzobjekt
- Nur längste Kante ermittelt (nicht Breite), aber siehe Konzept:

    0. Grundannahme: Werkstück besitzt einen langen gleichseitigen Griff
    1. Die **beiden** längsten Linien finden
    2. Abstand zw. beiden Linien gibt Breite an

- Objekt liegt ungünstig


### Referenzen
<!---
**The quick brown [fox][1], jumped over the lazy [dog][2].**

[1]: https://en.wikipedia.org/wiki/Fox "Wikipedia: Fox"
[2]: https://en.wikipedia.org/wiki/Dog "Wikipedia: Dog"
-->

---

## Praktikum 3
Werkzeuge klassifizieren (VGG16?)