``` sh
# TRAIN SIZE [.8]
$ ls -l dataset/split/train/* | wc -l
2864

# VAL SIZE [.1]
$ ls -l dataset/split/val/* | wc -l
365

# TEST SIZE [.1]
$ ls -l dataset/split/test/* | wc -l
367
```

```
2856/2856 [==============================] - 9s 3ms/step - loss: 1.6427 - categorical_accuracy: 0.8032 - val_loss: 1.5195 - val_categorical_accuracy: 0.8235
Epoch 2/15
2856/2856 [==============================] - 7s 2ms/step - loss: 1.0440 - categorical_accuracy: 0.8852 - val_loss: 1.1754 - val_categorical_accuracy: 0.8768
Epoch 3/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.7824 - categorical_accuracy: 0.9086 - val_loss: 1.0851 - val_categorical_accuracy: 0.8655
Epoch 4/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.5117 - categorical_accuracy: 0.9394 - val_loss: 1.0368 - val_categorical_accuracy: 0.8711
Epoch 5/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.3671 - categorical_accuracy: 0.9566 - val_loss: 1.2026 - val_categorical_accuracy: 0.8487
Epoch 6/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.2731 - categorical_accuracy: 0.9706 - val_loss: 0.8639 - val_categorical_accuracy: 0.8880
Epoch 7/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.2335 - categorical_accuracy: 0.9720 - val_loss: 1.1999 - val_categorical_accuracy: 0.8571
Epoch 8/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.2012 - categorical_accuracy: 0.9783 - val_loss: 0.9795 - val_categorical_accuracy: 0.8796
Epoch 9/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.1763 - categorical_accuracy: 0.9807 - val_loss: 1.1262 - val_categorical_accuracy: 0.8739
Epoch 10/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.1750 - categorical_accuracy: 0.9797 - val_loss: 1.2266 - val_categorical_accuracy: 0.8543
Epoch 11/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.1856 - categorical_accuracy: 0.9793 - val_loss: 1.0753 - val_categorical_accuracy: 0.8739
Epoch 12/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.1480 - categorical_accuracy: 0.9835 - val_loss: 1.0895 - val_categorical_accuracy: 0.8711
Epoch 13/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.1540 - categorical_accuracy: 0.9814 - val_loss: 0.9033 - val_categorical_accuracy: 0.8739
Epoch 14/15
2856/2856 [==============================] - 7s 2ms/step - loss: 0.1114 - categorical_accuracy: 0.9874 - val_loss: 0.8353 - val_categorical_accuracy: 0.8936
Epoch 15/15
2856/2856 [==============================] - 7s 3ms/step - loss: 0.1025 - categorical_accuracy: 0.9881 - val_loss: 0.8710 - val_categorical_accuracy: 0.8908

test_acc: 0.8997214484679665
```

``` python
def class_accuracy(y_true, y_pred):
        return np.mean(np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)))
```
