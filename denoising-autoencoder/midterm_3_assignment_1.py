# -*- coding: utf-8 -*-
"""Midterm 3 Assignment 1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17ihFTlW9AWwVf2Tk566be4jAMpFZigSx

---


#Midterm 3 - Assignment 1

---

##Training of autoencoders for the MNIST dataset

###Import libraries and dataset
"""

from keras.layers import Input, Dense, Dropout, Add
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import optimizers
from keras import regularizers
from keras.backend import stack, concatenate, clear_session
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

clear_session()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train_noisy = x_train + 0.5 * np.random.normal(loc=0.0, scale=1, size=x_train.shape)
x_test_noisy = x_test + 0.5 * np.random.normal(loc=0.0, scale=1, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

"""###1st: shallow AE, 16 neurons"""

encoding_dim = 32

input_img = Input(shape = (28*28,))
encoded_sh = Dense(encoding_dim, activation='relu', activity_regularizer = regularizers.l1(1e-6))(input_img)
decoded_sh = Dense(28*28, activation='sigmoid')(encoded_sh)

autoencoder_shallow = Model(input_img, decoded_sh)
#encoder = Model(input_img, encoded)
#encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]

# create the decoder model
#decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder_shallow.compile(optimizer='nadam', loss='binary_crossentropy')

"""####Fit model and test"""

history_shallow = autoencoder_shallow.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

"""###2nd try: deep AE, 128/64/32 neurons"""

h_dim = [128, 64, 32]

input_img = Input(shape=(28*28, ))
encoded_dp = Dense(h_dim[0], activation='relu')(input_img)
encoded_dp = Dense(h_dim[1], activation='relu')(encoded_dp)
encoded_dp = Dense(h_dim[2], activation='relu')(encoded_dp)

decoded_dp = Dense(h_dim[1], activation='relu')(encoded_dp)
decoded_dp = Dense(h_dim[0], activation='relu')(decoded_dp)
decoded_dp = Dense(28*28, activation='sigmoid')(decoded_dp)

autoencoder_deep = Model(input_img, decoded_dp)

autoencoder_deep.compile(optimizer='nadam', loss='binary_crossentropy')

"""####Fit model and test"""

history_deep = autoencoder_deep.fit(x_train_noisy, x_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_test_noisy, x_test))

"""###3rd try: deep AE, 128/64/32/16/8 neurons, L1 regularized"""

h_dim = [128, 64, 32, 16, 8]
reg = regularizers.l1(1e-8)

input_img = Input(shape=(28*28, ))
encoded = Dense(h_dim[0], activation='relu', activity_regularizer = reg)(input_img)
encoded = Dense(h_dim[1], activation='relu', activity_regularizer = reg)(encoded)
encoded = Dense(h_dim[2], activation='relu', activity_regularizer = reg)(encoded)
encoded = Dense(h_dim[3], activation='relu', activity_regularizer = reg)(encoded)

encoded = Dense(h_dim[2], activation='relu', activity_regularizer = reg)(encoded)
decoded = Dense(h_dim[1], activation='relu', activity_regularizer = reg)(encoded)
decoded = Dense(h_dim[0], activation='relu', activity_regularizer = reg)(decoded)
decoded = Dense(28*28, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='nadam', loss='binary_crossentropy')

"""####Fit model and test"""

history = autoencoder.fit(x_train, x_train, epochs=200, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

"""###Plot losses"""

fig = plt.figure()

plt.plot(history_shallow.history['loss'], label = 'T. loss shallow AE %.4f' %history_shallow.history['loss'][-1])
plt.plot((history_shallow.history['val_loss']),'.', label = 'V. loss shallow AE %.4f' %history_shallow.history['val_loss'][-1])
plt.plot(history_deep.history['loss'], label = 'T. loss deep AE  %.4f' %history_deep.history['loss'][-1])
plt.plot((history_deep.history['val_loss']),'.', label = 'V. loss deep AE %.4f' %history_deep.history['val_loss'][-1])
plt.title('Loss comparison shallow AE (regularized) vs deep AE')
#plt.ylim((0.05, 0.5))
plt.grid()
plt.legend()
plt.show()

fig.savefig('AE_loss.png')

"""###Decode images from different AEs"""

dec_imgs_shallow = autoencoder_shallow.predict(x_test_noisy)
dec_imgs_deep = autoencoder_deep.predict(x_test_noisy)

"""###Visualize predictions"""

n = 5
fig = plt.figure(figsize=(20, 8))
for i in range(n):
    
    ax1 = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i+1000].reshape(28, 28))
    plt.gray()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2 = plt.subplot(3, n, i + 1 + n)
    plt.imshow(dec_imgs_shallow[i+1000].reshape(28, 28))
    plt.gray()
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    ax3 = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(dec_imgs_deep[i+1000].reshape(28, 28))
    plt.gray()
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    
ax1.set_title('Noisy test images')
ax2.set_title('Shallow AE reconstruction')
ax3.set_title('Deep AE reconstruction')
plt.show()

fig.savefig('test_figures.png')

"""###Attach classifier to deep AE to classify dataset"""

encoded_sh.trainable = False
classifier = Dense(10, activation = 'softmax')(encoded_sh)

Classifier = Model(input_img, classifier)

Classifier.compile(optimizers.Nadam(lr = 2e-4), 
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

"""####Fit classifier"""

history_classifier = Classifier.fit(x_train, y_train,epochs= 100, batch_size=256, shuffle=True, validation_data=(x_test, y_test))

"""####*Just in case* plot classifier performance"""

fig, (acc, loss) = plt.subplots(2,1)
fig.set_size_inches(6, 10, forward=True)

acc.plot(history_classifier.history['acc'], label = 'Training acc.: %.4f' %history_classifier.history['acc'][-1])
acc.plot((history_classifier.history['val_acc']),'.', label = 'Validation acc.: %.4f' %history_classifier.history['val_acc'][-1])
acc.grid()
acc.set_ylim((0.7,1.01))
acc.legend()

loss.plot(history_classifier.history['loss'], label = 'Training loss: %.4f' %history_classifier.history['loss'][-1])
loss.plot((history_classifier.history['val_loss']),'.', label = 'Validation loss: %.4f' %history_classifier.history['val_loss'][-1])
loss.grid()
loss.set_ylim((-0.099,1.0))
loss.legend()


plt.show()

fig.savefig('classifier.png')

"""####Predict labels"""

encoded_label = Classifier.predict(x_test)

"""###Encode test images and visualize data with t-SNE"""

encoder = Model(input_img, encoded_sh)
encoded_imgs = encoder.predict(x_test)

col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:grey']
custom_legend = [Line2D([0], [0], color = col[0], marker = '.', lw=1),
                Line2D([0], [0], color = col[1], marker = '.', lw=1),
                Line2D([0], [0], color = col[2], marker = '.', lw=1),
                Line2D([0], [0], color = col[3], marker = '.', lw=1),
                Line2D([0], [0], color = col[4], marker = '.', lw=1),
                Line2D([0], [0], color = col[5], marker = '.', lw=1),
                Line2D([0], [0], color = col[6], marker = '.', lw=1),
                Line2D([0], [0], color = col[7], marker = '.', lw=1),
                Line2D([0], [0], color = col[8], marker = '.', lw=1),
                Line2D([0], [0], color = col[9], marker = '.', lw=1),]

for p in [5, 50, 100, 250, 500, 1000]:
    X_embedded = TSNE(n_components = 2, perplexity = p, learning_rate = 50).fit_transform(encoded_imgs)
    fig = plt.figure()

    for i in range(len(X_embedded)):
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c = col[np.argmax(encoded_label[i])], s = 1)

    plt.xlim((-80, 80))
    plt.title('t-SNE representation with perplexity ' + str(p))
    plt.legend(custom_legend, ['0', '1', '2','3', '4', '5', '6', '7', '8', '9'])
    #plt.show()

    fig.savefig('tsne_dae_' + str(p) + '.png')

"""###Test AE with random noise

**Generate white noise sample**
"""

x_n = np.random.normal(loc=0.0, scale = 1, size=(1,28*28))
x_n = np.clip(x_n, 0., 1.)

x_n

"""**First try**"""

x_looped = x_n

for i in range(50):
    x_looped = autoencoder_deep.predict(x_looped)

"""**Second try**"""

input_img = Input((28*28,))

out_list = []

out_list.append(autoencoder_deep(input_img))

l = 10

for i in range(l):
    out_list.append(autoencoder_deep(out_list[i]))

stacked_ae = Model(input_img, out_list[l])

x_stacked = stacked_ae.predict(x_n)

input_img = Input((28*28,))
out_list = []

out_list.append(autoencoder_deep(input_img))

l = 10

for k in range(l):
    out_list.append(autoencoder_deep(out_list[k]))

stacked_ae = Model(input_img, out_list[l])

    
for j in range(25):
    x_n = np.random.normal(loc=0.0, scale = 1, size=(1,28*28))
    x_n = np.clip(x_n, 0., 1.)
    
    x_looped = x_n
    
    for i in range(10):
        x_looped = autoencoder_deep.predict(x_looped)
        
    x_stacked = stacked_ae.predict(x_n)
    
    fig = plt.figure(figsize = (10,8))

    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(x_n.reshape(28, 28))
    plt.gray()
    ax1.set_title('Sample')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(x_looped.reshape(28, 28))
    plt.gray()
    ax2.set_title('Looped AE')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(x_stacked.reshape(28, 28))
    plt.gray()
    ax3.set_title('Stacked AE')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    fig.savefig('sample_' + str(j) + '.png')

"""**Plot result**"""

fig = plt.figure(figsize = (10,8))

ax1 = plt.subplot(1, 3, 1)
plt.imshow(x_n.reshape(28, 28))
plt.gray()
ax1.set_title('Sample')
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)

ax2 = plt.subplot(1, 3, 2)
plt.imshow(x_looped.reshape(28, 28))
plt.gray()
ax2.set_title('Looped AE')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

ax3 = plt.subplot(1, 3, 3)
plt.imshow(x_stacked.reshape(28, 28))
plt.gray()
ax3.set_title('Stacked AE')
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

def build_base():
    h_dim = [128, 64, 32]

    input_dp = Input(shape=(28*28, ))
    encoded_dp = Dense(h_dim[0], activation='relu')(input_dp)
    encoded_dp = Dense(h_dim[1], activation='relu')(encoded_dp)
    encoded_dp = Dense(h_dim[2], activation='relu')(encoded_dp)

    decoded_dp = Dense(h_dim[1], activation='relu')(encoded_dp)
    decoded_dp = Dense(h_dim[0], activation='relu')(decoded_dp)
    output_dp = Dense(28*28, activation='sigmoid')(decoded_dp)

    ae_model = Model(input_img, output)

    return input_dp, output_dp, ae_model

input_1, output_1, model_1 = build_base()
input_2, output_2, model_2 = build_base()

model_1.set_weights(autoencoder_deep.get_weights())
model_2.set_weights(autoencoder_deep.get_weights())