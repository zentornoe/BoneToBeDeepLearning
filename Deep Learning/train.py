import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D, \
    Conv2DTranspose
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau


os.environ['KMP_DUPLICATE_LIB_OK']='True'
base_dir = './'
img_dir = './images/'

checkpoint_path = './checkpoint/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1,
                                                 period=100)


x_train = np.load('dataset/bone/x_data.npy')
y_train_onehot = np.load('dataset/bone/y_data_onehot.npy')
y_val = np.load('dataset/bone/y_data.npy')
x_val = x_train[:20]
y_val_onehot = y_train_onehot[:20]
x_train = x_train[20:]
y_train_onehot = y_train_onehot[20:]
print(x_train.shape, y_train_onehot.shape)

inputs = Input(shape=(640, 640, 1))

net = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(128, kernel_size=1, activation='relu')(net)

#net = UpSampling2D(size=2)(net)
net = Conv2DTranspose(128, kernel_size=1, strides=2, padding='same', activation='relu')(net)
net = Conv2D(128, kernel_size=3, activation='sigmoid', padding='same')(net)

#net = UpSampling2D(size=2)(net)
net = Conv2DTranspose(128, kernel_size=1, strides=2, padding='same', activation='relu')(net)
net = Conv2D(64, kernel_size=3, activation='sigmoid', padding='same')(net)

#net = UpSampling2D(size=2)(net)
net = Conv2DTranspose(64, kernel_size=1, strides=2, padding='same', activation='relu')(net)
outputs = Conv2D(2, kernel_size=3, activation='sigmoid', padding='same')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])

model.summary()

history = model.fit(x_train, y_train_onehot, validation_data=(x_val, y_val_onehot), epochs=5000, batch_size=10, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1,
                      mode='auto', min_lr=1e-05), cp_callback
])

fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss'], 'r')
ax[0, 1].set_title('acc')
ax[0, 1].plot(history.history['acc'], 'b')

ax[1, 0].set_title('val_loss')
ax[1, 0].plot(history.history['val_loss'], 'r--')
ax[1, 1].set_title('val_acc')
ax[1, 1].plot(history.history['val_acc'], 'b--')

preds = model.predict(x_val)
preds = preds[:7]
preds = np.argmax(preds, axis=-1)
fig, ax = plt.subplots(len(x_val), 3, figsize=(10, 100))
plt.figure(figsize=(14, 6))
for i, pred in enumerate(preds):
    plt.subplot(7, 3, 3*i+1)
    plt.imshow(x_val[i].squeeze(), cmap='gray')
    plt.subplot(7, 3, 3*i+2)
    plt.imshow(y_val[i].squeeze(), cmap='gray')
    plt.subplot(7, 3, 3*i+3)
    plt.imshow(pred.squeeze(), cmap='gray')

plt.show()
