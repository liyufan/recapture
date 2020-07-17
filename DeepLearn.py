import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import datetime
import matplotlib.pyplot as plt


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
	fig, axes = plt.subplots(1, 5, figsize=(20, 20))
	axes = axes.flatten()
	for img, ax in zip(images_arr, axes):
		ax.imshow(img)
		ax.axis('off')
	plt.tight_layout()
	plt.savefig("a.png")
	plt.show()


if __name__ == '__main__':
	PATH = '/home/lyf/dataset/'
	os.system(f'rm -rf ./logs/')
	train_dir = os.path.join(PATH, 'train')
	validation_dir = os.path.join(PATH, 'test')
	train_r_dir = os.path.join(train_dir, 'RecapturedImage')
	train_s_dir = os.path.join(train_dir, 'SingleCapturedImage')
	validation_r_dir = os.path.join(validation_dir, 'RecapturedImage')
	validation_s_dir = os.path.join(validation_dir, 'SingleCapturedImage')
	num_r_tr = len(os.listdir(train_r_dir))
	num_s_tr = len(os.listdir(train_s_dir))

	num_r_val = len(os.listdir(validation_r_dir))
	num_s_val = len(os.listdir(validation_s_dir))

	total_train = num_r_tr + num_s_tr
	total_val = num_r_val + num_s_val
	print('total training cat images:', num_r_tr)
	print('total training dog images:', num_s_tr)

	print('total validation cat images:', num_r_val)
	print('total validation dog images:', num_s_val)
	print("--")
	print("Total training images:", total_train)
	print("Total validation images:", total_val)

	batch_size = 128
	epochs = 15
	IMG_HEIGHT = 150
	IMG_WIDTH = 150

	image_gen_train = ImageDataGenerator(
		rescale=1. / 255,
		rotation_range=45,
		width_shift_range=.15,
		height_shift_range=.15,
		horizontal_flip=True,
		zoom_range=0.5
	)
	image_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.5)
	image_gen_val = ImageDataGenerator(rescale=1. / 255)
	train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
	                                                     directory=train_dir,
	                                                     shuffle=True,
	                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
	                                                     class_mode='binary')
	'''
	val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
	                                                 directory=validation_dir,
	                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
	                                                 class_mode='binary')
	'''
	# sample_training_images, _ = next(train_data_gen)
	augmented_images = [train_data_gen[0][0][0] for i in range(5)]
	plotImages(augmented_images)
	'''
	model = Sequential([
		Conv2D(16, 3, padding='same', activation='relu',
		       input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
		MaxPooling2D(),
		Dropout(0.2),
		Conv2D(32, 3, padding='same', activation='relu'),
		MaxPooling2D(),
		Conv2D(64, 3, padding='same', activation='relu'),
		MaxPooling2D(),
		Dropout(0.2),
		Flatten(),
		Dense(512, activation='relu'),
		Dense(1)
	])
	model.compile(optimizer='adam',
	              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	              metrics=['accuracy'])

	log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	model.summary()

	history = model.fit(
		train_data_gen,
		steps_per_epoch=total_train // batch_size,
		epochs=epochs,
		validation_data=val_data_gen,
		validation_steps=total_val // batch_size,
		callbacks=[tensorboard_callback]
	)

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs_range = range(epochs)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.show()
	'''
