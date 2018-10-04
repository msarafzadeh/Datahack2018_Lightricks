import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Input, Conv2D, Activation, merge, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
from scipy.misc import imread as imread
from scipy.ndimage.filters import convolve as conv
from skimage.color import rgb2gray as rgb2gray, gray2rgb
import matplotlib.pyplot as plt
import random
import pickle

# Constants
images = {}  # images dictionary, each element is a image representing a user, (key = id ,value = image)
churned = {}  # churned dictionary, (key = id ,value = '0','1'(churned/not churned))
TRAIN_DIR = 'datahack-master/train'
KAG_TRAIN_DIR = '../input/lighttricks-churn'


###############################   Data Preparation    ################################

def get_user_usage(user_id):
	user_details = train_usage_data.loc[(train_usage_data['id'] == user_id),]
	return user_details


def get_user_image(user_id, user_details):  # TODO: cut outliers
	user_image = np.zeros((7, 17))
	user_details['clean_date'] = user_details["end_use_date"].apply(lambda x: x.split()[0][-2:])
	sub_date = train_users_data.loc[train_users_data['id'] == user_id,]['subscripiton_date'].values[0].split(" ")[0][
	           -2:]
	user_usage_per_day = user_details.groupby('clean_date')['usage_duration'].sum()
	feature_usage_per_day = user_details.groupby(['clean_date', 'feature_name'])['usage_duration'].sum()
	avg_feature_use = feature_usage_per_day / user_usage_per_day
	# print(avg_feature_use['14']['Adjust'])
	for date in user_details["clean_date"].unique():
		row = (int(date) - int(sub_date)) % 30
		if row == 7:
			row = 6
		# print("current date: ", int(date), "sub_date: ", sub_date)
		for col in range(len(train_usage_data['feature_name'].unique())):
			try:
				user_image[row][col] = avg_feature_use[date][train_usage_data['feature_name'].unique()[col]]
			except KeyError:
				user_image[row][col] = 0
	user_image = np.floor(user_image.dot(255))
	user_image = user_image[:, :10]
	padding = np.zeros(23, 10)
	user_image = np.vstack((user_image, padding))
	print(user_image.shape)
	print(user_image)
	# plt.imshow(user_image, cmap='gray')
	# plt.show()
	return user_image


## Convert each user to image
def create_images_for_net():  # 6998 users
	global images
	id_df = train_users_data[['id']]
	for index, row in id_df.iterrows():
		print(index)
		images[row['id']] = get_user_image(row['id'], get_user_usage(row['id']))


def create_churn_dict():
	global churned
	keys = list(train_users_data['id'])
	values = list(train_users_data['churned'])
	churned = dict(zip(keys, values))


def batch_generator(images_gen, batch_size):
	# using images dictionary to create batches
	while True:
		# ( h, w) = (7, 17)
		source_batch = np.zeros((batch_size, 1, 30, 10))
		target_batch = np.zeros((batch_size, 1, 30, 10))
		for i in range(batch_size):
			keys_set = set(images_gen.keys())
			id = random.sample(keys_set, 1)
			#             id = np.random.randint(len(images_gen))
			source_batch[i, 0, :, :] = images_gen[id[0]]
			target_batch[i, 0, :, :] = churned[id[0]]
		yield (source_batch.astype(np.float32), target_batch.astype(np.float32))


###############################   Build Net    ################################

def build_cnn_model(height, width, num_channels):
	"""
	This function gets height, width and number of channels and returns an
	untrained Keras model.
	:param height: an integer represents a shape height parameter of the input
	 tensor
	:param width: an integer represents a shape width parameter of the input
	 tensor
	:param num_channels: number of channels for each of the activation layers
	:return:  an untrained Keras model
	"""
	# input --> convolve --> activation --> maxpool --> convolve --> activation -->
	# maxpool --> FC --> FC --> FC --> output

	# Model 0
	# a = Input(shape=(1, height, width))
	# print("input shape:", a.shape)
	# # Two convolutional layers with maxpooling
	# b = Convolution2D(4, 3, 1, border_mode='same')(a)
	# c = Activation('relu')(b)
	# print("convolution shape:", a.shape)
	# # g = MaxPooling2D(pool_size=(1, 2))(b)
	# # print("pooling shape:", g.shape)
	# i = Flatten()(g)
	# j = Dense(128)(i)
	# print("dense shape:", j.shape)
	# p = Dense(2, activation='softmax')(j)
	# print("dense_2 shape:", p.shape)
	# model = Model(input=a, output=p)

	# # Model 1
	a = Input(shape=(1, height, width))
	print("a shape:", a.shape)
	# Two convolutional layers with maxpooling
	b = Conv2D(4, (7, 1), padding="same")(a)
	print("b shape:", b.shape)
	c = Activation('relu')(b)
	print("c shape:", c.shape)
	e = Conv2D(2, (1, 10), padding="same")(c)
	print("e shape:", e.shape)
	f = Activation('relu')(e)
	print("f shape:", f.shape)
	g = MaxPooling2D(pool_size=(1, 2))(f)
	print("g shape:", g.shape)
	i = Flatten()(g)
	print("i shape:", i.shape)
	j = Dense(128)(i)
	print("j shape:", j.shape)
	p = Dense(1, activation='softmax')(j)
	print("p shape:", p.shape)
	print("p: ",p)
	model = Model(inputs=a, outputs=p)

	# Model 2
	#     a = Input(shape=(1, height, width))
	#     print("a shape:", a.shape)
	#     # Two convolutional layers with maxpooling
	#     b = Convolution2D(4, 3, 1, border_mode='same')(a)
	#     print("b shape:", b.shape)
	#     c = Activation('relu')(b)
	#     print("c shape:", c.shape)
	#     d = MaxPooling2D(pool_size=(1, 2))(c)
	#     print("d shape:", d.shape)
	#     e = Convolution2D(3, 1, 4, border_mode='same')(d)
	#     print("e shape:", e.shape)
	#     f = Activation('relu')(e)
	#     print("f shape:", f.shape)
	#     g = MaxPooling2D(pool_size=(1, 2))(f)
	#     print("g shape:", g.shape)
	#     h = Dropout(0.25)(g)

	#     # Three fully connected layers
	#     i = Flatten()(h)
	#     j = Dense(100)(i)
	#     k = Dropout(0.2)(j)
	#     l = Dense(40)(k)
	#     m = Dropout(0.2)(l)
	#     n = Dense(20)(m)
	#     o = Dropout(0.2)(n)
	#     p = Dense(2, activation='softmax')(o)
	#     print("p shape:", p.shape)
	#     model = Model(input=a, output=p)
	return model


###############################   Customized Metrices    ################################

def precision(y_true, y_pred):
	# Calculates the precision
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision


def recall(y_true, y_pred):
	# Calculates the recall
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def fbeta_score(y_true, y_pred, beta=1):
	# Calculates the F score, the weighted harmonic mean of precision and recall.

	if beta < 0:
		raise ValueError('The lowest choosable beta is zero (only precision).')

	# If there are no true positives, fix the F score at 0 like sklearn.
	if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
		return 0

	p = precision(y_true, y_pred)
	r = recall(y_true, y_pred)
	bb = beta ** 2
	fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
	return fbeta_score


def fmeasure(y_true, y_pred):
	# Calculates the f-measure, the harmonic mean of precision and recall.
	return fbeta_score(y_true, y_pred, beta=1)


###############################   Training    ################################

def train_model(model, images, batch_size, samples_per_epoch,
                num_epochs, num_valid_samples):
	"""
	This function gets images, a model , corrupted function and training
	parameters and trains the model accordingly.
	:param model: a general neural network model for image restoration
	:param images: a list of file paths pointing to imagefile
	:param batch_size: An integer (represents a size of a batch of examples)
	:param samples_per_epoch:An integer (represents number of samples in epoch)
	:param num_epochs: An integer (represents the number of epochs)
	:param num_valid_samples: An integer (represents the number of samples in
	the validation set to test on after every epoch
	:return: None
	"""

	# First create training set and validation set using outer function
	# split images 80-20 split for training - validation set
	border = int(len(images) * 0.8)
	train_set = dict(list(images.items())[:border])
	val_se = dict(list(images.items())[border:])
	train_set_gen = batch_generator(train_set, batch_size)
	valid_set_gen = batch_generator(val_se, batch_size)

	# Second prepare the model
	# TODO check custom metrics
	model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
	              metrics=['accuracy', fmeasure, recall, precision])

	# third train the model
	model.fit_generator(train_set_gen, samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
	                    validation_data=valid_set_gen, nb_val_samples=num_valid_samples)
	# TODO check save function
	model.save('churn_cnn.h5')
	return


if __name__ == '__main__':
	print("test CNN.....")
	# TODO check if need to be global variables
	train_users_data = pd.read_csv("train_users_data.csv")
	# train_users_data.info()
	train_usage_data = pd.read_csv("train_usage_data.csv")
	# train_users_data.info()

	# Prepare the data for conversion:
	# creates images dictionary
	#     create_images_for_net()
	f = open("images_train.pkl", "rb")
	bin_data = f.read()
	images = pickle.loads(bin_data)[
		0]  # images dictionary, each element is a image representing a user, (key = id ,value = image)
	print("Images Creation Done.....")

	create_churn_dict()
	print("churn dict Creation Done.....")

	# build model
	model = build_cnn_model(30, 10, 1)
	# model = load_model('churn_cnn.h5')  # load model
	print("Building Model Done.....")

	# Train and save model

	train_model(model, images, batch_size=100, samples_per_epoch=10000, num_epochs=20, num_valid_samples=1000)
	print("Training Model Done.....")

# 1. divide user usage to days
# joined_data = train_usage_data.join(train_users_data.set_index('id'),on='id')
# print(joined_data)
# 2. group by user and then group by days
# 3. for each feature sum daily use
# 4. modify features exceeding use - find average usage and cut the outliers
# 4. normalize features use to 0% - 100%
