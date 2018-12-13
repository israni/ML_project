from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute

num_actions = 4

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

input_shape = INPUT_SHAPE + (WINDOW_LENGTH,)
print(input_shape)

model = Sequential()
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_actions))
model.add(Activation('linear'))
print(model.summary())
