import numpy as np
from tensorflow import keras
from keras import layers
from keras import losses
from keras.utils import to_categorical

num_data = 1000
data = np.loadtxt('iris.data',delimiter=',')
xdata = data[:,:-1]
ydata = to_categorical(data[:,-1])

## Building the model
#inputs = keras.Input(shape=(xdata.shape[-1],))
#x = layers.Dense(10,activation='relu')(inputs)
#x = layers.Dense(10,activation='relu')(x)
#x = layers.Dense(10,activation='relu')(x)
#x = layers.Dense(3,activation='softmax')(x)
#print('x: ', x)
#
#model = keras.Model(inputs=inputs,outputs=x)
#model.compile(loss=losses.CategoricalCrossentropy(),metrics='accuracy')
#model.summary()

#
#y = model.layers[-1]
#
#for idx,l in enumerate(model.layers):
#    print('Layer [%d] --> ' %(idx), l.weights)
#
#print('y.weights: ',y.weights)
#
##import sys
#sys.exit(0)
#model.fit(xdata,ydata,batch_size=32,epochs=100)

# Transfer learning now
#model.save('frozen_model')
model = keras.models.load_model('./frozen_model')
#y = model.layers[-1]
#y = layers.Activation(keras.activations.sigmoid)(y)
#print('model.layers[-1].weights: ', model.layers[-1].weights)

y = None
inputs = keras.Input(shape=(xdata.shape[-1],))
for idx,l in enumerate(model.layers[:-1]):
    print('idx: ', idx, ', l: ', l)
    if idx == 0:
        y = inputs
    else:
        y = l(y)

y = layers.Dense(3,name='y_dense_last')(y)
y = layers.Activation('softmax')(y)

new_model = keras.Model(inputs=inputs,outputs=y)

new_model.layers[-2].set_weights(model.layers[-1].weights)

new_model.compile(loss=losses.CategoricalCrossentropy(),metrics='accuracy')

new_model.summary()
model.summary()

#model.trainable = False
#x2 = model(inputs)
#x2 = layers.Dense(3,activation='softmax')(x2)
#model.summary()
#new_model = keras.Model(inputs=inputs,outputs=x2)
#new_model.compile(loss=losses.CategoricalCrossentropy(),metrics='accuracy')
#new_model.summary()
#new_model.fit(xdata,ydata,batch_size=32,epochs=100)

model_preds = model.predict(xdata)
new_model_preds = new_model.predict(xdata)
print('np.allclose(model_preds,new_model_preds): ', np.allclose(model_preds,new_model_preds))
