import keras
from keras_layer_normalization import LayerNormalization
import keras2onnx

input_layer = keras.layers.Input(shape=(2, 3))
norm_layer = LayerNormalization()(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=norm_layer)
model.compile(optimizer='adam', loss='mse', metrics={},)
model.summary()

model.save('layer_normalization.h5')
#keras.backend.set_learning_phase(0)

onnx_model = keras2onnx.convert_keras(model, 'layer_normalization', debug_mode=1)
keras2onnx.save_model(onnx_model, 'layer_normalization.onnx')
