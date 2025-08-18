from tensorflow.keras.models import load_model
model = load_model('outputs/modelv2.keras')
model.summary()


tensorflowjs_converter --input_format=keras outputs/modelv2.h5 ../outputs/model_tfjs
