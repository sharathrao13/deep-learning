import sys
from keras.models import model_from_json


def save_model(model, name):
    try:
        with open(name + '_architecture.json', 'w') as f:
            f.write(model.to_json())
        model.save_weights(name + '_weights.h5', overwrite=True)
        return True
    except:
        print sys.exc_info()
        return False
    
    
def load_model(name):
    with open(name + '_architecture.json') as f:
        model = model_from_json(f.read())
    model.load_weights(name + '_weights.h5')
    return model

def output_at_layer(image, model, layer_num):
    model.layers = model.layers[:layer_num]
    model.compile(loss=model.loss, optimizer=model.optimizer)
    return model.predict(image)
