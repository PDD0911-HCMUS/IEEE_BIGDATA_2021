from tensorflow.keras import models
import config as cf
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np 
import model_create as eff


'''REMEMBER
LOAD MODEL FIRST -> NOT COMPILE 
LOAD WEIGHTS  
'''

model = eff.Model_EfficientNetB6()

model.load_weights('EfficientNetB6_weight.hdf5')
model.summary()

target_size = (224,224)

classes = [ 'animals',
            'collapse',
            'crash',
            'fire',
            'flooding',
            'landslide',
            'snow',
            'treefall']

class_ = 'test_temp'
path_test = os.path.join(cf.DATA_TEST_DIR, class_)
for item in os.listdir(path_test):
    #print(os.path.join(path_test, item))
    img = io.imread(os.path.join(path_test, item))
    img = img / 255
    img = trans.resize(img, target_size)
    img = np.reshape(img,(1,)+img.shape)
    results = model.predict_generator(img,1,verbose=0)
    res = np.reshape(results, results.shape[1])
    print(res)
    max_value = max(res)
    max_index = res.tolist().index(max_value)
    if(max_value >= 0.85):
        print(item, " -- " , max_value, classes[max_index])
    else:
        print("not incidents")
    

