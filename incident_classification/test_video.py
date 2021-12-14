from tensorflow.keras import models
import config as cf
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np 
import model_create as eff
import cv2


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
path_test = os.path.join(cf.DATA_TEST_DIR, class_, "DXEhOr6R_RQgt.mp4")
    
cap = cv2.VideoCapture(path_test)
if not (cap.isOpened()):
    print("<h1>Can not open video file !!!</h1>")
while(cap.isOpened()):
    ret, frame = cap.read()
    
    img = frame / 255
    img = trans.resize(img, target_size)
    img = np.reshape(img,(1,)+img.shape)
    results = model.predict_generator(img,1,verbose=0)
    res = np.reshape(results, results.shape[1])
    max_value = max(res)
    max_index = res.tolist().index(max_value)
    if(max_value >= 0.97):
        cv2.putText(frame, str(classes[max_index]) + '---' + str(max_value), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), cv2.LINE_4)
    else:
        cv2.putText(frame, str("not incidents"), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), cv2.LINE_4)
    
    if ret == True:
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
