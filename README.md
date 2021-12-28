# IEEE_BIGDATA_2021
## BACKEND
Link dowwnload Data for backend: https://drive.google.com/file/d/1amRQBmJ3VDUcaN-G_PKLSP4sx9IKfHh5/view?usp=sharing
you have to extract this rar file follow by direction ***"backend/static"***.

You will run ```api_controller.py``` to start all the API.
## INCIDENTS CLASSIFICATION
* **Data**: Link dowwnload Data: https://drive.google.com/file/d/10jRIIN7HHTMPU4FSRw7sADGShpFydFof/view?usp=sharing
you have to extract this rar file follow by direction ***"incident_classification/incidents_cleaned"***. It will include 3 folder: train, val, test with 8 classes of incidents.
* **Training**: the python code for training will be in ```train_run.py```.
* **Testing**: 
  * Image: the python code will be in ```test_on_image.py```.
  * Video: the python code will be in ```test_on_video.py```.
 
I have pushed the best weight into the project ```EfficientNetB6_weight.hdf5``` , you can run the test with this best weight.

***Note***: ```config.py``` this file will be include all the config information about folder direction of data and parameters for training, you can customize it according to your purpose. ```model_create.py``` this file is the model creating you can modify it according to your purpose too.
## FRONTEND
You have extract the Node modules file ```node_modules.rar``` follow by direction ***"frontend/node_modules"*** and you can follow the ```README.md``` into ***frontend*** to start the web frontend.
