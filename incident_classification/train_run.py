import os
import config as cf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import model_create as eff

#Use this to check if the GPU is configured correctly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def main():

    # Create Model
    model = eff.Model_EfficientNetB6()
    
    # Prepare Data Generator
    train_datagen = ImageDataGenerator(
        rescale = 1.0/255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )

    test_datagen = ImageDataGenerator(rescale = 1.0/255)

    train_generator = train_datagen.flow_from_directory(
        cf.DATA_TRAIN_DIR,
        target_size = (cf.IMG_WIDTH, cf.IMG_HEIGHT),
        batch_size = cf.BATCH_SIZE,
        class_mode = 'categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        cf.DATA_VALIDATE_DIR,
        target_size = (cf.IMG_WIDTH, cf.IMG_HEIGHT),
        batch_size = cf.BATCH_SIZE,
        class_mode = 'categorical'
    )

    # Trainning Data
    # 1. Compile model
    opt = optimizers.Adam(learning_rate=0.0002)
    model.compile(
        loss = 'categorical_crossentropy',
        #optimizer = optimizers.RMSprop(lr=0.0002),
        optimizer = opt,
        metrics = ['acc']
    )

    # 2. Create checkpoint to save model weight
    model_checkpoint = ModelCheckpoint('EfficientNetB6_weight.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.summary()

    #
    reduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, mode='min', verbose=1)
    # 3. train
    history_train = model.fit_generator(
        train_generator,
        steps_per_epoch = 2000,
        epochs = 15,
        validation_data = validation_generator,
        validation_steps = 50,
        verbose = 1,
        # use_multiprocessing = True,
        # workers = 4,
        callbacks = [model_checkpoint, reduceLr]
    )

if __name__=='__main__':
    main()
