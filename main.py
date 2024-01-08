import gc
import glob
import itertools

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pygbif as gbif
from keras.preprocessing.image import ImageDataGenerator
from keras.src.metrics import Recall, Precision, F1Score
from keras.src.utils.layer_utils import print_summary
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.utils import resample
from pygbif import species, occurrences

# mudar para 'peçonhento' e 'não-peçonhento'
labels = ['pequeno', 'medio', 'grande']


# def snakes_in_brazil():
# # search for snakes
#     snakes = species.search(q='Serpentes', limit=1000) # adjust limit

# # filter snakes found in Brazil
#     brazil_snakes = []
#     for snake in snakes['results']:
#         species_key = snake['key']
#         occ = occurrences.search(country='BR', taxonKey=species_key, limit=1)
#         if occ['count'] > 0:
#             brazil_snakes.append(snake['scientificName'])
# # now, brazil_snakes contain a list of snakes found in Brazil

#     snake_images = {}

#     for snake_name in brazil_snakes:
#         # search for occurences with images for each species
#         occ_data = occurrences.search(scientificName=snake_name, country='BR', mediaType='StillImage', limit=20)

#         # extract image URLs
#         images = []
#         for record in occ_data['results']:
#             if 'media' in record and 'licence' in record and 'basisOfRecord' in record:
#                 if 'creativecommons.org' in record['licence']: # filtering for Creative Commons licences
#                     for media in record['media']:
#                         if 'identifier' in media:
#                             images.append(media['identifier'])
#         snake_images[snake_name] = images
#         # now snake_images is a dictionary with species names as keys as list of image URLs as values

# not needed
def pre_built_dogs_cats():
    CATS = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian',
            'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx']
    cats_images = []
    dogs_images = []
    for img in glob.glob('images/*.jpg'):
        if any(cat in img for cat in CATS):
            cats_images.append(img)
        else:
            dogs_images.append(img)
    print('There are {} images of cats'.format(len(cats_images)))
    print('There are {} images of dogs'.format(len(dogs_images)))
    # shuffle the lists
    np.random.shuffle(cats_images)
    np.random.shuffle(dogs_images)
    # split the data into train, validation and test sets
    train_d, val_d, test_d = np.split(dogs_images, [int(len(dogs_images) * 0.7), int(len(dogs_images) * 0.8)])
    train_c, val_c, test_c = np.split(cats_images, [int(len(cats_images) * 0.7), int(len(cats_images) * 0.8)])
    train_dog_df = pd.DataFrame({'image': train_d, 'label': 'dog'})
    val_dog_df = pd.DataFrame({'image': val_d, 'label': 'dog'})
    test_dog_df = pd.DataFrame({'image': test_d, 'label': 'dog'})
    train_cat_df = pd.DataFrame({'image': train_c, 'label': 'cat'})
    val_cat_df = pd.DataFrame({'image': val_c, 'label': 'cat'})
    test_cat_df = pd.DataFrame({'image': test_c, 'label': 'cat'})
    train_df = pd.concat([train_dog_df, train_cat_df])
    val_df = pd.concat([val_dog_df, val_cat_df])
    test_df = pd.concat([test_dog_df, test_cat_df])
    print('There are {} images for training'.format(len(train_df)))
    print('There are {} images for validation'.format(len(val_df)))
    print('There are {} images for testing'.format(len(test_df)))
    BATCH_SIZE = 32
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    trainGenerator = ImageDataGenerator(rescale=1. / 255.)
    valGenerator = ImageDataGenerator(rescale=1. / 255.)
    testGenerator = ImageDataGenerator(rescale=1. / 255.)
    trainDataset = trainGenerator.flow_from_dataframe(
        dataframe=train_df,
        class_mode="binary",
        x_col="image",
        y_col="label",
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    valDataset = valGenerator.flow_from_dataframe(
        dataframe=val_df,
        class_mode='binary',
        x_col="image",
        y_col="label",
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    testDataset = testGenerator.flow_from_dataframe(
        dataframe=test_df,
        class_mode='binary',
        x_col="image",
        y_col="label",
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    images, labels = next(iter(testDataset))
    print('Batch shape: ', images.shape)
    print('Label shape: ', labels.shape)
    plt.imshow(images[3])
    plt.show()
    print('Label: ', labels[3])
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(512, (3, 3), activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    epochs = 5
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(trainDataset, epochs=epochs, validation_data=(valDataset))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    plt.show()
    loss, acc = model.evaluate(testDataset)
    print('Loss:', loss)
    print('Accuracy:', acc)


def custom_neural_net():
    # test_images, test_labels = next(test_batches)

    # plotImages(test_images)
    # print(test_labels)

    k_fold_path = 'mixed_dataset/k-fold'

    k_fold_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=k_fold_path, target_size=(224, 224), classes=labels, batch_size=170)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    k_fold_input, k_fold_labels = k_fold_batches.next()

    fold_no = 1
    history_per_fold = []
    histories = {'accuracy': [], 'f1_score': [], 'val_accuracy': [], 'val_f1_score': []}
    scores_per_fold = []
    # models_per_fold = []
    predictions = []
    correct_predictions = []

    for train, test in kf.split(k_fold_input, k_fold_labels):
        print(f'Training for fold {fold_no} ...')

        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(224, 224, 3)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(512, (3, 3), activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(units=3, activation='softmax')
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), F1Score()])

        history = model.fit(k_fold_input[train], k_fold_labels[train],
                            validation_data=(k_fold_input[test], k_fold_labels[test]), epochs=20, verbose=2)
        history_per_fold.append(history)
        histories['accuracy'].append(history.history['accuracy'])
        histories['f1_score'].append(np.mean(history.history['f1_score'], axis=-1))
        histories['val_accuracy'].append(history.history['val_accuracy'])
        histories['val_f1_score'].append(np.mean(history.history['val_f1_score'], axis=-1))

        print_summary(model)

        model.save('models/custom/' + str(fold_no))
        # model = keras.models.load_model('models/custom/' + str(fold_no))
        # models_per_fold.append(model)

        scores = model.evaluate(k_fold_input[test], k_fold_labels[test], verbose=2)
        scores_per_fold.append(scores)

        predict = model.predict(x=k_fold_input[test])
        predictions.append(np.argmax(predict, axis=1))
        correct_predictions.append(np.argmax(k_fold_labels[test], axis=-1))

        fold_no = fold_no + 1

    predictions = np.concatenate(predictions)
    correct_predictions = np.concatenate(correct_predictions)
    best_fold_index = 0
    best_acc = 0.0
    best_loss = 1.0
    accs = []
    precisions = []
    recalls = []
    f1s = []
    for score in scores_per_fold:
        accs.append(score[1])
        precisions.append(score[2])
        recalls.append(score[3])
        f1s.append(np.mean(score[4]))
        if score[1] > best_acc:
            best_acc = score[1]
            best_loss = score[0]
            best_fold_index = scores_per_fold.index(score)
        elif score[1] == best_acc and score[0] < best_loss:
            best_acc = score[1]
            best_loss = score[0]
            best_fold_index = scores_per_fold.index(score)

    print('best scores in fold: ' + str(best_fold_index))
    print(scores_per_fold[best_fold_index])
    print('Accuracy: ', (np.mean(accs)))
    print('Precision: ', (np.mean(precisions)))
    print('Recall: ', (np.mean(recalls)))
    print('F1: ', (np.mean(f1s)))

    # del models_per_fold
    del k_fold_batches
    del test
    del scores
    del model
    del train
    del kf
    del f1s
    del recalls
    del precisions
    del accs

    gc.collect()

    for i in range(len(histories['accuracy'])):
        acc = histories['accuracy'][i]
        val_acc = histories['val_accuracy'][i]
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('models/custom/custom_kfold' + str(i) + '_acc.svg')
        plt.show()

        f1_score = histories['f1_score'][i]
        val_f1_score = histories['val_f1_score'][i]
        plt.plot(epochs, f1_score, 'y', label='Training F1 score')
        plt.plot(epochs, val_f1_score, 'r', label='Validation F1 score')
        plt.title('Training and validation F1 score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 score')
        plt.legend()
        plt.savefig('models/custom/custom_kfold' + str(i) + '_f1.svg')
        plt.show()

    # cm_ong = confusion_matrix(y_true=correct_predictions, y_pred=predictions)
    # plot_confusion_matrix(cm=cm_ong, classes=labels, title="custom net 5-Fold", save_file='models/custom/custom_kfold_cm.svg')


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes, title, save_file=None, normalize=False, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted Label')
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def vgg16_neural_net():
    k_fold_path = 'mixed_dataset/k-fold'

    k_fold_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=k_fold_path, target_size=(224, 224), classes=labels, batch_size=170)
    # como fazer leitura das imagens para gerar os k-folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    k_fold_input, k_fold_labels = k_fold_batches.next()

    fold_no = 1
    history_per_fold = []
    histories = {'accuracy': [], 'f1_score': [], 'val_accuracy': [], 'val_f1_score': []}
    scores_per_fold = []
    # models_per_fold = []
    predictions = []
    correct_predictions = []

    for train, test in kf.split(k_fold_input, k_fold_labels):
        print(f'Training for fold {fold_no} ...')

        vgg16_model = tf.keras.applications.vgg16.VGG16()

        model = keras.Sequential()
        for layer in vgg16_model.layers[:-1]:
            model.add(layer)

        for layer in model.layers:
            layer.trainable = False

        # units => número de espécies das quais eu tenho imagens
        model.add(keras.layers.Dense(units=3, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), F1Score()])

        history = model.fit(k_fold_input[train], k_fold_labels[train],
                            validation_data=(k_fold_input[test], k_fold_labels[test]), epochs=20, verbose=2)
        history_per_fold.append(history)
        histories['accuracy'].append(history.history['accuracy'])
        histories['f1_score'].append(np.mean(history.history['f1_score'], axis=-1))
        histories['val_accuracy'].append(history.history['val_accuracy'])
        histories['val_f1_score'].append(np.mean(history.history['val_f1_score'], axis=-1))

        model.save('models/vgg16/' + str(fold_no))
        # model = keras.models.load_model('models/vgg16/' + str(fold_no))
        # models_per_fold.append(model) ----ignorar

        scores = model.evaluate(k_fold_input[test], k_fold_labels[test], verbose=2)
        scores_per_fold.append(scores)

        predict = model.predict(x=k_fold_input[test])
        predictions.append(np.argmax(predict, axis=1))
        correct_predictions.append(np.argmax(k_fold_labels[test], axis=-1))

        fold_no = fold_no + 1

    predictions = np.concatenate(predictions)
    correct_predictions = np.concatenate(correct_predictions)
    best_fold_index = 0
    best_acc = 0.0
    best_loss = 1.0
    accs = []
    precisions = []
    recalls = []
    f1s = []
    for score in scores_per_fold:
        accs.append(score[1])
        precisions.append(score[2])
        recalls.append(score[3])
        f1s.append(np.mean(score[4]))
        if score[1] > best_acc:
            best_acc = score[1]
            best_loss = score[0]
            best_fold_index = scores_per_fold.index(score)
        elif score[1] == best_acc and score[0] < best_loss:
            best_acc = score[1]
            best_loss = score[0]
            best_fold_index = scores_per_fold.index(score)

    print('best scores in fold: ' + str(best_fold_index))
    print(scores_per_fold[best_fold_index])
    print('Accuracy: ', (np.mean(accs)))
    print('Precision: ', (np.mean(precisions)))
    print('Recall: ', (np.mean(recalls)))
    print('F1: ', (np.mean(f1s)))

    # del models_per_fold
    del k_fold_batches
    del test
    del scores
    del model
    del train
    del kf
    del f1s
    del recalls
    del precisions
    del accs

    gc.collect()

    for i in range(len(histories['accuracy'])):
        acc = histories['accuracy'][i]
        val_acc = histories['val_accuracy'][i]
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('models/vgg16/vgg16_kfold' + str(i) + '_acc.svg')
        plt.show()

        f1_score = histories['f1_score'][i]
        val_f1_score = histories['val_f1_score'][i]
        plt.plot(epochs, f1_score, 'y', label='Training F1 score')
        plt.plot(epochs, val_f1_score, 'r', label='Validation F1 score')
        plt.title('Training and validation F1 score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 score')
        plt.legend()
        plt.savefig('models/vgg16/vgg16_kfold' + str(i) + '_f1.svg')
        plt.show()

    cm_ong = confusion_matrix(y_true=correct_predictions, y_pred=predictions)
    plot_confusion_matrix(cm=cm_ong, classes=labels, title="VGG16 5-Fold", save_file='models/vgg16/vgg16_kfold_cm.svg')


def predict_and_plot(file_name):
    k_fold_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory='mixed_dataset/k-fold', target_size=(224, 224), classes=labels, batch_size=50)

    final_model = keras.models.load_model(file_name)

    predictions = final_model.predict(x=k_fold_batches)
    cm = confusion_matrix(y_true=k_fold_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    plot_confusion_matrix(cm=cm, classes=labels, title="Mixed dataset")

    del predictions
    del cm
    del k_fold_batches

    gc.collect()

    test_ong_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory='ong_dataset', target_size=(224, 224), classes=labels, batch_size=70,
                             shuffle=False)
    predictions_ong = final_model.predict(x=test_ong_batches)
    cm_ong = confusion_matrix(y_true=test_ong_batches.classes, y_pred=np.argmax(predictions_ong, axis=-1))
    plot_confusion_matrix(cm=cm_ong, classes=labels, title="ONG dataset")


def resnet50_neural_net():
    k_fold_path = 'mixed_dataset/k-fold'

    k_fold_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input) \
        .flow_from_directory(directory=k_fold_path, target_size=(224, 224), classes=labels, batch_size=170)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    k_fold_input, k_fold_labels = k_fold_batches.next()

    fold_no = 1
    history_per_fold = []
    histories = {'accuracy': [], 'f1_score': [], 'val_accuracy': [], 'val_f1_score': []}
    scores_per_fold = []
    # models_per_fold = []
    predictions = []
    correct_predictions = []

    for train, test in kf.split(k_fold_input, k_fold_labels):
        print(f'Training for fold {fold_no} ...')

        resnet50_model = tf.keras.applications.resnet50.ResNet50(include_top=False, pooling='avg')

        model = resnet50_model
        for layer in model.layers:
            layer.trainable = False

        flatten = keras.layers.Flatten()(model.output)
        output = keras.layers.Dense(3, activation='softmax')(flatten)
        model = keras.Model(model.input, output)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), F1Score()])

        history = model.fit(k_fold_input[train], k_fold_labels[train],
                            validation_data=(k_fold_input[test], k_fold_labels[test]), epochs=20, verbose=2)
        history_per_fold.append(history)
        histories['accuracy'].append(history.history['accuracy'])
        histories['f1_score'].append(np.mean(history.history['f1_score'], axis=-1))
        histories['val_accuracy'].append(history.history['val_accuracy'])
        histories['val_f1_score'].append(np.mean(history.history['val_f1_score'], axis=-1))

        model.save('models/resnet/' + str(fold_no))
        # model = keras.models.load_model('models/resnet/' + str(fold_no))
        # models_per_fold.append(model)

        scores = model.evaluate(k_fold_input[test], k_fold_labels[test], verbose=2)
        scores_per_fold.append(scores)

        predict = model.predict(x=k_fold_input[test])
        predictions.append(np.argmax(predict, axis=1))
        correct_predictions.append(np.argmax(k_fold_labels[test], axis=-1))

        fold_no = fold_no + 1

    predictions = np.concatenate(predictions)
    correct_predictions = np.concatenate(correct_predictions)
    best_fold_index = 0
    best_acc = 0.0
    best_loss = 1.0
    accs = []
    precisions = []
    recalls = []
    f1s = []
    for score in scores_per_fold:
        accs.append(score[1])
        precisions.append(score[2])
        recalls.append(score[3])
        f1s.append(np.mean(score[4]))
        if score[1] > best_acc:
            best_acc = score[1]
            best_loss = score[0]
            best_fold_index = scores_per_fold.index(score)
        elif score[1] == best_acc and score[0] < best_loss:
            best_acc = score[1]
            best_loss = score[0]
            best_fold_index = scores_per_fold.index(score)

    print('best scores in fold: ' + str(best_fold_index))
    print(scores_per_fold[best_fold_index])
    print('Accuracy: ', (np.mean(accs)))
    print('Precision: ', (np.mean(precisions)))
    print('Recall: ', (np.mean(recalls)))
    print('F1: ', (np.mean(f1s)))

    # del models_per_fold
    del k_fold_batches
    del test
    del scores
    del model
    del train
    del kf
    del f1s
    del recalls
    del precisions
    del accs

    gc.collect()

    for i in range(len(histories['accuracy'])):
        acc = histories['accuracy'][i]
        val_acc = histories['val_accuracy'][i]
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('models/resnet/resnet_kfold' + str(i) + '_acc.svg')
        plt.show()

        f1_score = histories['f1_score'][i]
        val_f1_score = histories['val_f1_score'][i]
        plt.plot(epochs, f1_score, 'y', label='Training F1 score')
        plt.plot(epochs, val_f1_score, 'r', label='Validation F1 score')
        plt.title('Training and validation F1 score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 score')
        plt.legend()
        plt.savefig('models/resnet/resnet_kfold' + str(i) + '_f1.svg')
        plt.show()

    cm_ong = confusion_matrix(y_true=correct_predictions, y_pred=predictions)
    plot_confusion_matrix(cm=cm_ong, classes=labels, title="Resnet 5-Fold",
                          save_file='models/resnet/resnet_kfold_cm.svg')


def densenet201_neural_net():
    k_fold_path = 'mixed_dataset/k-fold'

    k_fold_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input) \
        .flow_from_directory(directory=k_fold_path, target_size=(224, 224), classes=labels, batch_size=170)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    k_fold_input, k_fold_labels = k_fold_batches.next()

    fold_no = 1
    history_per_fold = []
    histories = {'accuracy': [], 'f1_score': [], 'val_accuracy': [], 'val_f1_score': []}
    scores_per_fold = []
    # models_per_fold = []
    predictions = []
    correct_predictions = []

    for train, test in kf.split(k_fold_input, k_fold_labels):
        print(f'Training for fold {fold_no} ...')

        densenet201_model = tf.keras.applications.densenet.DenseNet201(include_top=False, pooling='avg')

        model = densenet201_model
        for layer in model.layers:
            layer.trainable = False

        flatten = keras.layers.Flatten()(model.output)
        output = keras.layers.Dense(3, activation='softmax')(flatten)
        model = keras.Model(model.input, output)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), F1Score()])

        history = model.fit(k_fold_input[train], k_fold_labels[train],
                            validation_data=(k_fold_input[test], k_fold_labels[test]), epochs=20, verbose=2)
        history_per_fold.append(history)
        histories['accuracy'].append(history.history['accuracy'])
        histories['f1_score'].append(np.mean(history.history['f1_score'], axis=-1))
        histories['val_accuracy'].append(history.history['val_accuracy'])
        histories['val_f1_score'].append(np.mean(history.history['val_f1_score'], axis=-1))

        model.save('models/densenet/' + str(fold_no))
        # model = keras.models.load_model('models/densenet/' + str(fold_no))
        # models_per_fold.append(model)

        scores = model.evaluate(k_fold_input[test], k_fold_labels[test], verbose=2)
        scores_per_fold.append(scores)

        predict = model.predict(x=k_fold_input[test])
        predictions.append(np.argmax(predict, axis=1))
        correct_predictions.append(np.argmax(k_fold_labels[test], axis=-1))

        fold_no = fold_no + 1

    predictions = np.concatenate(predictions)
    correct_predictions = np.concatenate(correct_predictions)
    best_fold_index = 0
    best_acc = 0.0
    best_loss = 1.0
    accs = []
    precisions = []
    recalls = []
    f1s = []
    for score in scores_per_fold:
        accs.append(score[1])
        precisions.append(score[2])
        recalls.append(score[3])
        f1s.append(np.mean(score[4]))
        if score[1] > best_acc:
            best_acc = score[1]
            best_loss = score[0]
            best_fold_index = scores_per_fold.index(score)
        elif score[1] == best_acc and score[0] < best_loss:
            best_acc = score[1]
            best_loss = score[0]
            best_fold_index = scores_per_fold.index(score)

    print('best scores in fold: ' + str(best_fold_index))
    print(scores_per_fold[best_fold_index])
    print('Accuracy: ', (np.mean(accs)))
    print('Precision: ', (np.mean(precisions)))
    print('Recall: ', (np.mean(recalls)))
    print('F1: ', (np.mean(f1s)))

    # del models_per_fold
    del k_fold_batches
    del test
    del scores
    del model
    del train
    del kf
    del f1s
    del recalls
    del precisions
    del accs

    gc.collect()

    for i in range(len(histories['accuracy'])):
        acc = histories['accuracy'][i]
        val_acc = histories['val_accuracy'][i]
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('models/densenet/densenet_kfold' + str(i) + '_acc.svg')
        plt.show()

        f1_score = histories['f1_score'][i]
        val_f1_score = histories['val_f1_score'][i]
        plt.plot(epochs, f1_score, 'y', label='Training F1 score')
        plt.plot(epochs, val_f1_score, 'r', label='Validation F1 score')
        plt.title('Training and validation F1 score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 score')
        plt.legend()
        plt.savefig('models/densenet/densenet_kfold' + str(i) + '_f1.svg')
        plt.show()

    cm_ong = confusion_matrix(y_true=correct_predictions, y_pred=predictions)
    plot_confusion_matrix(cm=cm_ong, classes=labels, title="Densenet 5-Fold",
                          save_file='models/densenet/densenet_kfold_cm.svg')


def inception_neural_net():
    k_fold_path = 'mixed_dataset/k-fold'

    k_fold_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
        .flow_from_directory(directory=k_fold_path, target_size=(224, 224), classes=labels, batch_size=170)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    k_fold_input, k_fold_labels = k_fold_batches.next()

    fold_no = 1
    history_per_fold = []
    histories = {'accuracy': [], 'f1_score': [], 'val_accuracy': [], 'val_f1_score': []}
    scores_per_fold = []
    # models_per_fold = []
    predictions = []
    correct_predictions = []

    for train, test in kf.split(k_fold_input, k_fold_labels):
        print(f'Training for fold {fold_no} ...')

        inception_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg')

        model = inception_model
        for layer in model.layers:
            layer.trainable = False

        flatten = keras.layers.Flatten()(model.output)
        output = keras.layers.Dense(3, activation='softmax')(flatten)
        model = keras.Model(model.input, output)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), F1Score()])

        history = model.fit(k_fold_input[train], k_fold_labels[train],
                            validation_data=(k_fold_input[test], k_fold_labels[test]), epochs=20, verbose=2)
        history_per_fold.append(history)
        histories['accuracy'].append(history.history['accuracy'])
        histories['f1_score'].append(np.mean(history.history['f1_score'], axis=-1))
        histories['val_accuracy'].append(history.history['val_accuracy'])
        histories['val_f1_score'].append(np.mean(history.history['val_f1_score'], axis=-1))

        model.save('models/inception/' + str(fold_no))
        # model = keras.models.load_model('models/inception/' + str(fold_no))
        # models_per_fold.append(model)

        scores = model.evaluate(k_fold_input[test], k_fold_labels[test], verbose=2)
        scores_per_fold.append(scores)

        predict = model.predict(x=k_fold_input[test])
        predictions.append(np.argmax(predict, axis=1))
        correct_predictions.append(np.argmax(k_fold_labels[test], axis=-1))

        fold_no = fold_no + 1

    predictions = np.concatenate(predictions)
    correct_predictions = np.concatenate(correct_predictions)
    best_fold_index = 0
    best_acc = 0.0
    best_loss = 1.0
    accs = []
    precisions = []
    recalls = []
    f1s = []
    for score in scores_per_fold:
        accs.append(score[1])
        precisions.append(score[2])
        recalls.append(score[3])
        f1s.append(np.mean(score[4]))
        if score[1] > best_acc:
            best_acc = score[1]
            best_loss = score[0]
            best_fold_index = scores_per_fold.index(score)
        elif score[1] == best_acc and score[0] < best_loss:
            best_acc = score[1]
            best_loss = score[0]
            best_fold_index = scores_per_fold.index(score)

    print('best scores in fold: ' + str(best_fold_index))
    print(scores_per_fold[best_fold_index])
    print('Accuracy: ', (np.mean(accs)))
    print('Precision: ', (np.mean(precisions)))
    print('Recall: ', (np.mean(recalls)))
    print('F1: ', (np.mean(f1s)))

    # del models_per_fold
    del k_fold_batches
    del test
    del scores
    del model
    del train
    del kf
    del f1s
    del recalls
    del precisions
    del accs

    gc.collect()

    for i in range(len(histories['accuracy'])):
        acc = histories['accuracy'][i]
        val_acc = histories['val_accuracy'][i]
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('models/inception/inception_kfold' + str(i) + '_acc.svg')
        plt.show()

        f1_score = histories['f1_score'][i]
        val_f1_score = histories['val_f1_score'][i]
        plt.plot(epochs, f1_score, 'y', label='Training F1 score')
        plt.plot(epochs, val_f1_score, 'r', label='Validation F1 score')
        plt.title('Training and validation F1 score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 score')
        plt.legend()
        plt.savefig('models/inception/inception_kfold' + str(i) + '_f1.svg')
        plt.show()

    cm_ong = confusion_matrix(y_true=correct_predictions, y_pred=predictions)
    plot_confusion_matrix(cm=cm_ong, classes=labels, title="Inception 5-Fold",
                          save_file='models/inception/densenet_kfold_cm.svg')


def inception_bagging():
    k_fold_path = 'mixed_dataset/k-fold'
    k_fold_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
        .flow_from_directory(directory=k_fold_path, target_size=(224, 224), classes=labels, batch_size=170,
                             shuffle=True)

    k_fold_input, k_fold_labels = k_fold_batches.next()

    del k_fold_batches

    # multiple train-test splits
    n_splits = 10
    total_acc, total_f1, members = list(), list(), list()
    for split_index in range(n_splits):
        # select indexes
        ix = [i for i in range(len(k_fold_input) - 34)]
        train_ix = resample(ix, replace=True, n_samples=120)
        test_ix = [x for x in ix if x not in train_ix]
        # select data
        trainx, trainy = k_fold_input[train_ix], k_fold_labels[train_ix]
        testx, testy = k_fold_input[test_ix], k_fold_labels[test_ix]
        # evaluate model
        model, scores = evaluate_model(trainx, trainy, testx, testy)
        # print('acc = ' + str(scores[1]) + ' F1-score = ' + str(scores[4]))
        total_acc.append(scores[1])
        total_f1.append(np.mean(scores[4]))
        # members.append(model)
        model.save('models/inception/bagging/' + str(split_index))

    print('Total acc: ' + str(np.mean(total_acc)) + ' total F1-score = ' + str(np.mean(total_f1)))

    acc, f1 = evaluate_n_members(n_splits, k_fold_input, k_fold_labels, 'Full_dataset')
    print('ensamble acc: ' + str(acc))
    print('ensamble macro f1: ' + str(f1))

    acc, f1 = evaluate_n_members(n_splits, k_fold_input[-34:], k_fold_labels[-34:], 'Holdout')
    print('ensamble on holdout acc: ' + str(acc))
    print('ensamble on holdout macro f1: ' + str(f1))

    test_ong_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
        .flow_from_directory(directory='ong_dataset', target_size=(224, 224), classes=labels, batch_size=70,
                             shuffle=False)

    ong_x, ong_y = test_ong_batches.next()

    acc_ong, f1_ong = evaluate_n_members(n_splits, ong_x, ong_y, 'ONG_dataset')
    print('using only ong dataset')
    print('ensamble acc: ' + str(acc_ong))
    print('ensamble macro f1: ' + str(f1_ong))

    stanford_ong_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
        .flow_from_directory(directory='stanford_dataset', target_size=(224, 224), classes=labels, batch_size=110,
                             shuffle=False)

    stanford_x, stanford_y = stanford_ong_batches.next()

    acc_stanford, f1_stanford = evaluate_n_members(n_splits, stanford_x, stanford_y, 'Stanford_dataset')
    print('using only stanford dataset')
    print('ensamble acc: ' + str(acc_stanford))
    print('ensamble macro f1: ' + str(f1_stanford))


def manual_evaluation():
    k_fold_path = 'mixed_dataset/k-fold'
    k_fold_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
        .flow_from_directory(directory=k_fold_path, target_size=(224, 224), classes=labels, batch_size=170,
                             shuffle=True)

    k_fold_input, k_fold_labels = k_fold_batches.next()
    n_splits = 100

    acc, f1 = evaluate_n_members(n_splits, k_fold_input, k_fold_labels, 'Full_dataset')
    print('ensamble acc: ' + str(acc))
    print('ensamble macro f1: ' + str(f1))

    acc, f1 = evaluate_n_members(n_splits, k_fold_input[-34:], k_fold_labels[-34:], 'Holdout')
    print('ensamble on holdout acc: ' + str(acc))
    print('ensamble on holdout macro f1: ' + str(f1))

    del k_fold_batches
    del k_fold_input
    del k_fold_labels

    test_ong_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
        .flow_from_directory(directory='ong_dataset', target_size=(224, 224), classes=labels, batch_size=70,
                             shuffle=False)

    ong_x, ong_y = test_ong_batches.next()

    acc_ong, f1_ong = evaluate_n_members(n_splits, ong_x, ong_y, 'ONG_dataset')
    print('using only ong dataset')
    print('ensamble acc: ' + str(acc_ong))
    print('ensamble macro f1: ' + str(f1_ong))

    stanford_ong_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
        .flow_from_directory(directory='stanford_dataset', target_size=(224, 224), classes=labels, batch_size=110,
                             shuffle=False)

    stanford_x, stanford_y = stanford_ong_batches.next()

    acc_stanford, f1_stanford = evaluate_n_members(n_splits, stanford_x, stanford_y, 'Stanford_dataset')
    print('using only stanford dataset')
    print('ensamble acc: ' + str(acc_stanford))
    print('ensamble macro f1: ' + str(f1_stanford))


# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy):
    # encode targets
    inception_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg')

    model = inception_model
    for layer in model.layers:
        layer.trainable = False

    flatten = keras.layers.Flatten()(model.output)
    output = keras.layers.Dense(3, activation='softmax')(flatten)
    model = keras.Model(model.input, output)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall(), F1Score()])

    # fit model
    model.fit(trainX, trainy, epochs=20, verbose=2)
    # evaluate the model
    scores = model.evaluate(testX, testy, verbose=2)
    return model, scores


# make an ensemble prediction for multi-class classification
def ensemble_predictions(n_members, testX):
    yhats = []
    for i in range(n_members):
        model = keras.models.load_model('models/inception/bagging/' + str(i))
        # make predictions
        predicted = model.predict(testX)
        yhats.append(predicted)

    yhats = np.array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)
    return result


# evaluate a specific number of members in an ensemble
def evaluate_n_members(n_members, testX, testy, title=None):
    # make prediction
    yhat = ensemble_predictions(n_members, testX)
    # calculate accuracy
    if title is not None:
        cm_ong = confusion_matrix(y_true=np.argmax(testy, axis=1), y_pred=yhat)
        plot_confusion_matrix(cm=cm_ong, classes=labels, title=title,
                              save_file='models/inception/bagging/cm/' + title + str(n_members) + '.png')
    return accuracy_score(np.argmax(testy, axis=1), yhat), f1_score(np.argmax(testy, axis=1), yhat, average="macro")


def print_preprocess_images():
    k_fold_path = 'stanford_dataset'
    k_fold_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input) \
        .flow_from_directory(directory=k_fold_path, target_size=(512, 512), classes=labels, batch_size=170,
                             shuffle=True)

    k_fold_input, k_fold_labels = k_fold_batches.next()

    plt.imshow(np.clip(k_fold_input[0] / np.amax(k_fold_input[0]), 0, 1))
    plt.savefig('pre-process/0.png')
    plt.imshow(np.clip(k_fold_input[1] / np.amax(k_fold_input[1]), 0, 1))
    plt.savefig('pre-process/1.png')
    plt.imshow(np.clip(k_fold_input[2] / np.amax(k_fold_input[2]), 0, 1))
    plt.savefig('pre-process/2.png')
    plt.imshow(np.clip(k_fold_input[3] / np.amax(k_fold_input[3]), 0, 1))
    plt.savefig('pre-process/3.png')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # pre_built_dogs_cats()
    # custom_neural_net()
    # vgg16_neural_net()
    # resnet50_neural_net()
    # densenet201_neural_net()
    # inception_neural_net()
    inception_bagging()
    # predict_and_plot('models/vgg16/1')
    # manual_evaluation()
    # print_preprocess_images()
