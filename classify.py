from load_data_from_csv import load_data_from_csv
from load_data_from_mongoDb import load_data_from_mongoDb
from load_data_from_rawData_Img import imageLoader
from load_data_from_rawData_txt import sentenceLoader
from model import GaussianMixtureClassifierModel
import numpy as np
from os import listdir
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from .model import agregation_of_heterogenous_datas, reduction_of_dimension_with_PCA, reduction_of_dimension_with_LDA

# define image extractor function
def image__features_extractor(directory):
    for img in listdir(directory):
        print(img)
        filename = directory + '/' + img
        image = load_img(filename, target_size=(28, 28))
        image = img_to_array(image)
    return image

def classifyOne():
    # define categorical label
    label_dict = {
        0: 'T-Shirt/top',
        1: 'Trouser/pants',
        2: 'Pullover shirt',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot',
    }

    #load heterogenous data
    Xtrain1, Xtest1 = imageLoader()
    Ytrain1, Ytest1 = sentenceLoader()
    Xtest2, Ytest2, Xtrain2, Ytrain2 = load_data_from_csv()
    Xtest3, Ytest3,Xtrain3, Ytrain3 = load_data_from_mongoDb()

    #agregate data with numpy
    Xtrain, Ytrain, Xtest, Ytest = agregation_of_heterogenous_datas(Xtrain1, Ytrain1, Xtrain2, Ytrain2, Xtrain3, Ytrain3, Xtest1, Ytest1, Xtest2, Ytest2, Xtest3, Ytest3)
    #reduce dimension of agregated data with PCA
    Xtrain, Xtest = reduction_of_dimension_with_PCA(Xtrain, Xtest)
    # reduce dimension of agregated data with LDA
    #Xtrain, Xtest = reduction_of_dimension_with_LDA(Xtrain, Xtest, Ytrain)

    # call model
    GMixtureClassifier = GaussianMixtureClassifierModel()
    # fit data into LRclassifier
    GMixtureClassifier.fit(Xtrain, Ytrain)
    # evaluate
    print('test score : {} %'.format(GMixtureClassifier.score(Xtest, Ytest) * 100))
    predicted = GMixtureClassifier.predict(Xtest)
    print('accuracy : {}%'.format(str(accuracy_score(Ytest, predicted)*100)))
    print(classification_report(Ytest, predicted))
    print(confusion_matrix(Ytest, predicted))
    # rescale and reshape image
    input_image = 'image_to_recognize'
    image = image__features_extractor(input_image)

    image = np.resize(image, (1, 784))
    image = np.array(image).reshape((1, -1))
    pca = PCA()
    image = pca.transform(image)
    print('image shape : ', image.shape)
    # prediction
    predicted = GMixtureClassifier.predict(image)

    image2 = image__features_extractor(input_image)
    image2 = np.resize(image2, (1, 784))
    image2 = np.array(image2).reshape((1, -1))
    plt.figure(figsize=[5, 5])
    plt.subplot(121)
    plt.title("Predicted : {}".format(predicted))
    print('predicted : ', predicted)
    plt.imshow(np.reshape(image2, (28, 28)), cmap='gray_r')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    classifyOne()

"""
def classify():
    df_test_x, df_test_y = load_dataMongo()
    df_train_x, df_train_y = load_dataCsv()
    #split test data and train data into Xtrain, Ytrain, Xtest, Ytest
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_train_x, df_train_y, random_state=0, test_size=0.2)
    Xtrain1, Xtest1, Ytrain1, Ytest1 = train_test_split(df_test_x, df_test_y, random_state=0, test_size=0.2)
    print(' Xtrain shape : {} - Ytrain shape : {}'.format(Xtrain.shape, Ytrain.shape))
    print(' Xtest shape : {} - Ytest shape : {}'.format(Xtest.shape, Ytest.shape))
    print(' Xtrain1 shape : {} - Ytrain1 shape : {}'.format(Xtrain1.shape, Ytrain1.shape))
    print(' Xtest1 shape : {} - Ytest1 shape : {}'.format(Xtest1.shape, Ytest1.shape))

    #call model
    GMixtureClassifier = GaussianMixtureClassifierModel()
    #fit data into GMixtureClassifier
    #XtrainA = np.concatenate(Xtrain, Xtrain1) YtrainA = np.concatenate(Ytrain, Ytrain1)
    GMixtureClassifier.fit(Xtrain, Ytrain)

    #predict result
    predicted = GMixtureClassifier.predict(Xtest1)
    print('predicted : ', [int(predict) for predict in predicted])
    print('Ytest1 : ', Ytest1)
    print('mean error : ', np.mean(predicted - Ytest1)**2)
    #accuracy
    print(classification_report(Ytest1, [int(predict) for predict in predicted]))
    print(confusion_matrix(Ytest1, [int(predict) for predict in predicted]))

    plt.figure(figsize=(20, 5))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.title("Predicted : {}".format(predicted[i]))
        plt.imshow(np.reshape(Xtest1[i], (28, 28)), cmap='gray_r')
        plt.tight_layout()
    plt.show()
"""