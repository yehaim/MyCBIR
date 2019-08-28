import h5py
import numpy as np
from PIL import Image
from FeatureExtractor import VGGNet
from matplotlib import pyplot as plt
import json
import faiss

IndexFilePath = r"./featureCNN.hdf5"
# queryDataPath = r'/home/yehai/Work/MyCBIR/queryImage/dog.jpeg'
# queryDataPath = r'/home/yehai/Work/MyCBIR/101_ObjectCategories/grand_piano/image_0067.jpg'
# queryDataPath = r'/home/yehai/Work/MyCBIR/101_ObjectCategories/ewer/image_0015.jpg'
# queryDataPath = r'/home/yehai/Work/MyCBIR/101_ObjectCategories/bonsai/image_0005.jpg'
# queryDataPath = '/home/yehai/Work/MyCBIR/101_ObjectCategories/Faces_easy/image_0026.jpg'
# queryDataPath = r'/home/yehai/Work/MyCBIR/101_ObjectCategories/ceiling_fan/image_0009.jpg'
queryDataPath = r'/home/yehai/Work/MyCBIR/101_ObjectCategories/lotus/image_0012.jpg'
databasePath = r'./101_ObjectCategories'
faissIndexPath = r'./index'
I2PjsonFile = r'./I2P.json'
queryDataNum = 1
queryTopK = 3


# originalDataPathList = [queryDataPath]

def readHDF5File(index_path):
    file = h5py.File(index_path, 'r')
    features = file['features']
    paths = file['paths']
    features = np.array(features)
    paths = np.array(paths)
    file.close()
    return features, paths


def readJsonFile(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def readIndex(index_path):
    return faiss.read_index(index_path)


if __name__ == "__main__":
    # features, paths = readHDF5File(IndexFilePath)
    vgg = VGGNet()
    queryDataFeature = np.expand_dims(vgg.extractFeature(queryDataPath), 0)
    print(queryDataFeature.shape)
    index_model = readIndex(faissIndexPath)
    Dist, ID = index_model.search(queryDataFeature, queryTopK)
    # print(Dist)
    # print(ID)
    # resultMatrix = np.dot(queryDataFeature, np.transpose(features))
    # print(resultMatrix)
    # resultSortList = np.argsort(-resultMatrix, axis=-1)  # 排名后的索引
    I2P = readJsonFile(I2PjsonFile)
    resultSortList = ID[0]
    print(I2P)
    for i in range(1):  # 查询图片数量，暂定1
        fig = plt.figure()
        plt.subplot(221)
        plt.title("queryImage")
        img = plt.imread(queryDataPath)
        plt.imshow(img)
        for j in range(queryTopK):
            plt.subplot(2, 2, j + 2)
            plt.title("Top:" + str(j + 1) + " score:" + str(round(Dist[0][j],3)))
            imgPath = I2P[str(resultSortList[j])]
            img = plt.imread(databasePath + imgPath)
            plt.imshow(img)
        plt.show()
