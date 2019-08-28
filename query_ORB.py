import cv2
import os
import sys
import numpy as np
import faiss
from FeatureExtractor import SIFT, SIFT_Real_Kmeans, ORB
from matplotlib import pyplot as plt
import json
import pickle

# queryDataPath = r'/home/yehai/Work/MyCBIR/queryImage/dog.jpeg'
# queryDataPath = r'/home/yehai/Work/MyCBIR/101_ObjectCategories/grand_piano/image_0067.jpg'
# queryDataPath = r'/home/yehai/Work/MyCBIR/101_ObjectCategories/ewer/image_0015.jpg'
# queryDataPath = r'/home/yehai/Work/MyCBIR/101_ObjectCategories/bonsai/image_0005.jpg'
# queryDataPath = '/home/yehai/Work/MyCBIR/101_ObjectCategories/Faces_easy/image_0026.jpg'
# queryDataPath = r'/home/yehai/Work/MyCBIR/101_ObjectCategories/ceiling_fan/image_0009.jpg'
queryDataPath = r'/home/yehai/Work/MyCBIR/101_ObjectCategories/lotus/image_0012.jpg'
datapath = r'./101_ObjectCategories'
# datapath = r'./database'
IndexSavePath = r'./ORBFile/index_ORBFile'
I2PJsonPath = r'./ORBFile/Image2Path_ORB.json'
KmeansPath = r'./ORBFile/Kmeans.pkl'

queryTopK = 3


def init_sift_features(imagepath):
    '''
    默认保存特征文件到./featuresSITF.npy
    id到图像路径字典到./Image2Path_SIFT.json
    :param imagepath: 初始化图像数据库的路径
    :return:
    '''
    # sift = SIFT()
    orb = ORB()
    if not os.path.exists(KmeansPath):
        # orb.getClusterCentresRealKmeans(imagepath)
        orb.getClusterCentresMiniBatchKmeans(imagepath)
        orb.saveKmeans(KmeansPath)
    else:
        input = open(KmeansPath, "rb")
        kmeans = pickle.load(input)
        orb.kmeans = kmeans
        input.close()

    features_list, id_path_dict = orb.descriptors2features(imagepath)
    orb.saveI2PJson(id_path_dict)
    features_np = np.array(features_list)
    faiss_Ids = np.array(list(id_path_dict.keys()))
    # print("total Bow SIFT features:" + str(features_np.shape))
    # print("total Bow features space:" + str(features_np.nbytes))
    print("feature shape:", features_np.shape)
    print("ids shape", faiss_Ids.shape)
    return features_np, faiss_Ids


def CreateIndex(features, faiss_Ids):
    d = features.shape[1]
    # print(features.shape[1])
    index_base = faiss.IndexFlatL2(d)
    index_IdMap = faiss.IndexIDMap(index_base)
    # print(index_model.is_trained)
    print('\n\n***********************************')
    print("adding to faiss index.....")
    index_IdMap.add_with_ids(features, faiss_Ids)
    print("total image:" + str(index_IdMap.ntotal))
    print('***********************************\n\n')

    faiss.write_index(index_IdMap, IndexSavePath)


def readJsonFile(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    is_Inited = False
    if os.path.exists(IndexSavePath):
        is_Inited = True
    if not is_Inited:
        features , faiss_Ids= init_sift_features(datapath)
        CreateIndex(features.astype('float32'), faiss_Ids)
    # Kmeans
    input = open(KmeansPath, "rb")
    kmeans = pickle.load(input)
    input.close()
    # index
    index = faiss.read_index(IndexSavePath)

    # Id2Path
    I2P = readJsonFile(I2PJsonPath)

    # sift class
    orb = ORB()
    orb.kmeans = kmeans

    # extract feature
    img = cv2.imread(queryDataPath, 0)
    kps, descritors = orb.orb_detector.detectAndCompute(img, None)
    feature_vlad = orb.Vlad(descritors)

    feature_vlad = feature_vlad.astype('float32')
    if len(feature_vlad.shape) == 1:
        feature_vlad = np.expand_dims(feature_vlad, axis=0)
    print("feature_vlad:", feature_vlad.shape)

    # Faiss Search
    Dist, ID = index.search(feature_vlad, queryTopK)

    resultSortList = ID[0]
    # print(I2P)
    for i in range(1):  # 查询图片数量，暂定1
        fig = plt.figure()
        plt.subplot(221)
        plt.title("queryImage")
        img = plt.imread(queryDataPath)
        plt.imshow(img)
        for j in range(queryTopK):
            plt.subplot(2, 2, j + 2)
            plt.title("Top:" + str(j + 1) + " score:" + str(round(Dist[0][j], 3)))
            imgPath = I2P[str(resultSortList[j])]
            img = plt.imread(imgPath)
            plt.imshow(img)
        plt.show()
