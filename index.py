from FeatureExtractor import VGGNet
import argparse
import h5py
import numpy as np
import os
import faiss
import json
imgPath = r"/home/yehai/Work/MyCBIR/dog.jpeg"
dataPath = r"/home/yehai/Work/MyCBIR/101_ObjectCategories"
saveName = r"/home/yehai/Work/MyCBIR/featureCNN.hdf5"
SavePath = "./index"


# vgg = VGGNet()
# vgg.extractFeature(imgPath=imgPath)

# parse = argparse.ArgumentParser()
# parse.add_argument('-d',"-datapath",required=True,help="Path of data")
# parse.add_argument('-out',default=r"./index.hdf5",help="save name of index")
# args = vars(parse)


# 构建特征矩阵(m,n) m张n维向量，n=512目前
def GetFeatureAndPaths():
    imgPathList = [os.path.join(root, file) for root, dirs, files in os.walk(dataPath) for file in files]
    features = []
    paths = []
    vgg = VGGNet()
    for i, path in enumerate(imgPathList):
        feature = vgg.extractFeature(path)
        path = path.split(dataPath)[1]
        features.append(feature)
        paths.append(path)
        print("extracting feature:" + str(i + 1))
    features = np.array(features, dtype=np.float32)
    paths = np.array(paths)
    return features, paths


def CreateDict(relativePaths):
    # 构建image_id<->path的字典并保存json
    dict_path2ImageID = {}
    dict_ImageID2Path = {}
    for i, relativePath in enumerate(relativePaths):
        dict_path2ImageID[relativePath] = i
        dict_ImageID2Path[i] = relativePath
    # P2I = json.dumps(dict_path2ImageID)
    # I2P = json.dumps(dict_ImageID2Path)
    # print(dict_ImageID2Path)
    json.dump(dict_path2ImageID,open(r'./P2I.json', 'w'))
    json.dump(dict_ImageID2Path, open(r'./I2P.json', 'w'))


def CreateIndex(features):
    d = features.shape[1]
    # print(features.shape[1])
    index_model = faiss.IndexFlatIP(d)
    # print(index_model.is_trained)

    index_model.add(features)
    print(index_model.ntotal)
    faiss.write_index(index_model, SavePath)


# hdf5
# file = h5py.File(saveName, 'w')
# file.create_dataset("features", data=features)
# file.create_dataset("paths", data=relativePaths)
# file.close()


# print(features.shape,features.dtype)
# print(relativePaths.shape,relativePaths.dtype)

if __name__ == '__main__':
    features, paths = GetFeatureAndPaths()
    CreateDict(paths)
    CreateIndex(features)

