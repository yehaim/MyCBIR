import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import cv2
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import pickle
import json
import time
import sys


class VGGNet():
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(
            include_top=False,
            weights=self.weight,
            input_shape=self.input_shape,
            pooling=self.pooling,
        )

    def extractFeature(self, imgPath):
        img = image.load_img(imgPath, target_size=(self.input_shape[0], self.input_shape[1]))
        img = np.asarray(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        feature = self.model.predict(img)
        feature = feature[0] / np.linalg.norm(feature[0])  # [1,512],取feature[0],除以他的范数（即长度）化为单位向量方便后面直接求余弦相似度
        return feature


class SIFT():
    def __init__(self):
        self.sift_detector = cv2.xfeatures2d.SIFT_create()
        self.num_words = 64
        self.SIFT_dimention = 64
        self.random_state = 52
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_words, random_state=self.random_state, batch_size=500)
        if self.SIFT_dimention < 128:
            self.pca = PCA(n_components=self.SIFT_dimention, random_state=self.random_state)

    def getClusterCentres(self, imgRootPaths):
        '''

        :param imgRootPaths: image data root path
        :return: path_descriptors_list: list of image path and descriptors (image_num, ) each element contains a tuple consists of (image_path, descriptors)
        '''
        timestart = time.time()
        print('***********************************')
        print('getClusterCentres beginning..')
        print('***********************************\n\n')

        imgPaths = [os.path.join(root, file) for root, dirs, files in os.walk(imgRootPaths) for file in files if
                    (file.split('/')[-1]).split('.')[-1] in ['jpg', 'jpeg', 'png']]
        # extract features
        # descriptors_np = np.zeros((1, 128), dtype='float32')
        features = []
        for i, imgpath in enumerate(imgPaths):
            # try:
            if i % 100 == 0:
                print("extracting SIFT feature number:", i)
            img = cv2.imread(imgpath, 0)
            kps, descritors = self.sift_detector.detectAndCompute(img, None)
            if descritors.shape[0] < self.SIFT_dimention:
                continue
            if self.SIFT_dimention < 128:
                descritors = self.pca.fit_transform(descritors)
            # kmeans cluster
            self.kmeans.partial_fit(descritors)
            # except Exception as e:
            #     print(e)
            #     print(i, imgpath)

        print('\n\n***********************************')
        print('getClusterCentres end..using {} s'.format(time.time() - timestart))
        print('***********************************\n\n')

    def descriptors2features(self, imgRootPaths):
        '''

        :param path_descriptors_list:
        :return: features_norm, id_path_dict
        '''
        timestart = time.time()
        print('\n\n***********************************')
        print('Descriptors to features beginning..')
        print('***********************************\n\n')

        imgPaths = [os.path.join(root, file) for root, dirs, files in os.walk(imgRootPaths) for file in files if
                    (file.split('/')[-1]).split('.')[-1] in ['jpg', 'jpeg', 'png']]
        # extract features
        # descriptors_np = np.zeros((1, 128), dtype='float32')
        id_path_dict = {}
        features_list = []
        Skip_counter = 0  # 因为sift特征数小于PCA需要的维度数而省略的图片总数
        for i, imgpath in enumerate(imgPaths):
            if i % 100 == 0:
                print('Image {} is converting to feature'.format(i))
            img = cv2.imread(imgpath, 0)
            kps, descritors = self.sift_detector.detectAndCompute(img, None)
            if descritors.shape[0] < self.SIFT_dimention:
                Skip_counter += 1
                continue

            # 开始id和path字典放在continue前导致出现continue的情况下id和path也录入了
            id_path_dict[i] = imgpath
            # VLAD
            feature_vlad = self.Vlad(descritors)
            # print("i:", i, "imgpath:", imgpath, "feature_vlad:", feature_vlad)
            features_list.append(feature_vlad.tolist())
        # L2归一化
        # features_norm = features / np.linalg.norm(features, ord=2, axis=1, keepdims=True)

        print('\n\n***********************************')
        print('Descriptors to features end..using {} s'.format(time.time() - timestart))
        print('Skip_counter:', Skip_counter)
        print('***********************************\n\n')
        return features_list, id_path_dict

    def Vlad(self, descritors):
        if self.SIFT_dimention < 128:
            descritors = self.pca.fit_transform(descritors)
        pred = self.kmeans.predict(descritors)
        Vlad = np.zeros((self.num_words, self.SIFT_dimention))
        for j, index in enumerate(pred):
            # print(self.kmeans.cluster_centers_.shape)
            # print(descritors.shape)
            Vlad[index] += descritors[j] - self.kmeans.cluster_centers_[index]

        # 应该先做norm再flatten,开始反了结果不理想(后改进成论文中描述，先flatten，再
        # 做SSR(signed square rooting)norm,最后L2)
        norm = np.linalg.norm(Vlad, axis=1).reshape((-1, 1))
        # 可能遇上0/0的情况
        norm[norm < 1e-12] = 1e-12
        Vlad = Vlad / norm
        Vlad = Vlad.flatten()
        # SSR Normlization
        Vlad = np.sign(Vlad) * np.sqrt(np.abs(Vlad))
        # L2 normlization
        Vlad = Vlad / np.linalg.norm(Vlad)
        return Vlad

    def saveKmeans(self, savePath='./SIFTFile/Kmeans.pkl'):
        output = open(savePath, 'wb')
        pickle.dump(self.kmeans, output)
        output.close()

    def saveI2PJson(self, id_path_dict, savePath='./SIFTFile/Image2Path_SIFT.json'):
        json.dump(id_path_dict, open(savePath, 'w'))


class SIFT_Real_Kmeans():
    def __init__(self):
        self.sift_detector = cv2.xfeatures2d.SIFT_create()
        self.num_words = 16
        self.SIFT_dimention = 64
        self.random_state = 52
        self.kmeans = KMeans(n_clusters=self.num_words, random_state=self.random_state)
        if self.SIFT_dimention < 128:
            self.pca = PCA(n_components=self.SIFT_dimention, random_state=self.random_state)

    def getClusterCentres(self, imgRootPaths):
        '''

        :param imgRootPaths: image data root path
        :return: path_descriptors_list: list of image path and descriptors (image_num, ) each element contains a tuple consists of (image_path, descriptors)
        '''
        timestart = time.time()
        print('***********************************')
        print('getClusterCentres beginning..')
        print('***********************************\n\n')
        feature = []
        dirList = os.listdir(imgRootPaths)
        counter = 0
        for dir in dirList:
            counter += 1
            print(counter)
            dirPath = os.path.join(imgRootPaths, dir)
            imgList = os.listdir(dirPath)
            random = np.random.randint(1, 3, len(imgList))
            for i, imgpath in enumerate(imgList):
                # try:
                # 随机抽取
                if not random[i] == 1:
                    continue
                imgpath = os.path.join(dirPath, imgpath)
                img = cv2.imread(imgpath, 0)
                kps, descritors = self.sift_detector.detectAndCompute(img, None)
                # if descritors.shape[0] < self.SIFT_dimention:
                #     continue
                feature.append(descritors)
        features = np.vstack(feature)
        print(features.shape)
        if self.SIFT_dimention < 128:
            features = self.pca.fit_transform(features)
            print("pca ending")
            print(features.shape)
        # print("features shape:", feature_pca.shape)
        self.kmeans.fit(features)
        print(self.kmeans.cluster_centers_)
        # kmeans cluster

        # except Exception as e:
        #     print(e)
        #     print(i, imgpath)

        print('\n\n***********************************')
        print('getClusterCentres end..using {} s'.format(time.time() - timestart))
        print('***********************************\n\n')

    def descriptors2features(self, imgRootPaths):
        '''

        :param path_descriptors_list:
        :return: features_norm, id_path_dict
        '''
        timestart = time.time()
        print('\n\n***********************************')
        print('Descriptors to features beginning..')
        print('***********************************\n\n')

        imgPaths = [os.path.join(root, file) for root, dirs, files in os.walk(imgRootPaths) for file in files if
                    (file.split('/')[-1]).split('.')[-1] in ['jpg', 'jpeg', 'png']]
        # extract features
        # descriptors_np = np.zeros((1, 128), dtype='float32')
        id_path_dict = {}
        features_list = []
        Skip_counter = 0  # 因为sift特征数小于PCA需要的维度数而省略的图片总数
        for i, imgpath in enumerate(imgPaths):
            if i % 100 == 0:
                print('Image {} is converting to feature'.format(i))
            img = cv2.imread(imgpath, 0)
            kps, descritors = self.sift_detector.detectAndCompute(img, None)
            if descritors.shape[0] < self.SIFT_dimention:
                Skip_counter += 1
                continue

            # 开始id和path字典放在continue前导致出现continue的情况下id和path也录入了
            id_path_dict[i] = imgpath
            # VLAD
            feature_vlad = self.Vlad(descritors)
            # print("i:", i, "imgpath:", imgpath, "feature_vlad:", feature_vlad)
            features_list.append(feature_vlad.tolist())
        # L2归一化
        # features_norm = features / np.linalg.norm(features, ord=2, axis=1, keepdims=True)

        print('\n\n***********************************')
        print('Descriptors to features end..using {} s'.format(time.time() - timestart))
        print('Skip_counter:', Skip_counter)
        print('***********************************\n\n')
        return features_list, id_path_dict

    def Vlad(self, descritors):
        if self.SIFT_dimention < 128:
            descritors = self.pca.transform(descritors)
        pred = self.kmeans.predict(descritors)
        Vlad = np.zeros((self.num_words, self.SIFT_dimention))
        for j, index in enumerate(pred):
            # print(self.kmeans.cluster_centers_.shape)
            # print(descritors.shape)
            Vlad[index] += descritors[j] - self.kmeans.cluster_centers_[index]

        # 应该先做norm再flatten,开始反了结果不理想(后改进成论文中描述，先flatten，再
        # 做SSR(signed square rooting)norm,最后L2)
        norm = np.linalg.norm(Vlad, axis=1).reshape((-1, 1))
        # 可能遇上0/0的情况
        norm[norm < 1e-12] = 1e-12
        Vlad = Vlad / norm
        Vlad = Vlad.flatten()
        # SSR Normlization
        Vlad = np.sign(Vlad) * np.sqrt(np.abs(Vlad))
        # L2 normlization
        Vlad = Vlad / np.linalg.norm(Vlad)
        return Vlad

    def saveKmeans(self, savePath='./SIFTFile/Kmeans.pkl'):
        output = open(savePath, 'wb')
        pickle.dump(self.kmeans, output)
        output.close()

    def saveI2PJson(self, id_path_dict, savePath='./SIFTFile/Image2Path_SIFT.json'):
        json.dump(id_path_dict, open(savePath, 'w'))

    def savePCA(self, savePath='./SIFTFile/pca.pkl'):
        output = open(savePath, 'wb')
        pickle.dump(self.pca, output)
        output.close()


class SIFT_Real_PCA():
    def __init__(self):
        self.sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures=500)
        self.num_words = 256
        self.SIFT_dimention = 32
        self.random_state = 52
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_words, random_state=self.random_state, batch_size=500)
        if self.SIFT_dimention < 128:
            self.pca = PCA(n_components=self.SIFT_dimention, whiten=True, random_state=self.random_state)

    def do_PCA(self, imgRootPaths):

        imgPaths = [os.path.join(root, file) for root, dirs, files in os.walk(imgRootPaths) for file in files if
                    (file.split('/')[-1]).split('.')[-1] in ['jpg', 'jpeg', 'png']]
        # extract features
        # descriptors_np = np.zeros((1, 128), dtype='float32')
        features = []
        for i, imgpath in enumerate(imgPaths):
            # try:
            if i % 100 == 0:
                print("extracting SIFT feature number for PCA:", i)
            img = cv2.imread(imgpath, 0)
            kps, descritors = self.sift_detector.detectAndCompute(img, None)
            features.append(descritors)
            # kmeans cluster
            # except Exception as e:
            #     print(e)
            #     print(i, imgpath)
        features = np.vstack(features)
        print(features.shape)
        if self.SIFT_dimention < 128:
            timestart = time.time()
            features = self.pca.fit_transform(features)
            print("pca ending, using:", time.time() - timestart)
            print(features.shape)
            print("explained_variance_ratio_:", self.pca.explained_variance_ratio_)

    def getClusterCentres(self, imgRootPaths):
        '''

        :param imgRootPaths: image data root path
        :return: path_descriptors_list: list of image path and descriptors (image_num, ) each element contains a tuple consists of (image_path, descriptors)
        '''
        timestart = time.time()
        print('***********************************')
        print('getClusterCentres beginning..')
        print('***********************************\n\n')

        imgPaths = [os.path.join(root, file) for root, dirs, files in os.walk(imgRootPaths) for file in files if
                    (file.split('/')[-1]).split('.')[-1] in ['jpg', 'jpeg', 'png']]
        # extract features
        # descriptors_np = np.zeros((1, 128), dtype='float32')
        for i, imgpath in enumerate(imgPaths):
            # try:
            if i % 100 == 0:
                print("extracting SIFT feature number for Kmeans:", i)
            img = cv2.imread(imgpath, 0)
            kps, descritors = self.sift_detector.detectAndCompute(img, None)
            if descritors.shape[0] < self.SIFT_dimention:
                continue
            descritors_pca = self.pca.transform(descritors)
            self.kmeans.partial_fit(descritors_pca)
            # kmeans cluster
            # except Exception as e:
            #     print(e)
            #     print(i, imgpath)
        print("kmeans end")

        print('\n\n***********************************')
        print('getClusterCentres end..using {} s'.format(time.time() - timestart))
        print('***********************************\n\n')

    def descriptors2features(self, imgRootPaths):
        '''

        :param path_descriptors_list:
        :return: features_norm, id_path_dict
        '''
        timestart = time.time()
        print('\n\n***********************************')
        print('Descriptors to features beginning..')
        print('***********************************\n\n')

        imgPaths = [os.path.join(root, file) for root, dirs, files in os.walk(imgRootPaths) for file in files if
                    (file.split('/')[-1]).split('.')[-1] in ['jpg', 'jpeg', 'png']]
        # extract features
        # descriptors_np = np.zeros((1, 128), dtype='float32')
        id_path_dict = {}
        features_list = []
        Skip_counter = 0  # 因为sift特征数小于PCA需要的维度数而省略的图片总数
        for i, imgpath in enumerate(imgPaths):
            if i % 100 == 0:
                print('Image {} is converting to feature'.format(i))
            img = cv2.imread(imgpath, 0)
            kps, descritors = self.sift_detector.detectAndCompute(img, None)
            if descritors.shape[0] < self.SIFT_dimention:
                Skip_counter += 1
                continue

            # 开始id和path字典放在continue前导致出现continue的情况下id和path也录入了
            id_path_dict[i] = imgpath
            # VLAD
            feature_vlad = self.Vlad(descritors)
            # print("i:", i, "imgpath:", imgpath, "feature_vlad:", feature_vlad)
            features_list.append(feature_vlad.tolist())
        # L2归一化
        # features_norm = features / np.linalg.norm(features, ord=2, axis=1, keepdims=True)

        print('\n\n***********************************')
        print('Descriptors to features end..using {} s'.format(time.time() - timestart))
        print('Skip_counter:', Skip_counter)
        print('***********************************\n\n')
        return features_list, id_path_dict

    def Vlad(self, descritors):
        if self.SIFT_dimention < 128:
            descritors = self.pca.transform(descritors)
        pred = self.kmeans.predict(descritors)
        Vlad = np.zeros((self.num_words, self.SIFT_dimention))
        for j, index in enumerate(pred):
            # print(self.kmeans.cluster_centers_.shape)
            # print(descritors.shape)
            Vlad[index] += descritors[j] - self.kmeans.cluster_centers_[index]

        # 应该先做norm再flatten,开始反了结果不理想(后改进成论文中描述，先每行L2后flatten，再
        # 做SSR(signed square rooting)norm,最后L2)
        norm = np.linalg.norm(Vlad, axis=1).reshape((-1, 1))
        # 可能遇上0/0的情况
        norm[norm < 1e-12] = 1e-12
        Vlad = Vlad / norm
        Vlad = Vlad.flatten()
        # SSR Normlization
        Vlad = np.sign(Vlad) * np.sqrt(np.abs(Vlad))
        # L2 normlization
        Vlad = Vlad / np.linalg.norm(Vlad)
        return Vlad

    def saveKmeans(self, savePath='./SIFTFile/Kmeans.pkl'):
        output = open(savePath, 'wb')
        pickle.dump(self.kmeans, output)
        output.close()

    def saveI2PJson(self, id_path_dict, savePath='./SIFTFile/Image2Path_SIFT.json'):
        json.dump(id_path_dict, open(savePath, 'w'))

    def savePCA(self, savePath='./SIFTFile/pca.pkl'):
        output = open(savePath, 'wb')
        pickle.dump(self.pca, output)
        output.close()


class ORB():
    def __init__(self):
        self.orb_detector = cv2.ORB_create()
        self.num_words = 128
        self.orb_dimention = 32
        self.random_state = 52
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_words, random_state=self.random_state, batch_size=500)
        # self.kmeans = KMeans(n_clusters=self.num_words, random_state=self.random_state)

    def getClusterCentresMiniBatchKmeans(self, imgRootPaths):
        '''

        :param imgRootPaths: image data root path
        :return: path_descriptors_list: list of image path and descriptors (image_num, ) each element contains a tuple consists of (image_path, descriptors)
        '''
        timestart = time.time()
        print('***********************************')
        print('getClusterCentres beginning..')
        print('***********************************\n\n')

        imgPaths = [os.path.join(root, file) for root, dirs, files in os.walk(imgRootPaths) for file in files if
                    (file.split('/')[-1]).split('.')[-1] in ['jpg', 'jpeg', 'png']]
        # extract features
        # descriptors_np = np.zeros((1, 128), dtype='float32')
        for i, imgpath in enumerate(imgPaths):
            # try:
            if i % 100 == 0:
                print("extracting SIFT feature number for Kmeans:", i)
            img = cv2.imread(imgpath, 0)
            kps, descritors = self.orb_detector.detectAndCompute(img, None)
            if len(kps) < 1:
                continue
            if descritors.shape[0] < self.orb_dimention:
                continue

            self.kmeans.partial_fit(descritors)
            # kmeans cluster
            # except Exception as e:
            #     print(e)
            #     print(i, imgpath)
        print("kmeans end")

        print('\n\n***********************************')
        print('getClusterCentres end..using {} s'.format(time.time() - timestart))
        print('***********************************\n\n')

    def getClusterCentresRealKmeans(self, imgRootPaths):
        '''

        :param imgRootPaths: image data root path
        :return: path_descriptors_list: list of image path and descriptors (image_num, ) each element contains a tuple consists of (image_path, descriptors)
        '''
        timestart = time.time()
        print('***********************************')
        print('getClusterCentres beginning..')
        print('***********************************\n\n')
        feature = []
        dirList = os.listdir(imgRootPaths)
        counter = 0
        for dir in dirList:
            counter += 1
            print(counter)
            dirPath = os.path.join(imgRootPaths, dir)
            imgList = os.listdir(dirPath)
            random = np.random.randint(1, 11, len(imgList))
            for i, imgpath in enumerate(imgList):
                # try:
                # 随机抽取
                if not random[i] == 1:
                    continue
                imgpath = os.path.join(dirPath, imgpath)
                img = cv2.imread(imgpath, 0)
                kps, descritors = self.orb_detector.detectAndCompute(img, None)
                if len(kps) < 1:
                    continue
                if descritors.shape[0] < self.orb_dimention:
                    continue
                feature.append(descritors)

        features = np.vstack(feature)
        print(features.shape)
        # print("features shape:", feature_pca.shape)
        self.kmeans.fit(features)
        print(self.kmeans.cluster_centers_)
        # kmeans cluster

        # except Exception as e:
        #     print(e)
        #     print(i, imgpath)

        print('\n\n***********************************')
        print('getClusterCentres end..using {} s'.format(time.time() - timestart))
        print('***********************************\n\n')

    def descriptors2features(self, imgRootPaths):
        '''

        :param path_descriptors_list:
        :return: features_norm, id_path_dict
        '''
        timestart = time.time()
        print('\n\n***********************************')
        print('Descriptors to features beginning..')
        print('***********************************\n\n')

        imgPaths = [os.path.join(root, file) for root, dirs, files in os.walk(imgRootPaths) for file in files if
                    (file.split('/')[-1]).split('.')[-1] in ['jpg', 'jpeg', 'png']]
        # extract features
        # descriptors_np = np.zeros((1, 128), dtype='float32')
        id_path_dict = {}
        features_list = []
        Skip_counter = 0  # 因为sift特征数小于PCA需要的维度数而省略的图片总数
        for i, imgpath in enumerate(imgPaths):
            if i % 100 == 0:
                print('Image {} is converting to feature'.format(i))
            img = cv2.imread(imgpath, 0)

            kps, descritors = self.orb_detector.detectAndCompute(img, None)
            if len(kps) < 1:
                Skip_counter += 1
                continue

            # 开始id和path字典放在continue前导致出现continue的情况下id和path也录入了
            id_path_dict[i] = imgpath
            # VLAD
            feature_vlad = self.Vlad(descritors)
            # print("i:", i, "imgpath:", imgpath, "feature_vlad:", feature_vlad)
            features_list.append(feature_vlad.tolist())
        # L2归一化
        # features_norm = features / np.linalg.norm(features, ord=2, axis=1, keepdims=True)

        print('\n\n***********************************')
        print('Descriptors to features end..using {} s'.format(time.time() - timestart))
        print('Skip_counter:', Skip_counter)
        print('***********************************\n\n')
        return features_list, id_path_dict

    def Vlad(self, descritors):

        pred = self.kmeans.predict(descritors)
        Vlad = np.zeros((self.num_words, self.orb_dimention))
        for j, index in enumerate(pred):
            # print(self.kmeans.cluster_centers_.shape)
            # print(descritors.shape)
            Vlad[index] += descritors[j] - self.kmeans.cluster_centers_[index]

        # 应该先做norm再flatten,开始反了结果不理想(后改进成论文中描述，先每行L2后flatten，再
        # 做SSR(signed square rooting)norm,最后L2)
        norm = np.linalg.norm(Vlad, axis=1).reshape((-1, 1))
        # 可能遇上0/0的情况
        norm[norm < 1e-12] = 1e-12
        Vlad = Vlad / norm
        Vlad = Vlad.flatten()
        # SSR Normlization
        Vlad = np.sign(Vlad) * np.sqrt(np.abs(Vlad))
        # L2 normlization
        Vlad = Vlad / np.linalg.norm(Vlad)
        # print("VLAD:",Vlad)
        return Vlad

    def saveKmeans(self, savePath='./ORBFile/Kmeans.pkl'):
        output = open(savePath, 'wb')
        pickle.dump(self.kmeans, output)
        output.close()

    def saveI2PJson(self, id_path_dict, savePath='./ORBFile/Image2Path_ORB.json'):
        json.dump(id_path_dict, open(savePath, 'w'))

    # def savePCA(self, savePath='./ORBFile/pca.pkl'):
    #     output = open(savePath, 'wb')
    #     pickle.dump(self.pca, output)
    #     output.close()
