# -*- coding: utf-8 -*-
import multiprocessing
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from constants import ANALYZE_RESULT_FILENAME, MIN_COUNT, W2V_DIMENSION
from gensim.models.word2vec import LineSentence, Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import manifold

mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class Analyzer(object):
    """
    stem_result:分词结果
    poets: 词人列表
    w2v_model: 用word2vector得到的model
    w2v_word_vector: 用word2vector得到的词向量
    w2v_word_vector_tsne: 降维后的word2vector词向量
    tfidf_word_vector: 用tf-idf为标准得到的词向量
    tfidf_word_vector_tsne: 降维后的tf-idf词向量
    """

    def __init__(self, stem_result):
        self.stem_result = stem_result
        self.poets = list(stem_result.poet_poetry_dict.keys())
        print('begin analyzing stem result...')
        self.stem_result = stem_result
        print('calculating poets tf-idf word vector...')
        self.tfidf_word_vector = self._tfidf_word_vector(stem_result.poet_poetry_dict)
        print('calculating poets w2v word vector...')
        self.w2v_model, self.w2v_word_vector = self._w2v_word_vector(stem_result.poet_poetry_dict)
        print('use t-sne for dimensionality reduction...')
        self.tfidf_word_vector_tsne = self._tsne(self.tfidf_word_vector)
        self.w2v_word_vector_tsne = self._tsne(self.w2v_word_vector)
        print('result saved.')

    @staticmethod
    def _tfidf_word_vector(poet_poetry_dict):
        """tf-idf 解析每个作者的词向量"""
        poetry = list(poet_poetry_dict.values())
        vectorizer = CountVectorizer(min_df=MIN_COUNT)
        word_matrix = vectorizer.fit_transform(poetry).toarray()
        tfidf_word_vector = TfidfTransformer().fit_transform(word_matrix).toarray()
        return tfidf_word_vector

    @staticmethod
    def _w2v_word_vector(poet_poetry_dict):
        """word2vector 解析每个作者的词向量"""
        poets = list(poet_poetry_dict.keys())
        poetry = list(poet_poetry_dict.values())

        with open('temp', 'w') as f:
            f.write('\n'.join(poetry))

        model = Word2Vec(
            LineSentence('temp'),
            size=W2V_DIMENSION,
            min_count=MIN_COUNT,
            workers=multiprocessing.cpu_count(),
        )

        word_vector = []
        for i, poet in enumerate(poets):
            vec = np.zeros(W2V_DIMENSION)
            words = poetry[i].split()
            count = 0
            for word in words:
                word = word.strip()
                try:
                    vec += model[word]
                    count += 1
                except KeyError:  # 频率小于MIN_COUNT的词不会被统计
                    pass

            single_word_vector = np.array([v / count for v in vec]) \
                if count > 0 else np.zeros(len(vec))
            word_vector.append(single_word_vector)
        os.remove('temp')
        return model, word_vector

    @staticmethod
    def _tsne(word_vector):
        t_sne = manifold.TSNE(
            n_components=2,
            init='pca',
        )
        return t_sne.fit_transform(word_vector)

    def find_similar_poet(self, poet_name, use_w2v=False):
        """
        通过词向量寻找最相似的诗人
        :param: poet: 需要寻找的诗人名称
        :return:最匹配的诗人
        """
        word_vector = self.tfidf_word_vector if not use_w2v else self.w2v_word_vector
        poet_index = self.poets.index(poet_name)
        x = word_vector[poet_index]
        min_angle = np.pi
        min_index = 0
        for i, poet in enumerate(self.poets):
            if i == poet_index:
                continue
            y = word_vector[i]
            denominator = np.sqrt(x.dot(x)) * np.sqrt(y.dot(y))
            if denominator == 0:
                continue
            cos = x.dot(y) / denominator
            angle = np.arccos(cos)
            if min_angle > angle:
                min_angle = angle
                min_index = i
        return self.poets[min_index]

    def find_similar_word(self, word):
        return self.w2v_model.most_similar(word)


def plot_vectors(X, target, filename):
    """绘制结果"""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], target[i], fontdict={'size': 8})
    plt.savefig('images/%s' % filename)
    plt.show()


def get_analyzer(result, saved_dir):
    target_file_path = os.path.join(saved_dir, ANALYZE_RESULT_FILENAME)
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    if os.path.exists(target_file_path):
        print('load existed analyzer.')
        with open(target_file_path, 'rb') as f:
            analyzer = pickle.load(f)
    else:
        analyzer = Analyzer(result)
        with open(target_file_path, 'wb') as f:
            pickle.dump(analyzer, f)
        f.close()

    return analyzer
