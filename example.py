import os
import random

from analyzer import plot_vectors, get_analyzer
from preprocessor import CutResult, cut_poetry


def print_counter(counter):
    for key, value in counter:
        print(key, value)
    print()


def example():
    saved_dir = os.path.join(os.curdir, "out")
    result = cut_poetry("全宋词.txt", saved_dir)
    analyzer = get_analyzer(result, saved_dir)
    # 画图
    tf_idf_vector_list = []
    w2v_vector_list = []
    author_list = []
    for c in result.author_counter.most_common(100):
        author = c[0]
        index = analyzer.authors.index(author)
        w2v_vector_list.append(analyzer.w2v_word_vector_tsne[index])
        tf_idf_vector_list.append(analyzer.tfidf_word_vector_tsne[index])
        author_list.append(author)
    plot_vectors(tf_idf_vector_list, author_list, 'tf_idf')
    plot_vectors(w2v_vector_list, author_list, 'w2v')

    print("**基于统计的分析")
    print("写作数量排名：")
    most_productive_poets = result.author_counter.most_common(10)
    print_counter(most_productive_poets)

    print("最常用的词：")
    cnt = 0
    l = []
    for word, count in result.word_counter.most_common():
        if cnt == 10:
            break
        if len(word) > 1:
            l.append((word, count))
            cnt += 1
    print_counter(l)

    print("最常用的名词：")
    most_common_nouns = result.word_property_counter_dict['n'].most_common(10)
    print_counter(most_common_nouns)

    print("最常见的地名：")
    print_counter(result.word_property_counter_dict['ns'].most_common(10))

    print("最常见的形容词：")
    print_counter(result.word_property_counter_dict['a'].most_common(10))

    print("**基于词向量的分析")
    for word in list(most_common_nouns):
        print("与 %s 相关的词：" % word[0])
        print_counter(analyzer.find_similar_word(word[0]))

    for poet in list(most_productive_poets)[:4]:
        print("与 %s 用词相近的诗人：" % poet[0])
        print("根据tf-idf标准： %s" % analyzer.find_similar_poet(poet[0]))
        print("根据word2vector标准： %s\n" % analyzer.find_similar_poet(poet[0], use_w2v=True))


def shrink():
    poems = []
    with open('全宋词.txt', 'r', encoding='utf-8') as f:
        poem = ''
        choose_count = 0
        abandon_count = 0
        for line in f:
            if line.strip() == "":
                continue

            if "【" in line:
                if random.random() < 0.1:
                    choose_count += 1
                    poems.append('\n')
                    poems.append(poem)
                else:
                    abandon_count += 1

                poem = ''
            poem += line

    print(choose_count)
    print(abandon_count)

    with open('全宋词_tiny.txt', 'w', encoding='utf-8') as f:
        f.writelines(poems)


if __name__ == '__main__':
    example()
    # shrink()
