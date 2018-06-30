import os
import random

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from constants import WordType, DISPLAY_COUNT, POET_COUNT, WORD_COULD_COUNT
from analyzer import plot_vectors, get_analyzer
from preprocessor import StemResult


def entry():
    # 确定保存位置
    saved_dir = os.path.join(os.curdir, "out")

    # 对宋词进行分词
    result = StemResult()
    result = result.stem_poem("全宋词.txt", saved_dir)

    # 用word2vector和tfidf算法对所有宋词数据进行学习和分析，得到词向量
    analyzer = get_analyzer(result, saved_dir)

    # 装填词人的词向量信息
    tf_idf_vector_list = []
    w2v_vector_list = []
    poet_list = []
    for c in result.poet_counter.most_common(POET_COUNT):
        poet = c[0]
        index = analyzer.poets.index(poet)
        w2v_vector_list.append(analyzer.w2v_word_vector_tsne[index])
        tf_idf_vector_list.append(analyzer.tfidf_word_vector_tsne[index])
        poet_list.append(poet)
    plot_vectors(tf_idf_vector_list, poet_list, 'tf_idf')
    plot_vectors(w2v_vector_list, poet_list, 'w2v')

    print('统计分析')
    print('-----------------')
    print('统计了%s位词人的%s首词' % (len(result.poet_counter.keys()), sum(result.poet_counter.values())))
    print("写作数量排名：")
    most_productive_poets = result.poet_counter.most_common(DISPLAY_COUNT)
    show_counter(most_productive_poets)

    print("最常用的非单字词：")
    cnt = 0
    most_frequent_words = []
    for word, count in result.word_counter.most_common():
        if cnt >= WORD_COULD_COUNT:
            break
        if len(word) > 1:
            most_frequent_words.append((word, count))
            cnt += 1
    show_counter(most_frequent_words[:DISPLAY_COUNT])

    most_frequent_word_dict = {}
    for word in most_frequent_words:
        most_frequent_word_dict[word[0]] = word[1]

    show_wordcloud(most_frequent_word_dict, '全宋词词云')

    print("最常用的名词：")
    most_common_nouns = result.word_property_counter_dict[WordType.NOUN].most_common(DISPLAY_COUNT)
    show_counter(most_common_nouns)

    print("最常用的地名：")
    show_counter(result.word_property_counter_dict[WordType.PLACE].most_common(DISPLAY_COUNT))

    print("最常用的形容词：")
    show_counter(result.word_property_counter_dict[WordType.ADJ].most_common(DISPLAY_COUNT))

    print("最常用的连词：")
    show_counter(result.word_property_counter_dict[WordType.CONJ].most_common(DISPLAY_COUNT))

    print("最常用的数词：")
    show_counter(result.word_property_counter_dict[WordType.NUM].most_common(DISPLAY_COUNT))

    print("最常用的介词：")
    show_counter(result.word_property_counter_dict[WordType.PREP].most_common(DISPLAY_COUNT))

    print("最常用的动词：")
    show_counter(result.word_property_counter_dict[WordType.VERB].most_common(DISPLAY_COUNT))

    print("**基于词向量的分析")
    for word in list(most_common_nouns):
        print("与 %s 相关的词：" % word[0])
        show_counter(analyzer.find_similar_word(word[0]))

    for poet in list(most_productive_poets)[:4]:
        poet_name = poet[0]
        print("与 %s 用词相近的诗人：" % poet_name)
        print("tf-idf标准： %s" % analyzer.find_similar_poet(poet_name))
        print("word2vector标准： %s\n" % analyzer.find_similar_poet(poet_name, use_w2v=True))

        cnt = 0
        most_frequent_words = []
        for word, count in result.poet_word_counter[poet_name].most_common():
            if cnt >= WORD_COULD_COUNT:
                break
            if len(word) > 1:
                most_frequent_words.append((word, count))
                cnt += 1
        most_frequent_word_dict = {}
        for word in most_frequent_words:
            most_frequent_word_dict[word[0]] = word[1]

        show_wordcloud(most_frequent_word_dict, poet_name)


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


def show_counter(counter):
    for k, v in counter:
        print(k, v)

    print()


def show_wordcloud(word_dict, title=None):
    wordcloud = WordCloud(
        font_path='Library/Fonts/msyh.ttf',
        background_color='white',
        scale=2,
    ).fit_words(word_dict)
    plt.imshow(wordcloud)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


if __name__ == '__main__':
    entry()
    # shrink()
