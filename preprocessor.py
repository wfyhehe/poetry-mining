import os
import pickle

from collections import Counter, OrderedDict
from jieba import posseg as pseg
from constants import TITLE_CONTENT_THRESHOLD, END_POET_LIST_SEPARATOR, STEM_RESULT_FILENAME


class StemResult(object):
    """
    分词结果
    char_counter：字频统计
    poet_counter：作者计数
    word_set：词汇表
    word_counter：词汇计数
    word_property_counter_dict：词汇词性
    poet_poetry_dict：解析后的结果，作者与他对应的诗
    """

    def __init__(self):
        self.word_set = set()
        self.word_counter = Counter()
        self.word_property_counter_dict = {}
        self.poet_word_counter = {}
        self.char_counter = Counter()
        self.poet_counter = Counter()
        self.poet_poetry_dict = OrderedDict()

    def add_stem_poetry(self, poet, divided_lines):
        """为poet_poetry_dict添加对象"""
        ctp = self.poet_poetry_dict.get(poet)
        if ctp is None:
            self.poet_poetry_dict[poet] = ""
        else:
            self.poet_poetry_dict[poet] += ' '
        self.poet_poetry_dict[poet] += ' '.join(divided_lines)

    @classmethod
    def _is_chinese_character(cls, c):
        return '\u4e00' <= c <= '\u9fff'

    def stem_poem(self, filename, saved_dir):
        """
        对全宋词分词
        :param: filename: 全宋词文件名
                saved_dir: 储存位置(out)
        :return:分词结果
        """
        target_file_path = os.path.join(saved_dir, STEM_RESULT_FILENAME)
        if not os.path.exists(saved_dir):
            os.mkdir(saved_dir)
        if os.path.exists(target_file_path):
            print('load existed stem result.')
            with open(target_file_path, 'rb') as f:
                return pickle.load(f)

        else:
            print('stemming poetry...')
            line_count = 0
            current_poet = None
            divided_lines = []
            poet_list = []
            poet_filled = False
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    if line_count % 1000 == 0:
                        print('%d lines processed.' % line_count)
                    try:
                        line = line.strip()
                        if line == "":
                            continue

                        if line == END_POET_LIST_SEPARATOR:
                            poet_filled = True
                            continue

                        if not poet_filled:
                            poet_list.append(line)
                            continue

                        # 解析作者
                        if line in poet_list:
                            divided_lines.append("\n")
                            current_poet = line
                            continue

                        if len(line) < TITLE_CONTENT_THRESHOLD:  # title
                            # 将当前分词后的结果加入结果表中
                            self.add_stem_poetry(current_poet, divided_lines)
                            self.poet_counter[current_poet] += 1
                            divided_lines = []
                            continue

                        # 解析诗句
                        chars = [c for c in line if self._is_chinese_character(c)]
                        for char in chars:
                            self.char_counter[char] += 1
                        cut_line = pseg.cut(line)
                        for word, property in cut_line:
                            # 非中文则跳过
                            if not self._is_chinese_character(word):
                                continue

                            # 统计不同词性词频
                            if self.word_property_counter_dict.get(property) is None:
                                self.word_property_counter_dict[property] = Counter()
                            self.word_property_counter_dict[property][word] += 1
                            self.word_set.add(word)
                            self.word_counter[word] += 1

                            # 统计每个词人的用词词频
                            if self.poet_word_counter.get(current_poet) is None:
                                self.poet_word_counter[current_poet] = Counter()
                            self.poet_word_counter[current_poet][word] += 1

                            divided_lines.append(word)

                    except Exception as e:
                        print('{line_num}-解析全宋词文件异常 {line}'.format(
                            line_num=line_count,
                            line=line,
                        ))
                        raise e

            # 加入最后一次解析的结果
            self.add_stem_poetry(current_poet, divided_lines)
            with open(target_file_path, 'wb') as f:
                pickle.dump(self, f)
            f.close()
            print('closed')

        return self
