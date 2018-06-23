import os
import pickle

from collections import Counter, OrderedDict
from jieba import posseg as pseg
from constants import TITLE_CONTENT_THRESHOLD, END_POET_LIST_SEPARATOR, STEM_RESULT_FILENAME


class CutResult(object):
    """
    分词结果
    char_counter：字频统计
    author_counter：作者计数
    word_set：词汇表
    word_counter：词汇计数
    word_property_counter_dict：词汇词性
    author_poetry_dict：解析后的结果，作者与他对应的诗
    """

    def __init__(self):
        self.word_set = set()
        self.word_counter = Counter()
        self.word_property_counter_dict = {}
        self.char_counter = Counter()
        self.author_counter = Counter()
        self.author_poetry_dict = OrderedDict()

    def add_cut_poetry(self, author, divided_lines):
        """为author_poetry_dict添加对象"""
        ctp = self.author_poetry_dict.get(author)
        if ctp is None:
            self.author_poetry_dict[author] = ""
        else:
            self.author_poetry_dict[author] += ' '
        self.author_poetry_dict[author] += ' '.join(divided_lines)


def _is_chinese_character(c):
    return '\u4e00' <= c <= '\u9fff'


def stem_poem(filename, saved_dir):
    """
    对全宋词分词
    :PARAM: filename: 全宋词文件名
            saved_dir: 储存位置(out)
    :RETURN:分词结果
    """
    target_file_path = os.path.join(saved_dir, STEM_RESULT_FILENAME)
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    if os.path.exists(target_file_path):
        print('load existed cut result.')
        with open(target_file_path, 'rb') as f:
            result = pickle.load(f)
    else:
        print('cutting poetry...')
        result = CutResult()
        line_count = 0
        current_author = None
        divided_lines = []
        author_list = []
        author_filled = False
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
                        author_filled = True
                        continue

                    if not author_filled:
                        author_list.append(line)
                        continue

                    # 解析作者
                    if line in author_list:
                        divided_lines.append("\n")
                        current_author = line
                        continue

                    if len(line) < TITLE_CONTENT_THRESHOLD:  # title
                        # 将当前分词后的结果加入结果表中
                        result.add_cut_poetry(current_author, divided_lines)
                        result.author_counter[current_author] += 1
                        divided_lines = []
                        continue

                    # 解析诗句
                    chars = [c for c in line if _is_chinese_character(c)]
                    for char in chars:
                        result.char_counter[char] += 1
                    cut_line = pseg.cut(line)
                    for word, property in cut_line:
                        if not _is_chinese_character(word):
                            continue
                        if result.word_property_counter_dict.get(property) is None:
                            result.word_property_counter_dict[property] = Counter()
                        result.word_property_counter_dict[property][word] += 1
                        result.word_set.add(word)
                        result.word_counter[word] += 1
                        divided_lines.append(word)
                except Exception as e:
                    print('{line_num}-解析全宋词文件异常 {line}'.format(
                        line_num=line_count,
                        line=line,
                    ))
                    raise e
        # 加入最后一次解析的结果
        result.add_cut_poetry(current_author, divided_lines)
        with open(target_file_path, 'wb') as f:
            pickle.dump(result, f)
        f.close()
        print('closed')

    return result
