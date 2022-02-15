from jieba import posseg as pseg
from config.project_config import good_word_path
not_start_pos = ["uj", "c", "p"]


def query_word(title, word_dict):
    word_list = list()
    words, pos = [], []

    for w, p in pseg.lcut(title):
        words.append(w)
        pos.append(p)

    good_word_ids = []
    for i in range(len(words)):
        if len(good_word_ids) > 0:
            if i <= good_word_ids[-1]:
                continue
        if pos[i] in not_start_pos:
            continue
        if words[i] in word_dict:  # 如果本身在词库，直接添加
            good_word_ids = [i]
        if i < len(words) - 1:  # 往前加一
            if "".join(words[i:i + 2]) in word_dict:
                good_word_ids = [i, i + 1]
        if i < len(words) - 2:  # 往前加二
            if "".join(words[i:i + 3]) in word_dict:
                good_word_ids = [i, i + 1, i + 2]
        if i < len(words) - 3:  # 往前加三
            if "".join(words[i:i + 4]) in word_dict:
                good_word_ids = [i, i + 1, i + 2, i + 3]
        if i < len(words) - 4:  # 往前加4
            if "".join(words[i:i + 5]) in word_dict:
                good_word_ids = [i, i + 1, i + 2, i + 3, i + 4]
        if i < len(words) - 5:  # 往前加5
            if "".join(words[i:i + 6]) in word_dict:
                good_word_ids = [i, i + 1, i + 2, i + 3, i + 4, i + 5]
        # 去掉频率为千万级别的词
        if len(good_word_ids) > 0:
            if len(good_word_ids) > 0:
                new_word = "".join(words[good_word_ids[0]:good_word_ids[-1] + 1])
                if new_word not in word_list:
                    word_list.append(new_word)
    word_list_new = []
    for wor in word_list:
        if len(wor) < 2:
            continue

        if wor[0] == "与":
            wor = wor[1:]

        if wor[-1] == "的":
            wor = wor[:-1]

        if len(wor) > 1:
            word_list_new.append(wor)

    return word_list_new


def filter_stopword(text):  # 过滤掉连词等停用词
    words = []
    pos = []
    for w, p in pseg.cut(text):
        if p not in ["c", "u", "w", "xc", "a", "f"]:
            words.append(w)
            pos.append(p)
    return words, pos


def parentheses(title):
    p1 = re.compile('[「\[{【（(](.*?)[)）】}」\]]', re.S)
    p2 = re.compile('[《](.*?)[》]', re.S)

    special_1 = re.findall(p1, title)
    special_2 = re.findall(p2, title)

    special_2 = ["《" + i + "》" for i in special_2]
    special_1.extend(special_2)

    return special_1


def keyword_en(title):
    word = []
    s = ""
    new_title = ""
    remove_nota = u'[’·°!"#$%&\'()*+,. :;<=>?@，。?★、…【】（）？“”‘’！[\\]^`{|}~]+'

    for i in title:
        if i not in remove_nota:
            new_title += i

            if not '\u4e00' <= i <= '\u9fff':
                s += i
            else:
                if len(s) > 1:
                    word.append(s)
                s = ""

    return word, new_title.lower()


def load_good_word():
    word_dict = {}
    with open(good_word_path, "r", encoding="utf-8") as read:
        raw_data = read.read()
        for line_data in raw_data.split("\n"):
            w2f = line_data.split(" : ")
            try:
                if int(w2f[1]) * len(w2f[0]) > 6:
                    if len(w2f[0]) > 12 and (len(w2f[0]) - 12) * int(w2f[1]) > 6:
                        pass
                    else:
                        word_dict[w2f[0]] = w2f[1]
            except IndexError:
                print(line_data)
    return word_dict
