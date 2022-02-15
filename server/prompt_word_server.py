import logging
import logging.handlers

import requests
import json
import jsonlines
from config.project_config import top1B_path
from sanic import Sanic
from sanic.response import json as sanic_json
import time

formatter = logging.Formatter(fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
rf_handler = logging.handlers.TimedRotatingFileHandler(filename="logs/prompt_server.log", when='D', interval=7,
                                                       backupCount=60)

logger_prompt = logging.getLogger("prompt_keyword")
logger_prompt.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
rf_handler.setFormatter(formatter)

logger_prompt.addHandler(ch)
logger_prompt.addHandler(rf_handler)

logger_prompt.info("start")


def get_generate_keywords(text, num=1):
    """
        生成关键字接口--
        速度、效果做了特殊的优化
        返回 提取的关键词、生成的关键词
    """

    url = "http://192.168.6.247:61177/tree_keywords"

    payload = {"title": text, "num": num}
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    result_json = response.json()
    return result_json["keywords"]


def get_extract_keywords(text):
    """正文关键字提取接口--支持任意字数的提取"""

    url = "http://ai-title-keywords-extract.k8s.laibokeji.com/text2keywords"

    payload = {"text": text}
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    result_json = response.json()
    return result_json


def sort_extract_word(keywords, top1B, max_len=10):
    """正文提取的排序"""

    keyword_frequency = {}
    for word in keywords:
        if word in top1B:
            keyword_frequency[word] = top1B[word]

    keyword_frequency = sorted(keyword_frequency.items(), key=lambda x: x[1])
    keyword_frequency = [word[0] for word in keyword_frequency][:max_len]

    return keyword_frequency


def prompt_word(title=None, outlinetitle=None, content=None, max_len=5):
    """选择根据什么提取关键词"""

    if content and len(content) > 20:
        keywords = get_extract_keywords(content)["keywords"]
        keywords = sort_extract_word(keywords, top1B, max_len)

    elif outlinetitle and len(outlinetitle) > 9:

        outlinetitle = "".join(outlinetitle.split(" ")[1:])
        keywords = get_generate_keywords(outlinetitle)
        keywords = list(set(keywords["keywords"][0] + keywords["participle"]))

    elif title and len(title) > 5:
        keywords = get_generate_keywords(title)
        keywords = list(set(keywords["keywords"][0] + keywords["participle"]))

    else:
        keywords = []

    return keywords


def word_to_prompt_words(word):
    """构建标题"""
    title = "基于%s的研究" % word
    keywords = get_generate_keywords(title, 3)
    keywords = list(set([j for i in keywords["keywords"] for j in i] + keywords["participle"]))

    keywords = [i for i in keywords if i not in ("基于", "研究", "的")]
    return keywords


app = Sanic(__name__)


@app.route("/prompt", methods=['POST'])
async def reductions(request):
    jsonDic = request.json
    logger_prompt.info("输入->%s" % str(jsonDic))

    try:
        title = jsonDic["title"]
    except KeyError:
        title = None

    try:
        outlinetitle = jsonDic["outlinetitle"]
    except KeyError:
        outlinetitle = None

    try:
        content = jsonDic["content"]
    except KeyError:
        content = None

    try:
        num = jsonDic["num"]
    except KeyError:
        num = 10

    try:
        prompt_keyword = prompt_word(title=title, outlinetitle=outlinetitle, content=content, max_len=num)
        prompt_keyword = {"code": 0, "msg": "success", "data": {"prompt_keyword": prompt_keyword}, "time": int(time.time())}
    except Exception as ex:
        logger_prompt.error(ex)
        prompt_keyword = {"code": 0, "msg": "null", "data": {}, "time": int(time.time())}

    logger_prompt.info("输出->%s" % str(prompt_keyword))
    return sanic_json(prompt_keyword)


@app.route("/words", methods=['POST'])
async def words(request):
    jsonDic = request.json
    logger_prompt.info("输入->%s" % str(jsonDic))

    try:
        word = jsonDic["word"]
    except Exception:
        return sanic_json({"code": -1, "msg": "输入错误", "data": {}, "time": int(time.time())})

    try:
        prompt_keyword = word_to_prompt_words(word)
        prompt_keyword = {"code": 0, "msg": "success", "data": {"prompt_keyword": prompt_keyword}, "time": int(time.time())}

    except Exception as ex:
        logger_prompt.error(ex)
        prompt_keyword = {"code": 0, "msg": "null", "data": {}, "time": int(time.time())}

    return sanic_json(prompt_keyword)


if __name__ == '__main__':

    top1B = {}
    with open(top1B_path, "r") as r:
        read = r.read()
        for w in read.split("\n"):
            if len(w) > 1:
                w = w.split("\t")
                top1B[w[0]] = int(w[-1])

    app.run(host="0.0.0.0", port=51677, workers=4)
