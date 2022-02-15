import logging
import logging.handlers

import torch
import utils.utils_seq2seq as utils_seq2seq
import utils.utils_words as utils_words
import model.keyword_model as keyword

from config.project_config import *
from sanic import Sanic
from sanic.response import json as sanic_json
from model.tokenization_unilm import UnilmTokenizer
from model.modeling_unilm import UnilmForSeq2SeqDecode, UnilmConfig


formatter = logging.Formatter(fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
rf_handler = logging.handlers.TimedRotatingFileHandler(filename="logs/keyword_server.log", when='D', interval=7,
                                                       backupCount=60)

logger_keyword = logging.getLogger("keyword_server")
logger_keyword.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
rf_handler.setFormatter(formatter)

logger_keyword.addHandler(ch)
logger_keyword.addHandler(rf_handler)

logger_keyword.info("start")


device = torch.device(cuda if torch.cuda.is_available() else "cpu")
mask_word_id, eos_word_ids, sos_word_id = 103, 102, 100

config = UnilmConfig.from_pretrained(model_config_path, max_position_embeddings=512)

logger_keyword.info("开始读取字典信息")
word_dict = utils_words.load_good_word()
logger_keyword.info("读取字典信息完成")

logger_keyword.info("开始读取关键词模型")
model_keyword = UnilmForSeq2SeqDecode.from_pretrained(model_path,
                                                      state_dict=torch.load(model_path, map_location="cpu"),
                                                      config=config,
                                                      mask_word_id=mask_word_id,
                                                      search_beam_size=1,
                                                      length_penalty=0,
                                                      eos_id=eos_word_ids,
                                                      sos_id=sos_word_id,
                                                      ngram_size=3)
model_keyword = model_keyword.to(device).eval()
logger_keyword.info("读取关键词模型完成")

app = Sanic("keywords")


@app.route("/tree_keywords", methods=['POST'])
async def title_words(request):
    jsonDic = request.json

    logger_keyword.info("输入: %s" % str(jsonDic))

    try:
        title = jsonDic["title"]
        if len(title) > 50 or len(title) < 5:
            return sanic_json({"result_code": 0, "info": "输入过短或过短长 建议标题长度5-30"})

    except Exception:
        return sanic_json({"result_code": 0, "info": "缺少题目参数"})

    try:
        num = jsonDic["num"]
    except Exception:
        num = 1
    try:
        out = keyword.generate_keyword(title, num=num,
                                       model=model_keyword,
                                       device_=device,
                                       word_dict_=word_dict)
    except Exception as ex:
        logger_keyword.error(ex)
        out = {"keywords": [[]], "participle": [], "word_en": []}

    torch.cuda.empty_cache()

    logger_keyword.info("输出: %s" % str(out))

    return sanic_json({"keywords": out})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=61177)
