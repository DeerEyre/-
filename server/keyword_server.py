import torch
import utils.utils_seq2seq as utils_seq2seq
import utils.utils_words as utils_words
import jieba.posseg as pseg
import copy
from config.project_config import *
from sanic import Sanic
from sanic.response import json as sanic_json
from model.tokenization_unilm import UnilmTokenizer
from model.modeling_unilm import UnilmForSeq2SeqDecode, UnilmConfig

import model.keyword_model as keyword
import time

app = Sanic("keywords")

device = torch.device(cuda if torch.cuda.is_available() else "cpu")
mask_word_id, eos_word_ids, sos_word_id = 103, 102, 100

config = UnilmConfig.from_pretrained(model_config_path, max_position_embeddings=512)

print("开始读取字典信息")
word_dict = utils_words.load_good_word()
print("读取字典信息完成")


print("开始读取关键词模型")
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
print("读取关键词模型完成")


@app.route("/tree_keywords", methods=['POST'])
async def title_words(request):
    jsonDic = request.json

    print("输入: ", jsonDic)

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

    out = keyword.generate_keyword(title, num,
                                   model=model_keyword,
                                   device_=device,
                                   word_dict_=word_dict)

    print("输出: ", out)

    return sanic_json({"keywords": out})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=61177)
