import torch
import utils.utils_seq2seq as utils_seq2seq
import jieba.posseg as pseg
import random
import re
from model.tokenization_unilm import UnilmTokenizer
from config.project_config import tokenizer_path
from utils.utils_words import *

max_src_length = 512 - 2 - 200
tokenizer = UnilmTokenizer.from_pretrained(tokenizer_path)
bi_uni_pipeline = [utils_seq2seq.Preprocess4Seq2seqDecode(list(tokenizer.vocab.keys()),
                                                          tokenizer.convert_tokens_to_ids,
                                                          512, max_tgt_length=200)]


def generate_step(model, input_ids, token_type_ids, position_ids, attention_mask, top_k=5, penalty_factor=1):
    input_shape = list(input_ids.size())
    batch_size = input_shape[0]
    input_length = input_shape[1]
    output_shape = list(token_type_ids.size())
    output_length = output_shape[1]
    output_ids = []
    prev_embedding = None
    prev_encoded_layers = None
    curr_ids = input_ids
    mask_ids = input_ids.new(batch_size, 1).fill_(model.mask_word_id)
    next_pos = input_length

    is_sep = [0] * batch_size

    temporary = [[] for _ in range(batch_size)]

    generate_st = [[] for _ in range(batch_size)]

    while next_pos < output_length:
        curr_length = list(curr_ids.size())[1]
        start_pos = next_pos - curr_length
        x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
        curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
        curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
        curr_position_ids = position_ids[:, start_pos:next_pos + 1]

        new_embedding, new_encoded_layers, _ = \
            model.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                       output_all_encoded_layers=True, prev_embedding=prev_embedding,
                       prev_encoded_layers=prev_encoded_layers)

        last_hidden = new_encoded_layers[-1][:, -1:, :]
        prediction_scores = model.cls(last_hidden)  # 模型输出 shape(1, 1, tokenizer.size)
        prediction_scores[:, :, 100] = 10e-6
        prediction_scores[:, :, 8038] = 10e-6
        prediction_scores[:, :, 8024] = 10e-6
        prediction_scores[:, :, 131] = 10e-6
        prediction_scores[:, :, 117] = 10e-6  # ,
        prediction_scores[:, :, 8024] = 10e-6  # ，
        prediction_scores[:, :, 8020] = 10e-6  # （
        prediction_scores[:, :, 8021] = 10e-6  # ）
        prediction_scores[:, :, 150] = 10e-6  # ）
        prediction_scores[:, :, 114] = 10e-6  # ）
        prediction_scores[:, :, 113] = 10e-6  # ）

        for p in range(batch_size):
            if len(temporary[p]) == 0:
                for i in generate_st[p]:
                    prediction_scores[p, :, i[0]] *= penalty_factor
            else:
                for i in temporary[p]:
                    prediction_scores[p, :, i] *= penalty_factor

        top_k_prob, top_k_token = torch.topk(prediction_scores, dim=-1, k=top_k)
        choice_ids = torch.multinomial(torch.softmax(top_k_prob[:, 0, :], dim=-1), num_samples=1)

        true_choice_ids = []

        for i in range(batch_size):

            if top_k_prob[i, :, choice_ids[i]].item() / torch.mean(top_k_prob[i, :, :]).item() < 0.5:
                choice_ids = torch.multinomial(torch.softmax(top_k_prob[:, 0, :][:2], dim=-1), num_samples=1)
                next_token = top_k_token[i, :, choice_ids[i]]

            else:
                next_token = top_k_token[i, :, choice_ids[i]]

            true_choice_ids.append(next_token)

        true_choice_ids = torch.cat(true_choice_ids, dim=0)
        output_ids.append(true_choice_ids)

        if prev_embedding is None:
            prev_embedding = new_embedding[:, :-1, :]
        else:
            prev_embedding = torch.cat((prev_embedding, new_embedding[:, :-1, :]), dim=1)

        if prev_encoded_layers is None:
            prev_encoded_layers = [x[:, :-1, :]
                                   for x in new_encoded_layers]
        else:
            prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                   for x in zip(prev_encoded_layers, new_encoded_layers)]

        output = torch.reshape(true_choice_ids, (1, -1)).tolist()[0]

        for i in range(batch_size):
            if output[i] == 102:
                if is_sep[i] != 1 and len(temporary[i]) > 0:
                    generate_st[i].append(temporary[i])
                is_sep[i] = 1

            if is_sep[i] != 1:
                if output[i] == 8102:
                    generate_st[i].append(temporary[i])
                    temporary[i] = []
                else:
                    temporary[i].append(output[i])

        if sum(is_sep) == batch_size or next_pos - input_length > 50:
            break

        curr_ids = true_choice_ids
        next_pos += 1

    return torch.cat(output_ids, dim=1)


def output_format(unilm_model, batch_context, batch_size, device_, penalty_factor=1, topk=5):
    next_i = 0
    outputs = []
    input_lines = [tokenizer.tokenize(c)[:max_src_length] for c in batch_context]
    input_lines = [(v, i) for v, i in enumerate(input_lines)]

    while next_i < len(input_lines):
        _chunk = input_lines[next_i:next_i + batch_size]

        buf = [x[1] for x in _chunk]
        next_i += batch_size
        max_a_len = max([len(x) for x in buf])
        instances = []
        for instance in [(x, max_a_len) for x in buf]:
            for proc in bi_uni_pipeline:
                instances.append(proc(instance))

        with torch.no_grad():
            batch = utils_seq2seq.batch_list_to_batch_tensors(instances)
            batch = [t.to(device_) if t is not None else None for t in batch]
            input_ids, token_type_ids, position_ids, input_mask = batch
            traces = generate_step(unilm_model, input_ids, token_type_ids, position_ids,
                                   input_mask, topk, penalty_factor).tolist()
            [outputs.append(tokenizer.decode(t)) for t in traces]

    return [j.replace(" ", "") for j in [i.split("[SEP]")[0] for i in outputs]]


def generate_keyword(title, num, model, device_, word_dict_):
    titles = [title for _ in range(num * 2)]
    output_keyword = output_format(unilm_model=model,
                                   batch_context=titles,
                                   batch_size=len(titles),
                                   device_=device_,
                                   penalty_factor=0.9,
                                   topk=5)

    keywords = []
    for i in output_keyword:
        keyword = []

        for j in list(set(i.split("👍"))):
            if j in word_dict_:
                keyword.append(j)
        keywords.append(keyword)

    keywords = sorted(keywords, key=lambda x: -len(x))

    p = parentheses(title)

    word_en, new_title = keyword_en(title)

    for i in p:
        new_title = new_title.replace(i, "")

    participle = query_word(new_title, word_dict_)
    keywords = [list(set(keywords[random.randint(0, num)] + p)) for i in range(num)]

    for v, i in enumerate(keywords):
        if len(i) < 3:
            i.extend([participle[random.randint(0, len(participle) - 1)]])
            keywords[v] = list(set(i))

    for v, i in enumerate(keywords):
        for j in word_en:
            if j.lower() not in i:
                keywords[v].append(j)

    for j in word_en:
        if j.lower() not in participle:
            participle.append(j)

    return {"keywords": keywords, "participle": participle + p, "word_en": word_en}


def main():
    pass


if __name__ == '__main__':
    main()
    # print(generate_keyword("基于深度学习的文本分类", 3))
    # keyword.generate_keyword("基于深度学习的文本分类", 3)
