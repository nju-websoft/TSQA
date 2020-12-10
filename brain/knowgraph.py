# coding: utf-8
"""
KnowledgeGraph
"""
import random

import jieba
import jieba.analyse
import numpy as np
import pandas as pd

import brain.config as config


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, args):
        self.args = args
        self.special_tags = set(config.NEVER_SPLIT_TAG)
        self.rank_lookup_table = []
        self.rank_datas_merge = {}
        self.rank_datas = self.load_gaokao_rank()
        self.rank_datas = {**self.rank_datas, **self.rank_datas_merge}

    def load_gaokao_rank(self):
        datas = pd.read_csv(self.args.kg_rank_file, sep='\t')
        rank_datas = {}
        tmp_rank_data_merge = {}
        for i in range(len(datas)):
            guid = datas['guid'][i]
            rank = int(datas['rank'][i])
            score = float(datas['score'][i])
            if not guid in rank_datas:
                rank_datas[guid] = []
            if len(rank_datas[guid]) >= self.args.max_entities:  # config.MAX_ENTITIES:
                continue
            if score < 0.85:
                continue
            gguid = guid.split('-')[0]
            kv = datas['kv'][i].replace('nan', '')
            if len(kv) > config.EXTRACT_CORE_WORDS_LEN:
                kv = ''.join(jieba.analyse.extract_tags(kv, topK=3))
            rank_datas[guid].append((datas['title'][i], kv))
            if not gguid in self.rank_datas_merge:
                tmp_rank_data_merge[gguid] = []
            is_add = True
            for t in tmp_rank_data_merge[gguid]:
                if t[0] == datas['title'][i]:
                    is_add = False
                    break
            if is_add:
                tmp_rank_data_merge[gguid].append((datas['title'][i], kv, score))
            self.rank_lookup_table.append(datas['title'][i])
        for gguid in tmp_rank_data_merge:
            triples = tmp_rank_data_merge[gguid]
            triples.sort(key=lambda x: x[2], reverse=True)
            triples = triples[: self.args.max_entities]
            triples = [(triple[0], triple[1]) for triple in triples]
            self.rank_datas_merge[gguid] = triples
        return rank_datas

    def sentence_seg(self, sentence):
        tokens = [w for w in jieba.cut(sentence)]
        re_tokens = []
        tokens_len = len(tokens)
        for i in range(tokens_len):
            for j in range(3, 0, -1):
                if i + j < tokens_len:
                    if ''.join(tokens[i:i + j + 1]) in self.rank_lookup_table:
                        re_tokens.append(''.join(tokens[i:i + j + 1]))
                        i = i + j
                        break
            re_tokens.append(tokens[i])
        re_re_tokens = []
        speci_token = ['unused1', 'unused2', 'CLS', 'SEP']
        for i in range(len(re_tokens)):
            token = re_tokens[i]
            if token == '[':
                if i + 2 < len(re_tokens) and re_tokens[i + 1] in speci_token and re_tokens[i + 2] == ']':
                    re_re_tokens.append(''.join(re_tokens[i:i + 3]))
            if token not in speci_token and token not in ['[', ']']:
                re_re_tokens.append(token)
        return re_re_tokens

    def add_knowledge_with_vm(self, sent_batch, max_entities, add_pad=True, max_length=256,
                              guid=None):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        # split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        split_sent_batch = [self.sentence_seg(sent) for sent in sent_batch]

        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for index, split_sent in enumerate(split_sent_batch):

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            for token in split_sent:
                if True:
                    entities = []
                    if guid[index] in self.rank_datas:
                        rank_entities = self.rank_datas[guid[index]]
                        for re in rank_entities:
                            if re[0] == token:
                                if random.random() <= self.args.entities_prob:
                                    entities.append(re[1].replace(re[0], '', 1))
                    assert len(entities) <= self.args.max_entities
                    entities = list(filter(lambda x: x.strip() != '', entities))
                sent_tree.append((token, entities))

                if token in self.special_tags:
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                else:
                    token_pos_idx = [pos_idx + i for i in range(1, len(token) + 1)]
                    token_abs_idx = [abs_idx + i for i in range(1, len(token) + 1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx
            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = list(word)
                    know_sent += add_word
                    seg += [0] * len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = list(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch
