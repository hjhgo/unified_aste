import torch

import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from tqdm import tqdm
import pickle

def read_train_data(path):

    words = []
    tags = []
    with open(path,encoding="utf-8") as fp:

        for i ,l in enumerate(fp):
            sentence_tages = l.split("####")[1]
            words_i = []
            tags_i = []
            for t in sentence_tages.strip().split(" "):
                # print(t)
                ts = t.split("=")
                if len(ts) == 2:
                    words_i.append(ts[0].lower())
                    tags_i.append(ts[1])
                else:
                    words_i.append((len(ts) - 1) * "=")
                    tags_i.append(ts[-1])

            words.append(words_i)
            tags.append(tags_i)
    return words,tags

polaryity_label = {"T-POS": 0, "T-NEG": 1, "T-NEU": 2}

def read_data(file,mode="train"):
    words = []
    ote = []
    te = []
    spans = []
    polarity = []
    field = []
    with open(file, encoding="utf-8", mode="r") as fp:
        for l in fp:
            s, t = l.strip().split("####")
            words_i = []
            ote_i = []
            te_i = []
            spans_i = []
            polarity_i = []
            prev = "O"
            beg = -1

            for i, w_t in enumerate(t.split(" ")):
                a = w_t.split("=")
                if len(a)==2:
                    words_i.append(a[0].lower())
                else:
                    words_i.append((len(a) - 1) * "=")
                tag = a[-1]
                if tag=="O":
                    ote_i.append("O")
                    te_i.append("O")
                    if prev!="O":
                        spans_i.append([beg, i - 1])
                        polarity_i.append(prev)
                        beg = -1
                        prev = "O"
                else:
                    ote_i.append("T")

                    te_i.append(tag.split("-")[-1])
                    if prev=="O":
                        beg = i
                        prev = tag

                    elif prev!=tag:
                        spans_i.append([beg, i - 1])
                        polarity_i.append(prev)
                        beg = i
                        prev = tag

                    if i==len(t.split(" ")) - 1:
                        spans_i.append([beg, i])
                        polarity_i.append(prev)

            if len(spans_i)!=0:
                words.append(words_i)
                ote.append(ote_i)
                te.append(te_i)
                spans.append(spans_i)
                polarity.append(polarity_i)
                for s_i, s in enumerate(spans_i):
                    entity = words_i[s[0]:s[1] + 1]

    records = {}
    records["words"] = words
    records["ote"] = ote
    records["te"] = te
    records["spans"] = spans
    records["polaryity"] = polarity
    records["replace_entity"] = [0 for _ in range(len(words))]

    sentence_p = []
    for p in polarity:
            if len(set(p)) == 1:
                sentence_p.append(polaryity_label[p[0]])
            else:
                sentence_p.append(3)
    records["sentence_p"] = sentence_p
    return records

class Vocab:
    def __init__(self,file_list,
                 init=["PAD","UNK"]):

        self.init_words = init
        self.words = []
        self.chars = []
        self.words_counts = {}
        self.char_counts = {}
        self.word2ids = {}
        self.ids2word = {}
        self.char2ids = {}
        self.ids2char = {}
        self.file_list = file_list
        self.build_vocab()
        self.entiy_tag= {"O":0,"T-POS":1,"T-NEG":1,"T-NEU":1}
        self.poarity_tag  = {"T-POS":1,"T-NEG":2,"T-NEU":3}
        self.entiy_tag_bie = {"O":0,"B":1,"I":2,"E":3,"S":4}


    def build_vocab(self):
        # pass
        for f in self.file_list:
            print("process file ",f)
            words,tags = read_train_data(f)
            for l in words:
                for w in l:
                    try:
                        self.words_counts[w] += 1
                    except Exception as e:
                        # print(e)
                        self.words_counts[w] = 1

                    for c in w:
                        try:
                            self.char_counts[c] +=1
                        except Exception as e:
                            self.char_counts[c] = 1

        self.words.extend(self.init_words)
        self.words.extend(self.words_counts.keys())

        self.chars.extend(self.init_words)
        self.chars.extend(self.char_counts.keys())

        # print(self.words)
        for i,w in enumerate(self.words):
            self.word2ids[w] = i
            self.ids2word[i] = w

        for j,c in enumerate(self.chars):
            self.char2ids[c] = j
            self.ids2char[j] = c
            print(j,c)


    def covert_word2ids(self,sentences):
        sentences_ids = []
        for s in sentences:
            sentent = []
            for word in s:
                sentent.append(self.word2ids.get(word,1))
            sentences_ids.append(sentent)
        return sentences_ids

    def covert_char2ids(self,sentences):
        char_ids = []
        for s in sentences:
            temp = []
            for word in s:
                temp.append([self.char2ids.get(c, 1) for c in word])
            char_ids.append(temp)
        return char_ids


    def process_entity(self,entites):
        new_enties = []
        for l in entites:
            temp = []
            prev = 0
            for i in range(len(l)):
                if l[i] == 0:
                    temp.append(self.entiy_tag_bie["O"])
                    prev = 0
                else:
                    if (i <len(l)-1) and l[i+1] == 1:
                        if prev==0:
                            temp.append(self.entiy_tag_bie["B"])
                            prev = 1
                        else:
                            temp.append(self.entiy_tag_bie["I"])
                            prev = 1

                    else:
                        if prev==0:
                            temp.append(self.entiy_tag_bie["S"])
                            prev = 1
                        else:
                            temp.append(self.entiy_tag_bie["E"])
                            prev = 1


            new_enties.append(temp)
        return new_enties


    def covert_tag_entiy(self,tags_ids):
        sentences_entiys = []
        for sentences in tags_ids:
            entiy_i = []
            for w in sentences:
                entiy_i.append(self.entiy_tag[w])
            sentences_entiys.append(entiy_i)
        sentences_entiys = self.process_entity(sentences_entiys)
        return sentences_entiys

    def covert_polarity(self,tag_ids):
        sentences_entiys = []
        for sentences in tag_ids:
            entiy_i = []
            for w in sentences:
                entiy_i.append(self.poarity_tag[w])
            sentences_entiys.append(entiy_i)
        return sentences_entiys


    def covert_tag_postority(self,polarity):
        polarity_ids = []
        for s  in polarity:
            temp = []
            for t in s:
                p_t = self.poarity_tag[t]
                if p_t == -1:
                    continue
            polarity_ids.append(temp)
        return polarity_ids


    def get_word_embedding(self):
        return  self.embedding_vec


    def load_word_embedding(self,embedding_file=None,embedding_dim = 300):
        self.embedding_vec = np.random.uniform(-0.25, 0.25,size=[len(self.word2ids.keys()),embedding_dim])
        size = 0
        if embedding_file != None:
            embedding_dict = {}
            with open(embedding_file,encoding="utf-8") as fp:
                for l in tqdm(fp):
                    l = l.strip().split(" ")
                    word = l[0]
                    vec = list(map(float,l[1:]))
                    embedding_dict[word] = vec
                for i,word in enumerate(self.words):
                    try:
                        if word in embedding_dict.keys():
                            size+=1
                            self.embedding_vec[i] = embedding_dict[word]
                    except Exception as e:
                        pass
        print("load embedding form glove ",size)
        return self.embedding_vec


    def load_char_embedding(self,embedding_file=None):
        pass


    @property
    def vocab_size(self):
        return len(self.word2ids.keys())

    def save(self,path):
        with open(path,"wb") as f:
            pickle.dump(self,f)

    @staticmethod
    def load(path):
        with open(path,"rb") as f:
            v = pickle.load(f)
            return v

def build_vocab(file_list,sentimentes_file):
    v = Vocab(file_list = file_list,sentimentes_file=sentimentes_file)
    return v


# vocab = build_vocab(["../data/absa/rest_total_train.txt","../data/absa/rest_total_test.txt"])
# vocab.load_word_embedding(embedding_file="../../../glove.840B.300d.txt")
# vocab.save(path="./vocab.voc")
# vocab =Vocab.load("./vocab.voc")
# print(vocab.words)
# w,tag = read_train_data("../data/absa/rest_total_train.txt")
# print(v.covert_tag_entiy(tag))

def get_train_loader(records, batch_size = 32, mode="train"):
    max_length = 100
    max_length_c = 20
    max_spans_size = 5
    inputs = []
    inputs_c = []
    mask = []
    span = []
    polaryity = []
    polaryity_mask = []

    words =  records["words"]
    ids = records["inputs_ids"]
    idcs = records["char_ids"]
    for i ,w in enumerate(words):
        id_i = ids[i][:max_length]
        mask_i = [1 for _ in range(len(id_i))]
        idcs_i = []
        for j in range(len(id_i)):
            idc_i = idcs[i][j][:max_length_c]
            idc_i.extend([0 for _ in range(max_length_c-len(idc_i))])
            idcs_i.append(idc_i)

        for _ in range(max_length - len(id_i)):
            idcs_i.append([0 for _ in range(max_length_c)])

        mask_i.extend([0 for _ in range(max_length-len(id_i))])
        id_i.extend([0 for _ in range(max_length-len(id_i))])

        inputs.append(id_i)
        inputs_c.append(idcs_i)
        mask.append(mask_i)


    for i,s in enumerate(records["spans"]):
        s_i = s[:max_spans_size]
        for _ in range(max_spans_size-len(s_i)):
            s_i.append([0,0])
        span.append(s_i)

    for i,p in enumerate(records["polaryity"]):
        p_i = []
        p_mask_i = []
        for p_s in p[:max_spans_size]:
            p_i.append(polaryity_label[p_s])
            p_mask_i.append(1)
        for _ in range(max_spans_size-len(p_i)):
            p_i.append(0)
            p_mask_i.append(0)

        polaryity.append(p_i)
        polaryity_mask.append(p_mask_i)


    inputs = torch.LongTensor(inputs)
    inputs_c = torch.LongTensor(inputs_c)
    mask = torch.LongTensor(mask)
    spans = torch.LongTensor(span)
    polaryity = torch.LongTensor(polaryity)
    polaryity_mask = torch.ByteTensor(polaryity_mask)
    sentence_p = torch.LongTensor(records["sentence_p"])

    for i in range(2):
        print(inputs[i])
        print(mask[i])
        print(inputs_c[i])
        print(spans[i])
        print(polaryity[i])
        print(polaryity_mask[i])
        print(sentence_p[i])

    # inputs, inputs_c, mask, sub_span, polarity, mask_p, s_p

    train_data = TensorDataset(inputs,
                               inputs_c,
                               mask,
                               spans,
                               polaryity,
                               polaryity_mask,
                               sentence_p)
    if mode == "train":
        train_sampler = RandomSampler(train_data)

    else:
        train_sampler = SequentialSampler(train_data)

    dl = DataLoader(train_data,
                    sampler=train_sampler,
                    batch_size=batch_size)
    return dl


def split_records(rate,records):
    size = len(records["words"])
    ind = np.arange(0,size)
    dev_size = int(rate*size)
    np.random.seed(seed=2019)
    np.random.shuffle(ind)
    records_train = {}
    records_dev = {}
    for k in records.keys():
        records_dev[k]=[]
        records_train[k]=[]
        for i in ind[:dev_size]:
            records_dev[k].append(records[k][i])
        for j in ind[dev_size:]:
            records_train[k].append(records[k][j])

    print(records_dev)
    return records_train,records_dev

def build_data_helper(file,vocab,mode="train"):
    # read data
    records = read_data(file)

    inputs_ids = vocab.covert_word2ids(records["words"])
    char_ids = vocab.covert_char2ids(records["words"])
    records["inputs_ids"] = inputs_ids
    records["char_ids"] = char_ids
    if mode=="train":
        records_train ,records_dev = split_records(rate=0.1,records=records)
        dataloader_train = get_train_loader(records_train,mode=mode)
        dataloader_dev = get_train_loader(records_dev,mode="dev")
        return (dataloader_train,records_train),(dataloader_dev,records_dev)
    else:
        dataloader = get_train_loader(records, mode=mode)

    return dataloader,records

class DataHelper():

    def __init__(self,inputs_ids,
                 char_ids,
                 tag_ids,
                 tag_polarity,
                 sentiment_tag,
                 epoths=50,
                 spilt_train_dev=False):

        self.inputs_ids = inputs_ids
        self.char_ids = char_ids
        self.tag_ids = tag_ids
        self.tag_polarity = tag_polarity
        self.sentiment_tag = sentiment_tag
        self.epoths = epoths
        if spilt_train_dev:
            self.split_datasets()

    def split_datasets(self):
        n_dev = int(0.1*len(self.inputs_ids))
        # ids = np.random.choice(len(self.inputs_ids),n_dev,replace=False)
        ids = []
        for i in range(n_dev):
            ids.append(i)

        inputs_ids = []
        char_ids = []
        tag_ids = []
        tag_polarity = []
        sentiment_tag = []

        self.inputs_ids_dev = []
        self.char_ids_dev = []
        self.tag_ids_dev = []
        self.tag_polarity_dev = []
        self.sentiment_tag_dev = []
        for i in range(len(self.inputs_ids)):
            if i in ids:
                self.inputs_ids_dev.append(self.inputs_ids[i])
                self.char_ids_dev.append(self.char_ids[i])
                self.tag_ids_dev.append(self.tag_ids[i])
                self.tag_polarity_dev.append(self.tag_polarity[i])
                self.sentiment_tag_dev.append(self.sentiment_tag[i])
            else:
                inputs_ids.append(self.inputs_ids[i])
                char_ids.append(self.char_ids[i])
                tag_ids.append(self.tag_ids[i])
                tag_polarity.append(self.tag_polarity[i])
                sentiment_tag.append(self.sentiment_tag[i])
        self.inputs_ids = inputs_ids
        self.char_ids = char_ids
        self.tag_ids = tag_ids
        self.tag_polarity = tag_polarity
        self.sentiment_tag = sentiment_tag

    def Batch_data(self,batch_size=32):

        for i in range(self.epoths):
            for i,c,t,tp,st in self.epotch_batch(batch_size):
                yield i,c,t,tp,st

    def padding_batch_data(self,word_ids,char_ids,tag_ids,tag_polarity,sentiment_words):

        max_length = max([len(l) for l in word_ids])
        a = [max([len(c)  for c in l])for l in char_ids]
        max_length_c = np.max(a,axis=-1)

        # print(max_length_c)
        # print(max_length_c)
        word_ids_padding = []
        char_ids_padding = []
        tag_ids_padding = []
        tag_polarity_padding = []
        sentiment_tag_padding = []
        for l in word_ids:
            l.extend([0 for _ in range(max_length-len(l))])
            word_ids_padding.append(l)

        for l in tag_ids:
            l.extend(0 for _ in range(max_length-len(l)))
            tag_ids_padding.append(l)

        for l in tag_polarity:
            l.extend(0 for _ in range(max_length-len(l)))
            tag_polarity_padding.append(l)

        for example in char_ids:
            temp = []
            for w in example:
                w.extend([0 for _ in range(max_length_c-len(w))])
                temp.append(w)
            pad_c = [0 for _ in range(max_length_c)]
            for _ in range(max_length-len(temp)):
                temp.append(pad_c)
            char_ids_padding.append(temp)

        for l in sentiment_words:
            l.extend(0 for _ in range(max_length-len(l)))
            sentiment_tag_padding.append(l)

        return word_ids_padding,char_ids_padding,tag_ids_padding,tag_polarity_padding,sentiment_tag_padding


    def epotch_batch(self,batch_size=32):
        data_size = self.get_data_size
        for j in range(0,data_size,batch_size):
            yield self.padding_batch_data(self.inputs_ids[j:j+batch_size],
                                          self.char_ids[j:j+batch_size],
                                          self.tag_ids[j:j+batch_size],
                                          self.tag_polarity[j:j+batch_size],
                                          self.sentiment_tag[j:j+batch_size])


    def dev_data(self,batch_size=32):
        dev_data_size = len(self.inputs_ids_dev)
        for j in range(0,dev_data_size,batch_size):
            yield self.padding_batch_data(self.inputs_ids_dev[j:j+batch_size],
                                          self.char_ids_dev[j:j+batch_size],
                                          self.tag_ids_dev[j:j+batch_size],
                                          self.tag_polarity_dev[j:j+batch_size],
                                          self.sentiment_tag_dev[j:j+batch_size])

    @property
    def get_data_size(self):
        return len(self.inputs_ids)
















