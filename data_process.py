import torch

from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader

from until import Vocab


ote_model = {"O":0,"B":1,"I":2,"E":3,"S":4}
te_model = {"O":0,"POS":1,"NEG":2,"NEU":3}
polarity_map = {"POS":0,"NEG":1,"NEU":2}
union_tag = {"O":0,
             "B-POS":1,"I-POS":2,"E-POS":3,"S-POS":4,
             "B-NEG": 5, "I-NEG": 6, "E-NEG": 7, "S-NEG": 8,
             "B-NEU": 9, "I-NEU": 10, "E-NEU": 11, "S-NEU": 12
             }


def process_tag(file):
    words = []
    ote = []
    te = []
    ote_te = []
    spans = []
    polarity = []
    apsect_words = []

    with open(file,encoding="utf-8") as fp:
        for l in fp:
            words_i = []
            ote_i = []
            te_i = []
            ote_te_i = []
            spans_i = []
            apsect_words_i = []
            polarity_i = []

            words_tag = l.strip().split("####")
            prev = None
            beg = -1
            for i,p in enumerate(words_tag[1].split(" ")):
                w_t = p.split("=")
                words_i.append(w_t[0].lower())

                t = w_t[1]
                if t == "O":
                    if prev != None:
                        polarity_i.append(polarity_map[prev.split("-")[1]])
                        spans_i.append([beg, i - 1])
                        if ote_i[-1]==ote_model["B"]:
                            ote_i[-1] = ote_model["S"]
                            ote_te_i[-1 ] = union_tag["S-"+prev.split("-")[1]]
                        else:
                            ote_i[-1] = ote_model["E"]
                            ote_te_i[-1 ] = union_tag["E-"+prev.split("-")[1]]

                        apsect_words_i.append(words_i[beg:i])
                        prev=None


                    ote_i.append(ote_model["O"])
                    te_i.append(te_model["O"])
                    ote_te_i.append(union_tag["O"])

                else:
                    t_p = t
                    if prev == None:
                        beg = i
                        prev = t_p
                        ote_i.append(ote_model["B"])
                        ote_te_i.append(union_tag["B-"+t_p.split("-")[1]])
                    else:
                        if prev!= t_p:
                            polarity_i.append(polarity_map[prev.split("-")[1]])
                            spans_i.append([beg, i - 1])
                            if ote_i[-1]==ote_model["B"]:
                                ote_i[-1] = ote_model["S"]
                            else:
                                ote_i[-1] = ote_model["E"]
                            apsect_words_i.append(words_i[beg:i])
                            # prev = None

                            beg = i
                            prev = t_p
                            ote_i.append(ote_model["B"])
                            ote_te_i.append(union_tag["B-" + t_p.split("-")[1]])
                        else:
                            ote_i.append(ote_model["I"])
                            ote_te_i.append(union_tag["I-" + t_p.split("-")[1]])

                    te_i.append(te_model[t_p.split("-")[1]])

            if prev!=None:
                polarity_i.append(polarity_map[prev.split("-")[1]])
                spans_i.append([beg, i])
                if ote_i[-1]==ote_model["B"]:
                    ote_i[-1] = ote_model["S"]
                    ote_te_i[-1] = union_tag["S-" + prev.split("-")[1]]

                else:
                    ote_i[-1] = ote_model["E"]
                    ote_te_i[-1] = union_tag["E-" + prev.split("-")[1]]

            words.append(words_i)
            ote.append(ote_i)
            te.append(te_i)
            spans.append(spans_i)
            polarity.append(polarity_i)
            apsect_words.append(apsect_words_i)
            ote_te.append(ote_te_i)

    return words,ote,te,ote_te,spans,polarity,apsect_words


def process_option(file,polarity):

    opption = []
    opption_all = []
    count = 0
    with open(file,encoding="utf-8") as fp:
        for index,l in enumerate(fp):
            words_tag = l.strip().split("####")
            opption_all_i = []
            size = len(polarity[index])
            words_i = []
            for i,p in enumerate(words_tag[2].split(" ")):
                w_t = p.split("=")
                words_i.append(w_t[0])

            opption_i = [[0 for _ in range(len(words_i))] for _ in range(size)]

            for i,p in enumerate(words_tag[2].split(" ")):
                w_t = p.split("=")
                if w_t[1]!="O":
                    opption_i[len(w_t[1])-1][i] = 1
                    opption_all_i.append(1)
                else:
                    opption_all_i.append(0)

            for s in opption_i:
                prev = False
                for i,w in enumerate(s):
                    if w==0:
                        if prev==True:
                            if s[i-1] == ote_model["B"]:
                                s[i-1] = ote_model["S"]
                            else:
                                s[i-1] = ote_model["E"]
                            prev = False
                            count+=1

                    else:
                        if prev==False:
                            s[i] = ote_model["B"]
                            prev = True
                        else:
                            s[i] = ote_model["I"]
                if prev:
                    if s[i]==ote_model["B"]:
                        s[i] = ote_model["S"]
                    else:
                        s[i] = ote_model["E"]
                    count += 1

            opption.append(opption_i)
            opption_all.append(opption_all_i)
        print(count)
    return opption,opption_all


def covert_features(data_set, tokenizer):

    words = data_set["words"]
    subwords = []
    index_map = []
    index_map_orig = []

    sub_span = []
    for example in words:
        subwords_i = []
        index_map_i = {}
        index_map_orig_i = {}
        for i, w in enumerate(example):
            index_map_i[i] = len(subwords_i)
            subwords_i.extend(tokenizer.tokenize(w))
            index_map_orig_i[len(subwords_i)] = i

        index_map_i[i + 1] = len(subwords_i)
        index_map_orig_i[len(subwords_i)] = i + 1

        index_map_orig.append(index_map_orig_i)
        index_map.append(index_map_i)
        subwords.append(subwords_i)

    data_set["subwords"] = subwords
    data_set["index_map"] = index_map
    data_set["index_map_orig"] = index_map_orig

    for i, span in enumerate(data_set["spans"]):
        sub_span_i = []
        for s in span:
            sub_span_i.append([index_map[i][s[0]], index_map[i][s[1] + 1] - 1])
        sub_span.append(sub_span_i)
    data_set["sub_span"] = sub_span

    return data_set

def get_train_loader(records, batch_size=32, mode="train"):
    max_length = 100
    max_length_c = 20
    max_spans_size = 5
    inputs = []
    inputs_c = []
    mask = []
    span = []
    polaryity = []
    polaryity_mask = []
    ote = []
    te = []
    ote_te = []
    oe_all = []
    oe = []

    words = records["words"]
    ids = records["inputs_ids"]
    idcs = records["char_ids"]
    otes = records["ote"]
    tes = records["te"]
    ote_tes = records["ote_te"]
    oe_alls = records["oe_all"]
    oes = records["oe"]

    for i, w in enumerate(words):
        id_i = ids[i][:max_length]
        ote_i = otes[i][:max_length]
        te_i = tes[i][:max_length]
        ote_te_i = ote_tes[i][:max_length]
        oe_all_i = oe_alls[i][:max_length]
        mask_i = [1 for _ in range(len(id_i))]
        idcs_i = []
        oe_i = oes[i][:max_spans_size]

        for j,oe_i_i in enumerate(oe_i):
            oe_i_i = oe_i_i[:100]
            oe_i_i.extend([0 for _ in range(max_length-len(oe_i_i))])
            oe_i[j] = oe_i_i
        oe_i.extend([[0 for _ in range(max_length)] for _ in range(max_spans_size-len(oe_i))])

        for j in range(len(id_i)):
            idc_i = idcs[i][j][:max_length_c]
            idc_i.extend([0 for _ in range(max_length_c - len(idc_i))])
            idcs_i.append(idc_i)

        for _ in range(max_length - len(id_i)):
            idcs_i.append([0 for _ in range(max_length_c)])

        mask_i.extend([0 for _ in range(max_length - len(id_i))])
        id_i.extend([0 for _ in range(max_length - len(id_i))])
        ote_i.extend([0 for _ in range(max_length-len(ote_i))])
        te_i.extend([0 for _ in range(max_length-len(te_i))])
        ote_te_i.extend([0 for _ in range(max_length-len(ote_te_i))])
        oe_all_i.extend([0 for _ in range(max_length-len(oe_all_i))])
        inputs.append(id_i)
        inputs_c.append(idcs_i)
        mask.append(mask_i)
        ote.append(ote_i)
        te.append(te_i)
        ote_te.append(ote_te_i)
        oe_all.append(oe_all_i)
        oe.append(oe_i)

    for i, s in enumerate(records["spans"]):
        s_i = s[:max_spans_size]
        for _ in range(max_spans_size - len(s_i)):
            s_i.append([0, 0])
        span.append(s_i)

    for i, p in enumerate(records["polaryity"]):
        p_i = []
        p_mask_i = []
        for p_s in p[:max_spans_size]:
            p_i.append(p_s)
            p_mask_i.append(1)
        for _ in range(max_spans_size - len(p_i)):
            p_i.append(0)
            p_mask_i.append(0)

        polaryity.append(p_i)
        polaryity_mask.append(p_mask_i)

    inputs = torch.LongTensor(inputs)
    inputs_c = torch.LongTensor(inputs_c)
    mask = torch.ByteTensor(mask)
    spans = torch.LongTensor(span)
    polaryity = torch.LongTensor(polaryity)
    polaryity_mask = torch.ByteTensor(polaryity_mask)
    ote = torch.LongTensor(ote)
    te = torch.LongTensor(te)
    ote_te = torch.LongTensor(ote_te)
    oe_all = torch.LongTensor(oe_all)
    oe = torch.LongTensor(oe)


    for i in range(2):
        print(words[i])
        print(inputs[i])
        print(mask[i])
        print(inputs_c[i])
        print(spans[i])
        print(polaryity[i])
        print(polaryity_mask[i])
        print(ote[i])
        print(te[i])
        print(ote_te[i])
        print(oe_all[i])
        print(oe[i])

    # inputs, inputs_c, mask, sub_span, polarity, mask_p, s_p

    train_data = TensorDataset(inputs,
                               inputs_c,
                               mask,
                               spans,
                               polaryity,
                               polaryity_mask,
                               ote,
                               te,
                               ote_te,
                               oe_all,
                               oe)
    if mode=="train":
        train_sampler = RandomSampler(train_data)

    else:
        train_sampler = SequentialSampler(train_data)

    dl = DataLoader(train_data,
                    sampler=train_sampler,
                    batch_size=batch_size)
    return dl


def get_train_loader_bert(features, tokenizer, mode,batch_size=32,max_length=100):

    inputs = []
    seg = []
    mask = []
    inputs_subword = []
    ote = []
    sub_spans = []
    polaryity = features["polaryity"]
    oe = []
    oe_all = []

    # data_set["oe"] = data_set["oe"]
    # data_set["oe_all"] = data_set["oe_all"]

    for i, subwords in enumerate(features["subwords"]):

        inputs_subword_i = []
        inputs_subword_i.append("[CLS]")
        inputs_subword_i.extend(subwords)
        inputs_subword_i.append("[SEP]")

        inputs_ids = tokenizer.convert_tokens_to_ids(inputs_subword_i)
        input_mask = [1] * len(inputs_ids)
        seg_ids = [0] * len(inputs_ids)

        ote_id = [0] * len(inputs_ids)
        span = features["sub_span"][i]
        sub_spans_i = []

        for s in span:
            beg = s[0] + 1
            end = s[1] + 1
            sub_spans_i.append([beg, end])

            if end==beg:
                ote_id[beg] = ote_model["S"]
            else:
                ote_id[beg] = ote_model["B"]
                ote_id[end] = ote_model["E"]
                for k in range(beg + 1, end):
                    ote_id[k] = ote_model["I"]


        # polaryity_i = [te_model[p] for p in features["polaryity"][i]]
        # assert len(sub_spans_i)==len(polaryity_i)

        inputs.append(inputs_ids)
        seg.append(seg_ids)
        mask.append(input_mask)
        inputs_subword.append(inputs_subword_i)
        ote.append(ote_id)
        sub_spans.append(sub_spans_i)
        # polaryity.append(features["polaryity"][i])

    print(inputs)

    inputs = torch.LongTensor(inputs)
    seg = torch.LongTensor(seg)
    mask = torch.LongTensor(mask)
    ote = torch.LongTensor(ote)
    sub_spans = torch.LongTensor(sub_spans)
    polaryity = torch.LongTensor(polaryity)


    for i in range(2):
        print(inputs_subword[i])
        print(inputs[i])
        print(seg[i])
        print(mask[i])
        print(ote[i])
        print(polaryity[i])


    train_data = TensorDataset(inputs,
                               seg,
                               mask,
                               ote,
                               polaryity)
    if mode=="train":
        train_sampler = RandomSampler(train_data)
        # train_sampler =SequentialSampler(train_data)

    else:
        train_sampler = SequentialSampler(train_data)

    dl = DataLoader(train_data,
                    sampler=train_sampler,
                    batch_size=batch_size)
    return dl

def read_file(file):
    words, ote, te,ote_te, spans, polarity, apsect_words = process_tag(file)
    opption,opption_all = process_option(file,polarity)
    count = 0
    for p in polarity:
        count+=len(p)
    records = {}
    records["words"] = words
    records["polaryity"] = polarity
    records["ote"] = ote
    records["spans"] = spans
    records["te"] = te
    records["ote_te"] = ote_te
    records["oe_all"] = opption_all
    records["oe"] = opption
    return records




def build_data(file,vocab,mode):
    records = read_file(file)
    inputs_ids = vocab.covert_word2ids(records["words"])
    char_ids = vocab.covert_char2ids(records["words"])
    records["inputs_ids"] = inputs_ids
    records["char_ids"] = char_ids
    dataloader = get_train_loader(records, mode=mode,batch_size=1)
    return dataloader,records


def build_vocab(file_list):
    v = Vocab(file_list)
    return v
