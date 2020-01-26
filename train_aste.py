import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from until import  Vocab
from data_process import build_data,build_vocab
# from data_process import polaryity_label
import numpy as np
import torch
import torch.nn as nn
import warnings

from model_aste import ASTE

warnings.filterwarnings("ignore")


# for entity
def tags2spans(entits):
    ret = []
    for sample in entits:
        temp = []
        beg, end = -1, -1
        for i, t in enumerate(sample):
            if t == 4:
                temp.append((i, i))
            elif t == 1:
                beg = i
            elif t == 3:
                end = i
                if end > beg > -1:
                    temp.append((beg, end))
                    beg, end = -1, -1
        ret.append(temp)

    return ret


# for polarity
def tags2polarity(entits, spans):
    res = []
    for i, spans in enumerate(spans):
        temp = []
        for span in spans:
            beg = span[0]
            end = span[1]
            if len(set(entits[i][beg:end + 1])) > 1:
                # print("error ")
                pass
            else:
                temp.append((beg, end, entits[i][beg]))
        res.append(temp)
    return res


def evaluate_entity(gold, pred):
    pred_entity = [[np.argmax(l) for l in sample] for sample in pred]
    pred_entity = tags2spans(pred_entity)
    gold_entity = tags2spans(gold)

    pred_total = 0
    glob_total = 0
    match_ot = 0

    for i in range(len(pred_entity)):

        glob_total += len(gold_entity[i])
        pred_total += len(pred_entity[i])

        for t in pred_entity[i]:
            if t in gold_entity[i]:
                match_ot += 1

    # print("eneity match ",match_ot)
    # print("pred entity total ",pred_total)
    # print("glod entity total ", glob_total)

    p = match_ot / (pred_total + 0.001)
    r = match_ot / (glob_total + 0.001)
    f1 = 2 * p * r / (p + r + 0.001)
    print("p:", p, "r:", r, "f1:", f1)
    return p, r, f1


def evaluate_polarity(gold, pred, gold_entity, pred_entity):
    pred = [[np.argmax(l) for l in sample] for sample in pred]
    gold_polarity = tags2polarity(gold, gold_entity)
    pred_polarity = tags2polarity(pred, pred_entity)

    pred_total_p = 0
    glob_total_p = 0
    match_ot_p = 0

    for i in range(len(pred_polarity)):

        glob_total_p += len(gold_polarity[i])
        pred_total_p += len(pred_polarity[i])

        for t in pred_polarity[i]:
            if t in gold_polarity[i]:
                match_ot_p += 1

    # print("eneity match ",match_ot_p)
    # print("pred entity total ",pred_total_p)
    # print("glod entity total ", glob_total_p)

    p = match_ot_p / (pred_total_p + 0.001)
    r = match_ot_p / (glob_total_p + 0.001)
    f1 = 2 * p * r / (p + r + 0.001)
    # print("p:",p, "r:",r, "f1:",f1)
    return p, r, f1


def evaluate(model, data_helper):
    with torch.no_grad():
        pred_entity = []
        pred_polarity = []
        gold_entity = []
        gold_polarity = []
        for w, c, t, p, pw in data_helper:
            w = torch.LongTensor(w).to(device)
            c = torch.LongTensor(c).to(device)

            # t = torch.FloatTensor(t).to(device)
            p_e, p_p, p_w = model(w, c)
            logist = torch.softmax(p_e, dim=-1).tolist()
            logist_p = torch.softmax(p_p, dim=-1).tolist()

            pred_entity.extend(logist)
            pred_polarity.extend(logist_p)

            gold_polarity.extend(p)
            gold_entity.extend(t)

    evaluate_entity(gold=gold_entity, pred=pred_entity)

    pred_entity = [[np.argmax(l) for l in sample] for sample in pred_entity]
    pred_entity = tags2spans(pred_entity)
    gold_entity = tags2spans(gold_entity)
    p, r, f1 = evaluate_polarity(gold=gold_polarity, pred=pred_polarity, gold_entity=gold_entity,
                                 pred_entity=pred_entity)
    return p, r, f1


def run_train_epoch(model, data_loader, optimizer):
    model.train()
    running_loss, count = 0, 0
    words = data_loader["inputs_ids"]
    char_ids = data_loader["char_ids"]
    otes = data_loader["ote"]
    ote_tes = data_loader["ote_te"]
    oe_alls = data_loader["oe_all"]
    oe_splits = data_loader["oe"]
    spans = data_loader["spans"]
    polaryitys = data_loader["polaryity"]
    n_train = len(words)

    # for step, batch in enumerate(data_loader):
    for i in range(n_train):
        x = words[i]
        length = len(x)
        inputs = torch.tensor([x])
        inputs_c = None
        mask = torch.FloatTensor([[1 for _ in range(length)]])
        sub_span = torch.tensor([spans[i]])
        polarity = torch.tensor([polaryitys[i]])
        mask_p = torch.FloatTensor([[1 for _ in range(len(spans[i]))]])
        ote = torch.tensor([otes[i]])
        te = None
        ote_te = torch.tensor([ote_tes[i]])
        oe = torch.tensor([oe_alls[i]])
        oe_split = torch.tensor([oe_splits[i]])

        loss = model(inputs, inputs_c, mask, spans=sub_span, polarity=polarity,
                     polarity_mask=mask_p, ote = ote, te=te,ote_te=ote_te,oe=oe,oe_split=oe_split)
        # loss.backward()
        running_loss += loss.item()
        optimizer.step()
        model.zero_grad()

    print("step:", n_train,
          "loss:", running_loss / n_train)


ote_model = {"O":0,"B":1,"I":2,"E":3,"S":4}
te_model = {"O":0,"POS":1,"NEG":2,"NEU":3}

def cacluate_ote(ote_pred,te_pred,spans,te):

    hit = 0
    hit_ote_te = 0
    count_pred = 0
    count_real = 0
    count_te = 0
    for i,s in enumerate(ote_pred):
        spans_i_pred = []
        beg = -1
        for j,w in enumerate(s):
            if w == 4:
                spans_i_pred.append([j,j])
            elif w == 1:
                beg = j
            elif w==3:
                end = j
                if beg!=-1:
                    spans_i_pred.append([beg, end])
                beg=-1
                end=-1


        for s in spans_i_pred:
            for s_ in spans[i]:
                if s==s_:
                    hit+=1

        count_pred += len(spans_i_pred)
        count_real += len(spans[i])
    for i,t in enumerate(te):
        for j,t_j in enumerate(t[:5]):
            if t_j==te_pred[i][j]:
                hit_ote_te += 1
        count_te+=len(t)
    print("hit_ote ",hit)
    print("hit_te",hit_ote_te)
    print("count_pred ",count_pred)
    print("count_real ",count_real)

    p=hit/(count_pred+0.001)
    r=hit/(count_real+0.001)
    f1 = (2*p*r)/(p+r+0.001)
    print("p ",p,"r ",r,"f1 ",f1)

    acc = hit_ote_te/count_te
    print("acc ",acc)

    return f1

union_tag = {"O":0,
             "B-POS":1,"I-POS":2,"E-POS":3,"S-POS":4,
             "B-NEG": 5, "I-NEG": 6, "E-NEG": 7, "S-NEG": 8,
             "B-NEU": 9, "I-NEU": 10, "E-NEU": 11, "S-NEU": 12
             }

def cacluate_ote_te(ote_pred,spans,polarity):

    hit = 0
    hit_ote_te = 0
    count_pred = 0
    count_real = 0
    for i,s in enumerate(ote_pred):
        beg = -1
        p = -1
        spans_i_pred = []
        spans_i_true = []

        for j,w in enumerate(s):
            w = int(w)
            if w % 4 == 1:
                beg = j
                p = int(w/4)+1

            if w % 4 == 3:
                p_e = int(w/4)+1
                if beg!=-1 and p_e == p:
                    spans_i_pred.append([beg,j,p])
                    beg = -1
                    p=-1

            if w % 4==0 and w!=0:
                p = int(w/4)
                spans_i_pred.append([j,j,p])

        for j,s in enumerate(spans[i]):
            s_c = s.copy()
            s_c.append(polarity[i][j]+1)
            spans_i_true.append(s_c)

        for s in spans_i_pred:
            if s in spans_i_true:
                hit_ote_te+=1
        count_pred+= len(spans_i_pred)
        count_real+= len(spans_i_true)
        # print(spans_i_pred)
        # print(spans_i_true)
    print(hit_ote_te)
    print(count_pred)
    print(count_real)
    p=hit_ote_te/(count_pred+0.001)
    r=hit_ote_te/(count_real+0.001)
    f1 = (2*p*r)/(p+r+0.001)
    print("p ",p,"r ",r,"f1 ",f1)

    return f1


def cacluate_ote_te_2(span_pred, te, oppiton, spans, polarity,oppiton_t):


    hit = 0
    hit_ote_te = 0
    hit_ote_te_o = 0
    hit_ote_te_o_truple = 0

    count_pred = 0
    count_real = 0
    count_pred_o = 0
    count_real_o = 0

    for i,s in enumerate(span_pred):
        spans_i_pred = []
        spans_i_true = []
        for j,t in enumerate(s[:5]):
            spans_i_pred.append([t[0], t[1], te[i][j]])
            if t in spans[i]:
                hit += 1

        for j,s in enumerate(spans[i]):
            s_c = s.copy()
            s_c.append(polarity[i][j])
            spans_i_true.append(s_c)

        count_pred += len(spans_i_pred)
        count_real += len(spans_i_true)

        op = oppiton[i]
        op_i_pred = []
        op_i_pred_truple = []
        for ii in range(len(spans_i_pred[:5])):
            spans_i = spans_i_pred[ii]
            beg = -1
            for j, w_i in enumerate(op[ii]):
                if w_i == 4:
                    op_i_pred.append([spans_i[0],spans_i[1],j,j])
                    op_i_pred_truple.append([spans_i[0],spans_i[1],spans_i[2],j,j])
                elif w_i == 1:
                    beg = j
                elif w_i == 3:
                    if beg != -1:
                        op_i_pred.append([spans_i[0],spans_i[1],beg, j])
                        op_i_pred_truple.append([spans_i[0], spans_i[1], spans_i[2], beg, j])

                    beg = -1


        op_t = oppiton_t[i]
        op_i_true = []
        op_i_true_truple = []
        for ii in range(len(spans_i_true)):
            spans_i = spans_i_true[ii]
            beg = -1
            for j, w_i in enumerate(op_t[ii]):
                if w_i == 4:
                    op_i_true.append([spans_i[0],spans_i[1],j,j])
                    op_i_true_truple.append([spans_i[0],spans_i[1],spans_i[2],j,j])
                elif w_i == 1:
                    beg = j
                elif w_i == 3:
                    if beg != -1:
                        op_i_true.append([spans_i[0],spans_i[1],beg,j])
                        op_i_true_truple.append([spans_i[0], spans_i[1], spans_i[2], beg, j])

                    beg = -1

        for i,s in enumerate(spans_i_pred):
            for j,s_j in enumerate(spans_i_true):
                if s == s_j:
                    hit_ote_te += 1

        for i, op_i in enumerate(op_i_pred):
            if op_i in op_i_true:
                    hit_ote_te_o += 1

        for i, op_i in enumerate(op_i_pred_truple):
            if op_i in op_i_true_truple:
                hit_ote_te_o_truple += 1


        count_pred_o+=len(op_i_pred)
        count_real_o+=len(op_i_true)

    p=hit/(count_pred+0.001)
    r=hit/(count_real+0.001)
    f1 = (2*p*r)/(p+r+0.001)
    print("ATE: ","p ",p,"r ",r,"f1 ",f1)
    p=hit_ote_te/(count_pred+0.0001)
    r=hit_ote_te/(count_real+0.0001)
    f1 = (2*p*r)/(p+r+0.0001)
    print("ATE+ATC: ","p ",p,"r ",r,"f1 ",f1)

    p = hit_ote_te_o/(count_pred_o+0.0001)
    r = hit_ote_te_o/(count_real_o+0.0001)
    f1 = (2*p*r)/(p+r+0.0001)
    print("ATE+OTE: ","p ",p,"r ",r,"f1 ",f1)

    p = hit_ote_te_o_truple/(count_pred_o+0.0001)
    r = hit_ote_te_o_truple/(count_real_o+0.0001)
    f1 = (2*p*r)/(p+r+0.0001)
    print("ATE+ATC+OTE: ","p ",p,"r ",r,"f1 ",f1)

    return f1

best_f1 = 0
def evaluate(model, data_loader, features_test):
    global best_f1
    model.eval()
    with torch.no_grad():

        te_pred = []
        spans_preds = []
        opinion = []
        opinion_t = []

        words = data_loader["inputs_ids"]
        oe_splits = data_loader["oe"]
        spans = data_loader["spans"]
        n_train = len(words)

        # for step, batch in enumerate(data_loader):
        for i in range(n_train):
            x = words[i]
            length = len(x)
            inputs = torch.tensor([x])
            inputs_c = None
            mask = torch.FloatTensor([[1 for _ in range(length)]])
            oe_split = torch.tensor([oe_splits[i]])
            span = torch.tensor([spans[i]])
            mask_p = torch.FloatTensor([[1 for _ in range(len(spans[i]))]])
            spans_pred = []
            span_mask = []
            # inputs, inputs_c, mask, sub_span, polarity, mask_p, ote,te,ote_te,oe, oe_split= batch

            ote_pred_i, te_pred_i,_ = model(inputs, inputs_c, mask,  span, polarity_mask=mask_p, mode="polarity")

            for i, s in enumerate(ote_pred_i[1]):
                spans_i_pred = []
                beg = -1
                for j, w in enumerate(s):
                    if w == 4:
                        spans_i_pred.append([j, j])
                    elif w == 1:
                        beg = j
                    elif w == 3:
                        end = j
                        if beg != -1:
                            spans_i_pred.append([beg, end])
                        beg = -1
                spans_preds.append(spans_i_pred.copy())
                spans_i_pred = spans_i_pred[:5]
                span_mask_i = [1 for _ in range(len(spans_i_pred))]
                # spans_i_pred.extend([[0, 0] for _ in range(5-len(spans_i_pred))])
                # span_mask_i.extend([0 for _ in range(5-len(span_mask_i))])
                spans_pred.append(spans_i_pred)
                span_mask.append(span_mask_i)
            if len(spans_pred[0])==0:
                opinion.append([])
                te_pred.append([])
            else:
                _, te_pred_i, oppiton_i = model(inputs, inputs_c, mask,
                                   spans=torch.LongTensor(spans_pred),
                                   polarity_mask=torch.FloatTensor(span_mask),
                                   mode="polarity")

                opinion.extend(oppiton_i[1])
                te_pred.extend(te_pred_i[1])
            opinion_t.extend(oe_split)

        f1 = cacluate_ote_te_2(spans_preds, te_pred, opinion,
                               features_test["spans"],features_test["polaryity"],features_test["oe"])

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.model_path)

    return f1


def calcuate_polarity(features, polarity_pred, s_p_preds=None):
    polaryity = features["polaryity"]
    hit = 0
    true_count = 0
    size = len(polarity_pred)
    y_true = []
    y_pred = []

    for i in range(size):
        s_i = s_p_preds[i]

        for j, s in enumerate(polaryity[i][:5]):
            y_true.append(s)
            y_pred.append(polarity_pred[i][j].item())
            if s == polarity_pred[i][j].item():
                hit += 1

        for j, s in enumerate(polaryity[i][5:]):
            y_true.append(s)
            y_pred.append(-1)

        true_count += len(polaryity[i])

    print("hit", hit)
    print("count", true_count)
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    acc = hit / (true_count + 0.001)

    print("f1", macro_f1, "acc", acc)
    return macro_f1, acc


parse = argparse.ArgumentParser()

parse.add_argument("--mode", default="train", type=str)
parse.add_argument("--model_path", default="./model/best_model_14res.pt", type=str)
parse.add_argument("--train_file", default="data/triplet_data_only/14res/train.txt", type=str)
parse.add_argument("--dev_file", default="data/triplet_data_only/14res/dev.txt", type=str)
parse.add_argument("--test_file", default="data/triplet_data_only/14res/test.txt", type=str)
parse.add_argument("--vocab", default="./14res_vocab", type=str)
parse.add_argument("--embedding_file", default=None, type=str)



args = parse.parse_args()


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.vocab):
        vocab = build_vocab([args.train_file,args.dev_file, args.test_file])
        embedding_mat = vocab.load_word_embedding(embedding_file=args.embedding_file)
        vocab.save(args.vocab)
    else:
        vocab = Vocab.load(args.vocab)
        embedding_mat = vocab.get_word_embedding()

    data_helper, features_train = build_data(args.train_file, vocab, mode="train")
    data_helper_dev, features_dev = build_data(args.dev_file, vocab, mode="test")
    data_helper_test, features_test = build_data(args.test_file, vocab, mode="test")

    model_path = args.model_path

    num_embeddings = vocab.vocab_size
    num_embeddings_c = len(vocab.char2ids.keys())
    embedding_dim = 300
    rnn_1_dim = 100
    entiy_size = len(vocab.entiy_tag_bie.keys())
    polarity_size = len(vocab.poarity_tag)

    model = ASTE(num_embeddings=num_embeddings,
                num_embedding_c=num_embeddings_c,
                embedding_dim=embedding_dim,
                embedding_dim_c=50,
                rnn_1_dim=rnn_1_dim,
                label_size=entiy_size,
                polarity_size=polarity_size,
                word_vec=embedding_mat)
    model = model.to(device)
    model = nn.DataParallel(model)
    init_lr = 0.1
    optim = torch.optim.SGD(model.parameters(),lr=init_lr)

    model.train()
    num_train_epoch =60
    decay_rate = 0.02
    best_ = 0.0
    if args.mode=="train":
        for i in range(num_train_epoch):
            cur_lr = init_lr / (1 + decay_rate * i)
            print("***** epoch {}  lr{}******".format(i,cur_lr))
            run_train_epoch(model, features_train, optim)
            print("********evaluate********")
            f1 = evaluate(model, features_dev, features_dev)

    else:
        # model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.sentiment_model_path))
        evaluate(model, features_test, features_test)
