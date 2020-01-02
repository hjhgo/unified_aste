
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class BertSelfAttention(nn.Module):
    def __init__(self,hidden_size,num_attention_head):
        super(BertSelfAttention,self).__init__()

        self.hidden_size = hidden_size
        self.num_attention_head = num_attention_head
        self.attention_hidden_size = int(self.hidden_size/self.num_attention_head)
        self.all_head_size = self.num_attention_head * self.attention_hidden_size
        self.key = nn.Linear(self.hidden_size,self.all_head_size)
        self.query = nn.Linear(self.hidden_size,self.all_head_size)
        self.value = nn.Linear(self.hidden_size,self.all_head_size)

    def transponse_for_score(self,inputs):
        new_view_shape = inputs.size()[:-1]+(self.num_attention_head,self.attention_hidden_size)
        inputs = inputs.view(*new_view_shape)
        return inputs.permute(0,2,1,3)

    def forward(self,
                inputs,
                attention_mask=None,
                hard_mask=None):

        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        query = self.transponse_for_score(query)
        key = self.transponse_for_score(key)
        value = self.transponse_for_score(value)

        scores = torch.matmul(query,key.transpose(-1,-2))

        scores = scores.softmax(dim=-1)

        content_value = torch.matmul(scores,value).permute(0,2,1,3).contiguous()
        new_content_shape = content_value.size()[:-2]+(self.all_head_size,)
        content_value = content_value.view(*new_content_shape)

        return content_value

class BertOutput(nn.Module):
    def __init__(self,hidden_size):
        super(BertOutput,self).__init__()

        self.dense = nn.Linear(hidden_size,hidden_size)
        self.layer_normal = nn.LayerNorm(hidden_size)
        self.drop_out =nn.Dropout(0.2)

    def forward(self, state,input):
        x = self.dense(state)
        x = self.drop_out(x)
        x = self.layer_normal(x+input)
        return x




def seq_tag_loss(pred, position, mask=None):
    if mask is None:
        pred_logsoftmax = torch.log_softmax(pred, dim=-1)
        loss = -1 * torch.mean(torch.gather(pred_logsoftmax, index=torch.unsqueeze(position, dim=-1), dim=2))
    else:
        pred_logsoftmax = torch.log_softmax(pred, dim=-1)
        loss = torch.gather(pred_logsoftmax, index=torch.unsqueeze(position, dim=-1), dim=2).squeeze(2). \
            masked_select(mask)
        loss = -1 * torch.mean(loss)

    return loss

# 计算注意力机制 长度相关
class Atte_module(nn.Module):
    def __init__(self,
                 input_hidden_size,
                 hidden_size):
        super(Atte_module, self).__init__()

        self.length_embedding = nn.Embedding(num_embeddings = 100,
                                             embedding_dim = input_hidden_size)
        self.att_liner = nn.Linear(input_hidden_size,1)
        self.att2_liner = nn.Linear(input_hidden_size,5)

        self.drop_out = nn.Dropout(p=0.2)

        def init_weight(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight.data)

        self.apply(init_weight)

    # TODO 初始化
    def forward(self, q, k, m, length_mask=None, attention_mask=None):

        q = q.unsqueeze(dim=2)
        k = k.unsqueeze(dim=1)
        if length_mask is not None:
            length_embd = self.length_embedding(length_mask)
            k = k.expand(size=[k.size()[0],
                               q.size()[1],
                               k.size()[2],
                               k.size()[3]])
            k = k + length_embd

        attention_scores = torch.squeeze(self.att_liner(q+k), dim=-1)
        scores = nn.Softmax(dim=-1)(self.att2_liner(q+k))

        if attention_mask is not None:
            attention_scores *= attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.drop_out(attention_probs)

        attention_content = torch.matmul(attention_probs, m)

        return attention_content, scores

class ASTE_SPAN(nn.Module):

    def __init__(self,
                 num_embeddings,
                 num_embedding_c,
                 embedding_dim,
                 embedding_dim_c,
                 rnn_1_dim,
                 entiy_size,
                 polarity_size,
                 word_vec = None,
                 use_char = False):

        super(ASTE_SPAN, self).__init__()

        self.use_char = use_char
        self.word_embed = nn.Embedding(num_embeddings=num_embeddings
                                       ,embedding_dim=embedding_dim)
        if word_vec is not None:
            self.word_embed.weight.data.copy_(torch.from_numpy(word_vec))
            self.word_embed.weight.requires_grad = True

        if self.use_char:
            self.char_embed = nn.Embedding(num_embeddings=num_embedding_c,
                                           embedding_dim=embedding_dim_c)
            self.char_lstm = nn.LSTM(input_size=embedding_dim_c,hidden_size=embedding_dim_c,bidirectional=True,batch_first=True)

            self.input_dim = embedding_dim+embedding_dim_c*2
        else:
            self.input_dim = embedding_dim

        self.dropout = nn.Dropout(p=0.5)
        self.ote_lstm = nn.LSTM(input_size = self.input_dim, hidden_size=rnn_1_dim, bidirectional=True, batch_first=True)
        self.context_lstm = nn.LSTM(input_size = rnn_1_dim*2, hidden_size=rnn_1_dim, bidirectional=True, batch_first=True)
        self.fc_ote = nn.Linear(rnn_1_dim*2, 5)
        self.fc_ote_te = nn.Linear(rnn_1_dim*2,13)
        self.sc_gate = nn.Linear(rnn_1_dim*2,rnn_1_dim*2)
        self.fc_oe = nn.Linear(rnn_1_dim*2,2)

        # 计算句子级别的情感，POS,NEG,NEU,Multi
        self.atte_ss = Atte_module(input_hidden_size=rnn_1_dim*2,hidden_size=rnn_1_dim)
        self.liner_s_p_scores = nn.Linear(rnn_1_dim*2, 1)
        self.liner_ss = nn.Linear(rnn_1_dim*4, 4)

        # 计算目标级别情感
        self.atte = Atte_module(input_hidden_size=rnn_1_dim*2,hidden_size=rnn_1_dim)
        self.liner_gate = nn.Linear(rnn_1_dim*4, 1)
        self.polarity_lstm = nn.LSTM(input_size = rnn_1_dim*2, hidden_size=rnn_1_dim,bidirectional=True,batch_first=True)
        self.liner_p = nn.Linear(rnn_1_dim*2, 3)

        # 情感转移概率
        self.trans_weight = Parameter(torch.Tensor(4, 3, 3))
        nn.init.ones_(self.trans_weight)
        self.trans_gate = nn.Linear(rnn_1_dim*4, 1)

        def init(module):
            if isinstance(module,nn.Linear):
                module.weight.data.normal_(std=0.2)

        self.apply(init)

    def forward(self,inputs,input_c,attention_mask,spans=None,
                polarity=None,polarity_mask=None,ote=None,te=None,ote_te=None,oe=None,oe_split=None,mode="train"):

        para_size, bsz = inputs.size(1),  inputs.size(0)

        x_emd = self.word_embed(inputs)
        # advisersarial training methods

        x_emd = self.dropout(x_emd)

        ote_hidden,(h,c) = self.ote_lstm(x_emd)
        ote_hidden = self.dropout(ote_hidden)
        ote_logist = nn.Softmax(dim=-1)(self.fc_ote(ote_hidden))
        oe_logist = nn.Softmax()(self.fc_oe(ote_hidden))

        sequence_output, (h, c) = self.context_lstm(ote_hidden)
        sequence_output = self.dropout(sequence_output)

        # # SC begin
        # ote_size= sequence_output.size()
        # ts_hs_tilde = []
        # for i in range(ote_size[1]):
        #     if i ==0:
        #         h_tilde_t = sequence_output[:,i,:]
        #     else:
        #         ts_ht  = sequence_output[:,i,:]
        #         gate = torch.sigmoid(self.sc_gate(torch.cat([ts_ht],dim=-1)))
        #         h_tilde_t = torch.mul(ts_ht, gate)+torch.mul(h_tilde_tm1, 1-gate)
        #
        #     ts_hs_tilde.append(h_tilde_t)
        #     h_tilde_tm1 = h_tilde_t
        # ote_te_hidden = torch.stack(ts_hs_tilde,dim=0).transpose(0,1)
        # ote_te_hidden = self.dropout(ote_te_hidden)
        # ote_te_logist = nn.Softmax(dim=-1)(self.fc_ote_te(ote_te_hidden))
        # # SC end

        spans_size = spans.size()
        # 获取span 的相对位置信息
        spans_position = torch.mean(spans.to(torch.float), dim=-1).to(torch.long)
        positions = torch.arange(sequence_output.size()[1], device=spans.device,requires_grad=False).unsqueeze(0).expand(
            [spans_size[0] * spans_size[1],
             sequence_output.size()[1]])
        spans_position = spans_position.view(size=[spans_size[0] * spans_size[1], 1])
        positions = positions - spans_position
        positions = torch.abs(positions.view(size=[spans_size[0], spans_size[1], sequence_output.size()[1]]))

        spans_extend = spans.view(size=(spans_size[0], spans_size[1] * 2, 1)).expand(size=(spans_size[0],  # batch_size
                                                                                     spans_size[1] * 2,  # 5*2
                                                                                     sequence_output.size()[-1])
                                                                                     )

        sequence_spans = torch.gather(sequence_output, index=spans_extend, dim=1)

        sequence_spans = sequence_spans.view(size=(spans_size[0], spans_size[1], 2, -1))

        # sequence_states,index = sequence_spans.max(dim=2)
        # batch_size * 5* 756
        sequence_states = torch.sum(sequence_spans, dim=2) / 2

        length_mask = positions

        attention_mask_c = attention_mask.clone()

        # batch_size 5 length_size
        attention_mask_c = attention_mask_c.unsqueeze(dim=1).expand(size=[attention_mask.size()[0],
                                                                          spans_size[1],
                                                                          attention_mask_c.size()[1]])

        # 计算相对位置注意力机制，
        # batch_size 5 100
        states,score = self.atte(q=sequence_states,
                               k=sequence_output,
                               m=sequence_output,
                               attention_mask=attention_mask_c,
                               length_mask=length_mask)

        # gate
        s = torch.sum(score[:,:,:,1:],dim=-1)
        score_state = torch.sum(s.unsqueeze(dim=-1)*sequence_output,dim=-2)/(torch.sum(s)+1)
        alpha = torch.sum(s*s)/(torch.sum(s)+1)
        s = sequence_states*(1-alpha) + score_state*alpha
        # sequence_states = torch.cat([sequence_states,s],dim=-1)
        # print(sequence_states)
        # gate = torch.relu(self.liner_gate(torch.cat([states,sequence_states], dim=-1)))
        # states = states*gate + sequence_states*(1-gate)
        s,(_,_)=self.polarity_lstm(s)
        logits_p = self.liner_p(s)


        # 句子级别情感
        states_score = nn.Softmax(dim=-1)(torch.squeeze(self.liner_s_p_scores(states),dim=-1)*polarity_mask)
        states_s = torch.matmul(states_score.unsqueeze(dim=1),states).squeeze(dim=1)
        logits_s_p = self.liner_ss(torch.cat([states_s,torch.cat([h[0],h[1]],dim=-1)],dim=-1))

        # 情感状态转移
        logits_tran_p =torch.matmul(logits_s_p, self.trans_weight.view(size=[4,-1]))
        logits_tran_p = logits_tran_p.view(size=[logits_tran_p.size()[0],3,3])
        states = states.transpose(1,0)
        logits_p_t = logits_p.transpose(1,0)

        hx = None
        out = []
        prev = None
        for i in range(states.size()[0]):
            if hx is None:
                out.append(logits_p_t[i])
                hx = states[i]
                prev = logits_p_t[i]
            else:
                t_gate = torch.sigmoid(self.trans_gate(torch.cat([hx, states[i]], dim=-1)))
                out.append(logits_p_t[i]*t_gate + torch.squeeze(torch.bmm(torch.unsqueeze(prev,dim=1), logits_tran_p),dim=1)*(1-t_gate))
        out = torch.stack(out, dim=0).transpose(1,0)
        # 情感状态转移

        if mode == "train":
            # print(logits_p.size())
            # print(polarity.size())
            # print(polarity_mask.size())
            # p_loss = seq_tag_loss(logits_p, polarity,polarity_mask)
            p_loss = -torch.gather(torch.log_softmax(logits_p,dim=-1), dim=-1, index=torch.unsqueeze(polarity, dim=-1)).squeeze(dim=2)
            p_loss = torch.mean(torch.mean(p_loss * polarity_mask, dim=-1))

            seq_len = torch.sum(attention_mask,dim=-1)
            loss_ote = -torch.gather(torch.log(ote_logist), dim=-1, index=torch.unsqueeze(ote, dim=-1)).squeeze(dim=2)
            loss_ote = torch.mean(torch.sum(loss_ote * attention_mask, dim=-1) / seq_len)

            # loss_ote_te = -torch.gather(torch.log(ote_te_logist), dim=-1, index=torch.unsqueeze(ote, dim=-1)).squeeze(dim=2)
            # loss_ote_te = torch.mean(torch.sum(loss_ote_te * attention_mask, dim=-1) / seq_len)

            loss_oe = -torch.gather(torch.log(oe_logist), dim=-1, index=torch.unsqueeze(oe, dim=-1)).squeeze(dim=2)
            loss_oe = torch.mean(torch.sum(loss_oe * attention_mask, dim=-1) / seq_len)

            # size = score.size()
            # score = torch.reshape(score,shape=[size[0]*size[1],size[2],size[3]])
            # oe_split = torch.reshape(oe_split,shape=[size[0]*size[1],size[2],1])
            loss_oe_spilt = -torch.gather(torch.log(score), dim=-1, index= torch.unsqueeze(oe_split,dim=-1)).squeeze(dim=-1)
            loss_oe_spilt = torch.mean(polarity_mask*(torch.sum(loss_oe_spilt * attention_mask.unsqueeze(dim=1), dim=-1) / seq_len.unsqueeze(dim=1)))

            # def rand_loss():
            # a = torch.tensor(torch.randint(2,size= [4]))
            # l = p_loss*a[0]+loss_ote*a[1]+loss_oe_spilt*a[2] + loss_ote_te*a[3]
            # print(p_loss)
            # print(loss_ote)
            # print(loss_oe_spilt)
            # print(loss_oe)

            return p_loss+ loss_ote+ loss_oe_spilt+ loss_oe

        elif mode == "polarity":
            return torch.max(ote_logist,dim=-1),torch.max(logits_p,dim=-1),torch.max(score,dim=-1)
