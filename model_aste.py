
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter



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


class ATO(nn.Module):

    def __init__(self,
                 input_hidden_size,
                 hidden_size):
        super(ATO, self).__init__()

        self.length_embedding = nn.Embedding(num_embeddings = 100,
                                             embedding_dim = input_hidden_size)
        self.att_liner = nn.Linear(input_hidden_size,5)

        self.drop_out = nn.Dropout(p=0.2)

        def init_weight(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight.data)

        self.apply(init_weight)

    def forward(self, q, k, length_mask=None):

        q = q.unsqueeze(dim=2)
        k = k.unsqueeze(dim=1)
        if length_mask is not None:
            length_embd = self.length_embedding(length_mask)
            k = k.expand(size=[k.size()[0],
                               q.size()[1],
                               k.size()[2],
                               k.size()[3]])
            k = k + length_embd

        scores = nn.Softmax(dim=-1)(self.att_liner(q+k))
        return  scores

class ASTE(nn.Module):

    def __init__(self,
                 num_embeddings,
                 num_embedding_c,
                 embedding_dim,
                 embedding_dim_c,
                 rnn_1_dim,
                 label_size,
                 polarity_size,
                 word_vec = None,
                 use_char = False):

        super(ASTE, self).__init__()

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
        self.fc_ote = nn.Linear(rnn_1_dim*2, label_size)
        self.ato = ATO(input_hidden_size=rnn_1_dim*2,hidden_size=rnn_1_dim)
        self.liner_p = nn.Linear(rnn_1_dim*4, polarity_size)

        def init(module):
            if isinstance(module,nn.Linear):
                module.weight.data.normal_(std=0.2)

        self.apply(init)

    def forward(self,inputs,input_c,attention_mask,spans=None,
                polarity=None,polarity_mask=None,ote=None,te=None,ote_te=None,oe=None,oe_split=None,mode="train"):

        para_size, bsz = inputs.size(1),  inputs.size(0)

        x_emd = self.word_embed(inputs)
        x_emd = self.dropout(x_emd)

        ote_hidden,(h,c) = self.ote_lstm(x_emd)
        ote_hidden = self.dropout(ote_hidden)
        ote_logist = nn.Softmax(dim=-1)(self.fc_ote(ote_hidden))

        sequence_output, (h, c) = self.context_lstm(ote_hidden)
        sequence_output = self.dropout(sequence_output)

        spans_size = spans.size()
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


        sequence_states = torch.sum(sequence_spans, dim=2) / 2

        length_mask = positions

        attention_mask_c = attention_mask.clone()

        # batch_size 5 length_size
        attention_mask_c = attention_mask_c.unsqueeze(dim=1).expand(size=[attention_mask.size()[0],
                                                                          spans_size[1],
                                                                          attention_mask_c.size()[1]])

        score = self.ato(q=sequence_states,
                          k=sequence_output,
                          length_mask=length_mask)

        # gate
        score_g = torch.sum(score[:,:,:,1:],dim=-1)
        score_state = torch.sum(score_g.unsqueeze(dim=-1)*sequence_output,dim=-2)/(torch.sum(score_g)+1)
        s = torch.cat([score_state,sequence_states],dim=-1)
        logits_p = self.liner_p(s)



        if mode == "train":

            p_loss = -torch.gather(torch.log_softmax(logits_p,dim=-1), dim=-1, index=torch.unsqueeze(polarity, dim=-1)).squeeze(dim=2)
            p_loss = torch.mean(torch.mean(p_loss * polarity_mask, dim=-1))

            seq_len = torch.sum(attention_mask,dim=-1)
            loss_ote = -torch.gather(torch.log(ote_logist), dim=-1, index=torch.unsqueeze(ote, dim=-1)).squeeze(dim=2)
            loss_ote = torch.mean(torch.sum(loss_ote * attention_mask, dim=-1) / seq_len)

            loss_oe_spilt = -torch.gather(torch.log(score), dim=-1, index= torch.unsqueeze(oe_split,dim=-1)).squeeze(dim=-1)
            loss_oe_spilt = torch.mean(polarity_mask*(torch.sum(loss_oe_spilt * attention_mask.unsqueeze(dim=1), dim=-1) / seq_len.unsqueeze(dim=1)))

            return p_loss+ loss_ote+ loss_oe_spilt

        elif mode == "polarity":
            return torch.max(ote_logist,dim=-1),torch.max(logits_p,dim=-1),torch.max(score,dim=-1)
