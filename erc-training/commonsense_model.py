import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math
from model import SimpleAttention, MatchingAttention, Attention

class CommonsenseRNNCell(nn.Module):

    def __init__(self, roberta_features, comet_features, context_state, internal_state, external_state, intent_state, emotion_state, listener_state=False,
                            context_attention='simple', attention_dim=100, dropout=0.5, emo_lstm=True):
        super(CommonsenseRNNCell, self).__init__()

        self.roberta_features = roberta_features
        self.comet_features = comet_features
        self.context_state = context_state
        self.internal_state = internal_state
        self.external_state = external_state
        self.intent_state = intent_state
        self.emotion_state = emotion_state

        self.g_cell = nn.LSTMCell(roberta_features+internal_state+external_state, context_state)
        self.p_cell = nn.LSTMCell(comet_features+context_state, internal_state)
        self.r_cell = nn.LSTMCell(roberta_features+comet_features+context_state, external_state)
        self.i_cell = nn.LSTMCell(comet_features+internal_state, intent_state)
        self.e_cell = nn.LSTMCell(roberta_features+internal_state+external_state+intent_state, emotion_state)
        
        
        self.emo_lstm = emo_lstm
        self.listener_state = listener_state
        if listener_state:
            self.pl_cell = nn.LSTMCell(comet_features+context_state, internal_state)
            self.rl_cell = nn.LSTMCell(roberta_features+comet_features+context_state, external_state)

        self.dropout = nn.Dropout(dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        if context_attention=='simple':
            self.attention = SimpleAttention(context_state)
        else:
            self.attention = MatchingAttention(context_state, roberta_features, attention_dim, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, U, x1, x2, x3, o1, o2, qmask, g_hist, q0, r0, i0, e0, gc_hist, qc0, rc0, ic0, ec0):
        """
        U -> batch, roberta_features
        x1, x2, x3, o1, o2 -> batch, roberta_features
        x1 -> effect on self; x2 -> reaction of self; x3 -> intent of self
        o1 -> effect on others; o2 -> reaction of others
        qmask -> batch, party
        g_hist -> t-1, batch, context_state
        q0 -> batch, party, internal_state
        e0 -> batch, self.emotion_state
        """
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)
        r0_sel = self._select_parties(r0, qm_idx)

        ## global state ##
        g_, gc_ = self.g_cell(torch.cat([U, q0_sel, r0_sel], dim=1),
                (torch.zeros(U.size()[0],self.context_state).type(U.type()) if g_hist.size()[0]==0 else g_hist[-1],
                 torch.zeros(U.size()[0],self.context_state).type(U.type()) if gc_hist.size()[0]==0 else gc_hist[-1]))
        
        ## context ##
        if g_hist.size()[0]==0:
            c_ = torch.zeros(U.size()[0], self.context_state).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist, U)
       
        ## external state ##
        U_r_c_ = torch.cat([U, x2, c_], dim=1).unsqueeze(1).expand(-1, qmask.size()[1],-1)
        rs_, rsc_ = self.r_cell(U_r_c_.contiguous().view(-1, self.roberta_features+self.comet_features+self.context_state),
                                (r0.view(-1, self.external_state), rc0.view(-1, self.external_state)))

        rs_ = rs_.view(U.size()[0], -1, self.external_state)
        rsc_ = rsc_.view(U.size()[0], -1, self.external_state)
        
        ## internal state ##
        es_c_ = torch.cat([x1, c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_, qsc_ = self.p_cell(es_c_.contiguous().view(-1, self.comet_features+self.context_state),
                                (q0.view(-1, self.internal_state), qc0.view(-1, self.internal_state)))

        qs_ = qs_.view(U.size()[0], -1, self.internal_state)
        qsc_ = qsc_.view(U.size()[0], -1, self.internal_state)
        

        if self.listener_state:
            ## listener external state ##
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.roberta_features)
            er_ = o2.unsqueeze(1).expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.comet_features)
            ss_ = self._select_parties(rs_, qm_idx).unsqueeze(1).\
                    expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.external_state)
            U_er_ss_ = torch.cat([U_, er_, ss_], 1)
            rl_, rlc_ = self.rl_cell(U_er_ss_, (r0.view(-1, self.external_state), rc0.view(-1, self.external_state)))

            rl_ = rl_.view(U.size()[0], -1, self.external_state)
            rlc_ = rlc_.view(U.size()[0], -1, self.external_state)
            
            ## listener internal state ##
            es_ = o1.unsqueeze(1).expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.comet_features)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.internal_state)
            es_ss_ = torch.cat([es_, ss_], 1)
            ql_, qlc_ = self.pl_cell(es_ss_, (q0.view(-1, self.internal_state), qc0.view(-1, self.internal_state)))

            ql_ = ql_.view(U.size()[0], -1, self.internal_state)
            qlc_ = qlc_.view(U.size()[0], -1, self.internal_state)
            
        else:
            rl_ = r0
            rlc_ = rc0

            ql_ = q0
            qlc_ = qc0
            
        qmask_ = qmask.unsqueeze(2)
        
        r_ = rl_*(1-qmask_) + rs_*qmask_ 
        rc_ = rlc_*(1-qmask_) + rsc_*qmask_  

        q_ = ql_*(1-qmask_) + qs_*qmask_
        qc_ = qlc_*(1-qmask_) + qsc_*qmask_
                  
        
        ## intent ##        
        i_q_ = torch.cat([x3, self._select_parties(q_, qm_idx)], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)
        is_, isc_ = self.i_cell(i_q_.contiguous().view(-1, self.comet_features+self.internal_state),
                                (i0.view(-1, self.intent_state), ic0.view(-1, self.intent_state)))

        is_ = is_.view(U.size()[0], -1, self.intent_state)
        isc_ = isc_.view(U.size()[0], -1, self.intent_state)
        il_ = i0
        ilc_ = ic0

        i_ = il_*(1-qmask_) + is_*qmask_
        ic_ = ilc_*(1-qmask_) + isc_*qmask_
        
        ## emotion ##        
        es_ = torch.cat([U, self._select_parties(q_, qm_idx), self._select_parties(r_, qm_idx), 
                         self._select_parties(i_, qm_idx)], dim=1) 
        e0 = torch.zeros(qmask.size()[0], self.emotion_state).type(U.type()) if e0.size()[0]==0\
                else e0

        ec0 = torch.zeros(qmask.size()[0], self.emotion_state).type(U.type()) if ec0.size()[0]==0\
                else ec0
        
        if self.emo_lstm:
            e_, ec_ = self.e_cell(es_, (e0, ec0))
        else:
            e_ = es_ 
            ec_ = es_
        
        g_ = self.dropout1(g_)
        q_ = self.dropout2(q_)
        r_ = self.dropout3(r_)
        i_ = self.dropout4(i_)
        e_ = self.dropout5(e_)
        
        
        return g_, q_, r_, i_, e_, gc_, qc_, rc_, ic_, ec_, alpha


class CommonsenseRNN(nn.Module):

    def __init__(self, roberta_features, comet_features, context_state, internal_state, external_state, intent_state, emotion_state, listener_state=False,
                            context_attention='simple', attention_dim=100, dropout=0.5, emo_lstm=True):
        super(CommonsenseRNN, self).__init__()

        self.roberta_features = roberta_features
        self.context_state = context_state
        self.internal_state = internal_state
        self.external_state = external_state
        self.intent_state = intent_state
        self.emotion_state = emotion_state
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = CommonsenseRNNCell(roberta_features, comet_features, context_state, internal_state, external_state, intent_state, emotion_state,
                            listener_state, context_attention, attention_dim, dropout, emo_lstm)

    def forward(self, U, x1, x2, x3, o1, o2, qmask):
        """
        U -> seq_len, batch, roberta_features
        x1, x2, x3, o1, o2 -> seq_len, batch, comet_features
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type()) # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.internal_state).type(U.type()) # batch, party, internal_state
        r_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.external_state).type(U.type()) # batch, party, external_state
        i_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.intent_state).type(U.type()) # batch, party, intent_state

        gc_hist = torch.zeros(0).type(U.type()) # 0-dimensional tensor
        qc_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.internal_state).type(U.type()) # batch, party, internal_state
        rc_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.external_state).type(U.type()) # batch, party, external_state
        ic_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.intent_state).type(U.type()) # batch, party, intent_state
        
        e_ = torch.zeros(0).type(U.type()) # batch, emotion_state
        e = e_

        ec_ = torch.zeros(0).type(U.type()) # batch, emotion_state
        ec = ec_

        alpha = []
        for u_, x1_, x2_, x3_, o1_, o2_, qmask_ in zip(U, x1, x2, x3, o1, o2, qmask):
            g_, q_, r_, i_, e_, \
            gc_, qc_, rc_, ic_, ec_, alpha_ = self.dialogue_cell(u_, x1_, x2_, x3_, o1_, o2_, qmask_, \
                                                                g_hist, q_, r_, i_, e_, gc_hist, qc_, rc_, ic_, ec_)

            g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
            e = torch.cat([e, e_.unsqueeze(0)],0)

            gc_hist = torch.cat([gc_hist, gc_.unsqueeze(0)],0)
            ec = torch.cat([ec, ec_.unsqueeze(0)],0)
            
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        return e, alpha # seq_len, batch, emotion_state


class CommonsenseLSTMModel(nn.Module):

    def __init__(self, roberta_features, comet_features, context_state, internal_state, external_state, intent_state, emotion_state, hidden_dim, attention_dim=100, n_classes=7, listener_state=False, 
        context_attention='simple', dropout_rec=0.5, dropout=0.5, emo_lstm=True, mode1=0, norm=0):

        super(CommonsenseLSTMModel, self).__init__()

        if mode1 == 0:
            D_x = 4 * roberta_features
        elif mode1 == 1:
            D_x = 2 * roberta_features
        else:
            D_x = roberta_features

        self.mode1 = mode1
        self.norm_strategy = norm
        self.linear_in = nn.Linear(D_x, hidden_dim)

        self.r_weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))

        norm_train = True
        self.norm1a = nn.LayerNorm(roberta_features, elementwise_affine=norm_train)
        self.norm1b = nn.LayerNorm(roberta_features, elementwise_affine=norm_train)
        self.norm1c = nn.LayerNorm(roberta_features, elementwise_affine=norm_train)
        self.norm1d = nn.LayerNorm(roberta_features, elementwise_affine=norm_train)

        self.norm3a = nn.BatchNorm1d(roberta_features, affine=norm_train)
        self.norm3b = nn.BatchNorm1d(roberta_features, affine=norm_train)
        self.norm3c = nn.BatchNorm1d(roberta_features, affine=norm_train)
        self.norm3d = nn.BatchNorm1d(roberta_features, affine=norm_train)

        self.dropout   = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout_rec)
        self.cs_rnn_f = CommonsenseRNN(hidden_dim, comet_features, context_state, internal_state, external_state, intent_state, emotion_state, listener_state,
                                       context_attention, attention_dim, dropout_rec, emo_lstm)
        self.cs_rnn_r = CommonsenseRNN(hidden_dim, comet_features, context_state, internal_state, external_state, intent_state, emotion_state, listener_state,
                                       context_attention, attention_dim, dropout_rec, emo_lstm)
        self.sense_lstm = nn.LSTM(input_size=comet_features, hidden_size=comet_features//2, num_layers=1, bidirectional=True)
        self.matchatt = MatchingAttention(2*emotion_state,2*emotion_state,att_type='general2')
        self.linear     = nn.Linear(2*emotion_state, hidden_dim)
        self.smax_fc    = nn.Linear(hidden_dim, n_classes)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)
        return pad_sequence(xfs)

    def forward(self, r1, r2, r3, r4, x1, x2, x3, o1, o2, qmask, umask, att2=True, return_hidden=False):
        """
        U -> seq_len, batch, roberta_features
        qmask -> seq_len, batch, party
        """

        seq_len, batch, feature_dim = r1.size()

        if self.norm_strategy == 1:
            r1 = self.norm1a(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.norm1b(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.norm1c(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.norm1d(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        elif self.norm_strategy == 2:
            norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
            r1 = norm2(r1.transpose(0, 1)).transpose(0, 1)
            r2 = norm2(r2.transpose(0, 1)).transpose(0, 1)
            r3 = norm2(r3.transpose(0, 1)).transpose(0, 1)
            r4 = norm2(r4.transpose(0, 1)).transpose(0, 1)

        elif self.norm_strategy == 3:
            r1 = self.norm3a(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.norm3b(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.norm3c(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.norm3d(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        if self.mode1 == 0:
            r = torch.cat([r1, r2, r3, r4], axis=-1)
        elif self.mode1 == 1:
            r = torch.cat([r1, r2], axis=-1)
        elif self.mode1 == 2:
            r = (r1 + r2 + r3 + r4)/4
        elif self.mode1 == 3:
            r = r1
        elif self.mode1 == 4:
            r = r2
        elif self.mode1 == 5:
            r = r3
        elif self.mode1 == 6:
            r = r4
        elif self.mode1 == 7:
            r = self.r_weights[0]*r1 + self.r_weights[1]*r2 + self.r_weights[2]*r3 + self.r_weights[3]*r4

        r = self.linear_in(r)


        emotions_f, alpha_f = self.cs_rnn_f(r, x1, x2, x3, o1, o2, qmask) # seq_len, batch, emotion_state
        # emotions_f = self.dropout_rec(emotions_f)
        
        out_sense, _ = self.sense_lstm(x1)
        
        rev_r = self._reverse_seq(r, umask)
        rev_x1 = self._reverse_seq(x1, umask)
        rev_x2 = self._reverse_seq(x2, umask)
        rev_x3 = self._reverse_seq(x3, umask)
        rev_o1 = self._reverse_seq(o1, umask)
        rev_o2 = self._reverse_seq(o2, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.cs_rnn_r(rev_r, rev_x1, rev_x2, rev_x3, rev_o1, rev_o2, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        
        emotions = torch.cat([emotions_f,emotions_b],dim=-1)
        emotions = self.dropout_rec(emotions)
        
        alpha, alpha_f, alpha_b = [], [], []
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes

        if return_hidden:
            return hidden, alpha, alpha_f, alpha_b, emotions
        return log_prob, out_sense, alpha, alpha_f, alpha_b, emotions