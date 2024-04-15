import torch
from torch.nn import init
import torch.nn as nn
from CRN import CRN  # Import the CRN class from CRN.py
from torch.nn import functional as F

def init_modules(modules, w_init='kaiming_uniform'):
    if w_init == "normal":
        _init = init.normal_
    elif w_init == "xavier_normal":
        _init = init.xavier_normal_
    elif w_init == "xavier_uniform":
        _init = init.xavier_uniform_
    elif w_init == "kaiming_normal":
        _init = init.kaiming_normal_
    elif w_init == "kaiming_uniform":
        _init = init.kaiming_uniform_
    elif w_init == "orthogonal":
        _init = init.orthogonal_
    else:
        raise NotImplementedError
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _init(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _init(param)

class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj * v_proj), dim=-1)
        #print('v_q_cat', v_q_cat.shape)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)
        v_q_cat = torch.unsqueeze(v_q_cat, 1)
        #print('v_q_cat', v_q_cat.shape)
        attn = self.attn(v_q_cat)  # (bz, k, 1)
        #print('attn', attn.shape)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        #print('attn', attn.shape)
        ##print('attn_distill', attn*visual_feat.shape)
        v_distill = (attn * visual_feat).sum(1)
        #print('v_distill', v_distill.shape)

        return v_distill


class InputUnitLinguistic(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.15)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions = questions.long()
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)
        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        #print('question_embedding', question_embedding.shape)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding



class InputUnitGraph(nn.Module):
    def __init__(self, k_max_graph_level, k_max_clip_level, graph_embedding_dim, module_dim=512, question_dim=512, 
                 dropout_style=0, dropout_prob=0.10, crn_dropout_prob=0.10):
        """
        Initializes the InputUnitGraph for processing graph embeddings conditioned on the question.

        Args:
            k_max_graph_level (int): The maximum number of graph levels to consider.
            graph_embedding_dim (int): Dimensionality of the graph embeddings.
            module_dim (int): Dimensionality of the module's output.
            question_dim (int): Dimensionality of the question embedding.
            dropout_style (int): Style of dropout to apply.
            dropout_prob (float): Dropout probability for regularization.
            crn_dropout_prob (float): Dropout probability specifically for the CRN.
        """
        super(InputUnitGraph, self).__init__()

        # CRN units for processing graph embeddings conditioned on the question
        self.graph_level_question_cond = CRN(module_dim, k_max_graph_level, k_max_graph_level, gating=True, 
                                             spl_resolution=1, dropout_style=dropout_style, crn_dropout_prob=crn_dropout_prob)

        #video level CRN units 
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=True,
                                            spl_resolution=1, dropout_style=dropout_style, crn_dropout_prob=crn_dropout_prob)

        
            # Additional CRN units for graph level
        self.graph_level_question_cond_2 = CRN(module_dim, k_max_graph_level, k_max_graph_level, gating=True, 
                                            spl_resolution=1, dropout_style=dropout_style, crn_dropout_prob=crn_dropout_prob)

        # Additional CRN units for video level
        self.video_level_question_cond_2 = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=True,
                                            spl_resolution=1, dropout_style=dropout_style, crn_dropout_prob=crn_dropout_prob)

        
        # Fully connected layer for projecting graph embeddings
        if dropout_style == 0:
            self.graph_embedding_proj = nn.Linear(graph_embedding_dim, module_dim)
        elif dropout_style == 1:
            self.graph_embedding_proj = nn.Sequential(nn.Dropout(dropout_prob), nn.Linear(graph_embedding_dim, module_dim))

        # Fully connected layer for projecting question embeddings
        self.question_embedding_proj = nn.Linear(question_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, graph_embeddings, question_embedding):
        """
        Forward pass for processing graph embeddings conditioned on the question.

        Args:
            graph_embeddings: [Tensor] (batch_size, num_graphs, graph_embedding_dim)
            question_embedding: [Tensor] (batch_size, question_dim)

        Returns:
            Tensor: Encoded graph feature (batch_size, N, module_dim)
        """
        batch_size = graph_embeddings.size(0)
        graph_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        
        for i in range(graph_embeddings.size(1)):
            # Project graph embeddings to match module dimension
            if graph_embeddings.dim() >= 3:
                print('graph_embeddings shape:', graph_embeddings.shape)
                graph_level_embedding = graph_embeddings[:, i, :]
                print('then turned into shape:', graph_level_embedding.shape)
            else:
                print('graph_embeddings shape:', graph_embeddings.shape)
                graph_level_embedding = graph_embeddings
            graph_level_embedding_proj = self.graph_embedding_proj(graph_level_embedding)

            # Graph level CRN conditioned on the question
            graph_level_crn_question = self.graph_level_question_cond(
                torch.unbind(graph_level_embedding_proj, dim=1), 
                question_embedding_proj
            )

            graph_level_crn_output = torch.cat(
                [graph_relation.unsqueeze(1) for graph_relation in graph_level_crn_question],
                dim=1
            )
        
            graph_level_crn_output = graph_level_crn_output.view(batch_size, -1, self.module_dim)
            graph_level_crn_outputs.append(graph_level_crn_output)


        # Aggregating all graph level outputs for video level processing
        aggregated_graph_level_output = torch.cat(graph_level_crn_outputs, dim=1)

        # Video level CRN conditioned on the question
        video_level_crn_output = self.video_level_question_cond(
                                                            torch.unbind(aggregated_graph_level_output, dim=1),
                                                                question_embedding_proj)
        video_level_crn_output = torch.cat(video_level_crn_output, dim=1)
        final_video_level_output = video_level_crn_output.view(batch_size, -1, self.module_dim)
        final_video_level_output = torch.mean(final_video_level_output, dim=1)
        return final_video_level_output


class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out



class HCRNNetwork(nn.Module):
    def __init__(self, module_dim, graph_embedding_dim, word_dim, k_max_graph_level, k_max_clip_level, spl_resolution, vocab, 
                 dropout_style, dropout_prob, crn_dropout_prob): # NEW DROPOUT
        super(HCRNNetwork, self).__init__()

        self.feature_aggregation = FeatureAggregation(module_dim)

        self.dropout_style = dropout_style # NEW DROPOUT

        encoder_vocab_size = len(vocab['question_token_to_idx'])
        self.num_classes = len(vocab['answer_token_to_idx'])
        #print('num_classes: ', self.num_classes)
        self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                                module_dim=module_dim, rnn_dim=module_dim)
        
        self.lingual_projection =  nn.Linear(word_dim, module_dim)
        
        self.graph_input_unit = InputUnitGraph(k_max_graph_level = k_max_graph_level, k_max_clip_level = k_max_clip_level,
                                                graph_embedding_dim = graph_embedding_dim, module_dim=module_dim, dropout_style=self.dropout_style, 
                                                dropout_prob=dropout_prob, crn_dropout_prob=crn_dropout_prob) # NEW DROPOUT
        self.graph_input_unit2 = InputUnitGraph(k_max_graph_level = k_max_graph_level, k_max_clip_level = k_max_clip_level,
                                                graph_embedding_dim = module_dim, module_dim=module_dim, dropout_style=self.dropout_style, 
                                                dropout_prob=dropout_prob, crn_dropout_prob=crn_dropout_prob)
        
        
        self.output_unit = OutputUnitOpenEnded(num_answers=171)

        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, graph_feats, question,
                question_len = None):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        if question_len is None:
            question_len =  torch.full((batch_size,), 512).to(question.device)

        # get image, word, and sentence embeddings
        #question_embedding = self.linguistic_input_unit(question, question_len)
        question_embedding = self.lingual_projection(question)
        
        graph_embedding = self.graph_input_unit(graph_feats, question_embedding)

        print('intermediate graph shape', graph_embedding.shape)

        graph_embedding = self.graph_input_unit2(graph_embedding, question_embedding)

        
        print('question_embedding final shape', question_embedding.shape)
        print('graph_embedding final shape', graph_embedding.shape)

        #print('pre-final embs', question_embedding.shape, graph_embedding.shape)
        final_embedding = self.feature_aggregation(question_embedding, graph_embedding)
        ##print('final_shape', final_embedding.shape)
        out = self.output_unit(question_embedding, final_embedding)
        
        return out
    
