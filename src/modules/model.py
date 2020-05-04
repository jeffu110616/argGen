import torch
import torch.nn as nn
from transformers import XLNetModel

from modules import content_decoder
from modules import sentence_planner
from sys import exit

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class EncoderRNN(nn.Module):

    def __init__(self, opt):
        super(EncoderRNN, self).__init__()
        self.hidden_size = opt.hidden_size // 2 # use bidirectional RNN
        self.LSTM = nn.LSTM(input_size=768,
                            hidden_size=self.hidden_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=opt.dropout,
                            bidirectional=True)

        return

    def forward(self, input_embedded, input_lengths):
        """forward path, note that inputs are batch first"""

        lengths_list = input_lengths.view(-1).tolist()
        packed_emb = pack(input_embedded, lengths_list, True)

        memory_bank, encoder_final = self.LSTM(packed_emb)
        memory_bank = unpack(memory_bank)[0].view(input_embedded.size(0),input_embedded.size(1), -1)

        return memory_bank, encoder_final

class EncoderXLNet(nn.Module):

    def __init__(self, opt):
        super(EncoderXLNet, self).__init__()
        self.hidden_size = opt.hidden_size # target encoded dimension

        self.transformer = XLNetModel.from_pretrained('xlnet-base-cased')

        for param in self.transformer.parameters():
            param.requires_grad = False

        self.encodeLayer = nn.Linear(self.transformer.config.d_model, 
                                    opt.hidden_size, 
                                    bias=True)

        return

    def forward(self, src_inputs_tensor):
        outputs = self.transformer(src_inputs_tensor)
        encoded = self.encodeLayer(outputs[0])

        return encoded

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.vocab_size = 32000

        self.sp_dec = sentence_planner.SentencePlanner(opt)
        self.wd_dec = content_decoder.WordDecoder(self.vocab_size, opt)
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=-1)
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_word_loss_probs(self, word_prob, word_targets):
        """
        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=-1)

        Calculate cross-entropy loss on words.
        Args:
            word_prob: [batch_size, ]
            word_targets: [batch_size, ]
        """
        word_loss = self.nll_loss(torch.log(word_prob).view(-1, self.vocab_size), word_targets.view(-1))
        ppl = torch.exp(word_loss / torch.sum(word_targets >= 0))
        word_loss /= word_targets.size(0)
        return word_loss, ppl

    def compute_stype_loss(self, stype_pred, stype_labels):
        """
        Calculate cross-entropy loss on sentence type prediction.
        Args:
            stype_pred: [batch_size, max_sent_num, 4]: logits for type prediction
            stype_labels: [batch_size, max_sent_num]: gold-standard sentence type indices
        Returns:
            st_loss: scalar loss value averaged over all samples in the batch
        """
        st_loss = self.ce_loss(stype_pred.view(-1, self.sp_dec.sentence_type_n),
                               stype_labels.view(-1)) / stype_labels.size(0)
        return st_loss

    def compute_content_selection_loss(self, cs_pred, cs_labels, ph_bank_mask):
        """
        Calculate binary cross-entropy loss on keyphrase selection.
        Args:
            cs_pred: [batch_size, max_sent_num, max_ph_bank_size]
            cs_labels: [batch_size, max_sent_num, max_ph_bank_size]
            ph_bank_mask: [batch_size, max_sent_num, max_ph_bank_size]
        Returns:
            cs_loss: scalar loss value averaged over all samples in the batch.
        """
        cs_loss_flat = self.bce_loss(cs_pred.view(-1), cs_labels.view(-1))
        cs_loss_masked = ph_bank_mask.view(-1) * cs_loss_flat
        cs_loss = torch.sum(cs_loss_masked) / torch.sum(ph_bank_mask)
        return cs_loss


class ArgGenModel(Model):

    def __init__(self, opt):
        super(ArgGenModel, self).__init__(opt)
        self.encoder = EncoderRNN(opt)
        self.xlnetModel = EncoderXLNet(opt)
        self.word_emb = self.xlnetModel.transformer.word_embedding

    def forward_enc(self, src_inputs_tensor, src_len_tensor):
        src_emb = self.word_emb(src_inputs_tensor)
        enc_outs, enc_final = self.encoder.forward(input_embedded=src_emb, input_lengths=src_len_tensor)

        # self.sp_dec.init_state(enc_final)
        # self.wd_dec.init_state(enc_final)
        return enc_outs, enc_final
    
    def forward_xlnet_enc(self, src_inputs_tensor):
        encoded = self.xlnetModel(src_inputs_tensor)

        return encoded

    def forward(self, tensor_dict, device=None):

        batch_size, sent_num, _ = tensor_dict["phrase_bank_selection_index"].size()

        enc_outs_op, enc_final_op = self.forward_enc(src_inputs_tensor=tensor_dict["src_inputs"],
                         src_len_tensor=tensor_dict["src_lens"])

        # Needed to sort manually
        # _, sorted_indice = torch.sort(tensor_dict["src_inner_lens"], descending=True)
        # _, inv_sorted_indice = torch.sort(sorted_indice, descending=False)
        # enc_outs_inner, enc_final_inner = self.forward_enc(src_inputs_tensor=tensor_dict["src_inner_inputs"][sorted_indice],
        #                  src_len_tensor=tensor_dict["src_inner_lens"][sorted_indice])
        
        inner_enc = self.forward_xlnet_enc(src_inputs_tensor=tensor_dict["src_inputs"])

        # enc_outs_inner = enc_outs_inner[inv_sorted_indice]
        # print(enc_outs_inner.size())

        # enc_outs_inner_bi = enc_outs_inner.view(enc_outs_inner.size(0), enc_outs_inner.size(1), 2, -1)
        # print(enc_outs_inner_bi.size())

        # enc_outs_inner_last = torch.cat( [enc_outs_inner_bi[:, -1, 0], enc_outs_inner_bi[:, 0, 1]], -1 ).view(batch_size, 1, -1)
        # print(enc_outs_inner_last.size())

        # enc_outs_inner_last = enc_outs_inner_last.repeat_interleave(enc_outs_op.size(1), 1)
        enc_outs = torch.cat([enc_outs_op, inner_enc], -1)

        # print(enc_outs_op.size())
        # print(inner_enc.size())
        # print(enc_outs.size())
        # exit()

        self.sp_dec.init_state(enc_final_op)
        self.wd_dec.init_state(enc_final_op)

        ph_bank_emb_raw = self.word_emb(tensor_dict["phrase_bank"])

        ph_bank_emb = torch.sum(ph_bank_emb_raw, -2)

        _, sp_dec_outs, stype_pred_logits, next_sent_sel_pred_probs, kp_mem_outs = \
            self.sp_dec.forward(
                ph_bank_emb=ph_bank_emb,
                ph_bank_sel_ind_inputs=tensor_dict["phrase_bank_selection_index"],
                stype_one_hot_tensor=tensor_dict["tgt_sent_type_onehot"],
                ph_sel_ind_mask=tensor_dict["phrase_bank_selection_index_mask"],
            )

        wd_dec_state, enc_attn, wd_pred_prob, wd_logits = self.wd_dec.forward(
            word_inputs_emb=self.word_emb(tensor_dict["tgt_word_ids_input"]),
            sent_planner_output=sp_dec_outs,
            sent_id_tensor=tensor_dict["tgt_sent_ids"],
            sent_mask_tensor=tensor_dict["tgt_word_ids_input_mask"],
            memory_bank=enc_outs,
            memory_len=tensor_dict["phrase_bank_len"],
            ph_bank_word_ids=tensor_dict["phrase_bank"],
            ph_bank_word_mask=tensor_dict["phrase_bank_word_mask"],
            stype_one_hot=tensor_dict["tgt_sent_type_onehot"].float(),
        )

        return stype_pred_logits, next_sent_sel_pred_probs, wd_pred_prob, wd_logits, enc_attn, kp_mem_outs




