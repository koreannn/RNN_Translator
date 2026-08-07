import torch
import torch.nn as nn

def _init_gru_orthogonal_xavier(rnn: nn.GRU, hidden_dim: int) -> None:
    for gate in range(3):
        chunk = rnn.weight_hh_l0.data[gate * hidden_dim:(gate + 1) * hidden_dim, :]
        nn.init.orthogonal_(chunk)
    nn.init.xavier_uniform_(rnn.weight_ih_l0.data)

def _apply_init_scheme(rnn: nn.GRU, hidden_dim: int, init_scheme: str) -> None:
    if init_scheme == 'default':
        return
    elif init_scheme == 'Orthogonal_Xavier':
        _init_gru_orthogonal_xavier(rnn, hidden_dim)
    else:
        raise ValueError(f"Unknown init_scheme: {init_scheme}")
    


class LayerNormGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
    
    @property
    def weight_hh_l0(self):
        return self.cell.weight_hh
    
    @property
    def weight_ih_l0(self):
        return self.cell.weight_ih
    
    def forward(self, x, h0 = None):
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size) if h0 is None else h0.squeeze(0) # 배치 차원 제거
        
        outputs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            h = self.ln(h)
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim = 1) # (bs, seq_len, hidden_size)
        h_n = h.unsqueeze(0) # (1, bs, hidden_dim) - nn.GRU 출력 포맷과 맞추기 위함
        return outputs, h_n
        

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_weight = None, init_scheme = 'default', use_layer_norm = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(pretrained_weight)
        
        # # 사전학습된 가중치 Freeze 시킬 때
        # for param in self.embedding.parameters():
        #     param.requires_grad = False
        
        if use_layer_norm:
            self.rnn = LayerNormGRU(embedding_dim, hidden_dim)
        
        else:
            # nn.Embedding을 선언하고, self.embedding을 삽입
            self.rnn = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim, batch_first = True)
            _apply_init_scheme(self.rnn, hidden_dim, init_scheme)
        
    def forward(self, src_ids):
        embedded = self.embedding(src_ids)
        outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_weight = None, init_scheme = 'default', use_layer_norm = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(pretrained_weight)
            
        # # 사전학습된 가중치 Freeze 시킬 때
        # for param in self.embedding.parameters():
        #     param.requires_grad = False
        
        if use_layer_norm:
            self.rnn = LayerNormGRU(embedding_dim, hidden_dim)
        
        else:
            self.rnn = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim, batch_first = True)
            _apply_init_scheme(self.rnn, hidden_dim, init_scheme)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, target_ids, enc_last_hidden):
        embedded = self.embedding(target_ids)
        outputs, dec_last_hidden = self.rnn(embedded, enc_last_hidden)
        logits = self.fc(outputs)
        return logits, dec_last_hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, padding_id = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = padding_id
    
    def forward(self, src_ids, target_ids):
        _, encoder_hidden = self.encoder(src_ids)
        logits, _ = self.decoder(target_ids, encoder_hidden)
        return logits
