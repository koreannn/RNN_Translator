import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_weight = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(pretrained_weight)
        
        # # 사전학습된 가중치 Freeze 시킬 때
        # for param in self.embedding.parameters():
        #     param.requires_grad = False
        
        # nn.Embedding을 선언하고, self.embedding을 삽입
        self.rnn = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim, batch_first = True)
        
    def forward(self, src_ids):
        embedded = self.embedding(src_ids)
        outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_weight = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(pretrained_weight)
            
        # # 사전학습된 가중치 Freeze 시킬 때
        # for param in self.embedding.parameters():
        #     param.requires_grad = False
        
        self.rnn = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim, batch_first = True)
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
