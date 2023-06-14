import torch
from torch import nn
from torch.nn import functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaseModel(nn.Module):
    def __init__(self, inp_voc, out_voc, emb_size=64, hid_size=128):
        """
        Базовая модель архитектуры трансформера.
        """
        super().__init__() 

        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hid_size = hid_size
        
        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)
        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True)

        self.dec_start = nn.Linear(hid_size, hid_size)
        self.dec0 = nn.GRUCell(emb_size, hid_size)
        self.logits = nn.Linear(hid_size, len(out_voc))


    def forward(self, inp, out):
        """
        Применяется encode, затем decode.
        """
        initial_state = self.encode(inp)
        return self.decode(initial_state, out)


    def encode(self, inp, **flags):
        """
        Считаем скрытое состояние, которое будет начальным для decode
        :param inp: матрица входных токенов
        :returns: скрытое представление с которого будет начинаться decode
        """
        inp_emb = self.emb_inp(inp)

        enc_seq, _ = self.enc0(inp_emb)
        # enc_seq: [batch, time, hid_size]
        
        # последний токен, не последний на самом деле, так как мы делали pading, чтобы тексты были
        # одинакового размера, поэтому подсчитать длину исходного предложения не так уж тривиально
        lengths = (inp != self.inp_voc.eos_ix).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]
        # last_state: [batch_size, hid_size]
        
        dec_start = self.dec_start(last_state)
        return [dec_start]


    def decode_step(self, prev_state, prev_tokens, **flags):
        """
        Принимает предыдущее состояние декодера и токены, возвращает новое состояние и 
        логиты для следующих токенов
        """
        prev_gru0_state = prev_state[0]
        
        output = self.emb_out(prev_tokens)
        output = self.dec0(output, prev_gru0_state)
        output_logits = self.logits(output)
        return [output], output_logits


    def decode(self, initial_state, out_tokens, **flags):
        batch_size = out_tokens.shape[0]
        state = initial_state
        
        # первый символ всегда BOS
        onehot_bos = F.one_hot(torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64),
                               num_classes=len(self.out_voc)).to(device=out_tokens.device)
        first_logits = torch.log(onehot_bos.to(torch.float32) + 1e-9)
        
        logits_sequence = [first_logits]
        # в цикле делаем decode_step, получаем logits_sequence
        for i in range(out_tokens.shape[1] - 1):
            state, output_logits = self.decode_step(state, out_tokens[:, i])
            logits_sequence.append(output_logits)
        return torch.stack(logits_sequence, dim=1)


    def decode_inference(self, initial_state, max_len=100, **flags):
        """
        Генерируем токены для перевода.
        """
        batch_size, device = len(initial_state[0]), initial_state[0].device
        state = initial_state
        outputs = [torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64, device=device)]
        all_states = [initial_state]

        for _ in range(max_len):
            state, logits = self.decode_step(state, outputs[-1])
            outputs.append(logits.argmax(dim=-1))
            all_states.append(state)
        
        return torch.stack(outputs, dim=1), all_states


    def translate_lines(self, inp_lines, **kwargs):
        """
        Функция для перевода.
        """
        inp = self.inp_voc.to_matrix(inp_lines).to(device)
        initial_state = self.encode(inp)
        out_ids, states = self.decode_inference(initial_state, **kwargs)
        return self.out_voc.to_lines(out_ids.cpu().numpy()), states


class AttentionLayer(nn.Module):
    def __init__(self, enc_size, dec_size, hid_size):
        super().__init__()
        self.enc_size = enc_size 
        self.dec_size = dec_size 
        self.hid_size = hid_size 
        
        # все слои, которые нужны Attention
        self.enc = nn.Linear(enc_size, hid_size)
        self.dec = nn.Linear(dec_size, hid_size)
        self.out = nn.Linear(hid_size, 1)

        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, enc, dec, inp_mask):
        """
        Подсчитываем attention ответ and веса
        :param enc: [batch_size, ninp, enc_size]
        :param dec: decode state[batch_size, dec_size]
        :param inp_mask: маска, 0 там где pading [batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
        """
        # считаем логиты
        batch_size = enc.shape[0]
        a = self.enc(enc) + self.dec(dec).reshape(-1, 1, self.hid_size)
        a = torch.tanh(a)
        a = self.out(a)

        # Применим маску - если значение маски 0, логиты должны быть -inf или -1e9
        a[torch.where(inp_mask == 0)] = -1e9

        # Примените softmax
        probs = self.softmax(a.reshape(batch_size, -1))

        # Подсчитайте выход attention используя enc состояния и вероятностями
        attn = torch.sum(probs.reshape(batch_size, -1, 1) * enc, axis=1)

        return attn, probs


class AttentiveModel(BaseModel):
    def __init__(self, inp_voc, out_voc,
                 emb_size=64, hid_size=128, attn_size=128):
        """
        Переводчик с Attention.
        """
        super().__init__(inp_voc, out_voc, emb_size, hid_size)

        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.hid_size = hid_size
        self.emb_size = emb_size
        self.attn_size = attn_size

        self.dec0 = nn.GRUCell(self.emb_size + self.hid_size, self.hid_size)
        self.attention = AttentionLayer(self.hid_size, self.hid_size, self.attn_size)


    def encode(self, inp, **flags):
        """
        Считаем скрытые состояния, которые используем в decode.
        :param inp: матрица входных токенов
        """
        # делаем encode
        inp_emb = self.emb_inp(inp)
        enc_seq, _ = self.enc0(inp_emb)

        [dec_seq] = super().encode(inp, **flags)
        enc_mask = self.out_voc.compute_mask(inp)
        
        # применяем attention слой для скрытых состояний
        first_attn_probas = self.attention(enc_seq, dec_seq, enc_mask)[0]
        
        # Для декодера нужно вернуть:
        # - начальное состояние для RNN декодера
        # - последовательность скрытых состояний encoder, maskа для них
        # - последним передаем вероятности слоя attention
        first_state = [dec_seq, enc_seq, enc_mask, first_attn_probas]
        return first_state
   

    def decode_step(self, prev_state, prev_tokens, **flags):
        """
        Принимает предыдущее состояние декодера и токены, возвращает новое состояние и логиты для следующих токенов.
        :param prev_state: список тензоров предыдущих состояний декодера
        :param prev_tokens: предыдущие выходные токены [batch_size]
        :return: список тензоров состояния следующего декодера, тензор логитов [batch, n_tokens]
        """
        dec_seq, enc_seq, enc_mask, _ = prev_state
        attn, probs = self.attention(enc_seq, dec_seq, enc_mask)

        output = self.emb_out(prev_tokens)
        
        output = torch.cat((attn, output), 1)
        output = self.dec0(output, dec_seq)

        new_dec_state = [output, enc_seq, enc_mask, probs]

        output_logits = self.logits(output)
        return [new_dec_state, output_logits]