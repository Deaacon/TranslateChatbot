import os
import re
import json
import torch
import pickle
import telebot

from subword_nmt.apply_bpe import BPE
from model import AttentiveModel
from vocab import Vocab


device = 'cpu'
bpe = BPE(open('data/bpe_rules.ru', encoding='utf-8'))

with open('data/voc.ru', 'rb') as file:
    inp_voc = pickle.load(file)
with open('data/voc.en', 'rb') as file:
    out_voc = pickle.load(file)

path = 'data/model.st'
model = AttentiveModel(inp_voc, out_voc)
model.load_state_dict(torch.load(path, map_location=device))
model.eval()

with open('conf.json') as f:
    token = json.load(f)['TOKEN']
bot = telebot.TeleBot(token)


@bot.message_handler(content_types=['text'])
def start_translate(message):
    if message.text == '/translate':
        bot.send_message(message.from_user.id, 'Введите текст на русском для перевода')
        bot.register_next_step_handler(message, translate)
    else:
        bot.send_message(message.from_user.id, 'Для старта напишите /translate')

def translate(message):
    text = message.text.strip().lower()
    text = re.sub('([.,!?()])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    model_input = bpe.process_line(text)
    translate_result = model.translate_lines([model_input])[0]

    bot.send_message(message.from_user.id, translate_result)
    bot.register_next_step_handler(message, translate)


bot.polling(none_stop=True, interval=0)