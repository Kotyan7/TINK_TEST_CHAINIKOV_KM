import json
import time
import numpy as np
import pandas as pd
from flask import Flask, request, abort, Response
import time
import torch
import re
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification
import jamspell


app = Flask(__name__)
corrector = jamspell.TSpellCorrector() # корректор опечаток
corrector.LoadLangModel('en.bin') # предобученную модель взял здесь

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


print("Model loading...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=77)
model.load_state_dict(torch.load('my_model_loss_with_weight.pt'))
model.eval()
model.to(device)
print("Model loaded!")


dict_classes = json.load( open( "label_to_text.json" ) )



def predict_class(model, tokenizer, corrector, text, dict_classes, device):
    model.eval()
    if len(text) < 2:
        return 'text too small'
    query = query = pd.Series(text)
    query.apply(corrector.FixFragment)
    query = query.str.lower().str.replace("can't", 'can not').str.replace("didn't", 'did not').str.replace("hasn't", 'has not')\
        .str.replace("won't", 'will not').str.replace("wasn't", 'was not').str.replace("isn't", 'is not').str.replace("doesn't", 'does not')\
        .str.replace("haven't", 'have not').str.replace("don't", 'do not').str.replace("shouldn't", 'should not')\
        .str.replace("aren't", 'are not').str.replace("wouldn't", 'would not').str.replace("couldn't", 'could not')\
        .str.replace("i'm", 'i am').str.replace("i'd", 'i would').str.replace("it's", 'it is')\
        .str.replace("what's", 'what is').str.replace("i'll", 'i will')\
        .str.replace("there's", 'there is').str.replace("where's", 'where is').str.replace("that's", 'that is')\
        .str.replace("you're", 'you are').str.replace("they're", 'they are').str.replace("'ve", ' have').str.replace("\"", '\'')\
        .str.replace('€','$').str.replace('£','$').str.replace(r'[^a-z0-9\'\?\!\.\,\- ]',r' ').str.replace(r'\s{1,}',r' ')\
        .str.replace(r'([^a-z0-9 ])',r' \1 ').str.strip()
    query.apply(corrector.FixFragment)
    if (query.str.len()==0).values:
        return 'text include small'
    tokens = tokenizer(query.tolist(), return_tensors='pt', padding=True)
    input = tokens.input_ids.to(device)
    mask = tokens.attention_mask.to(device)
    logits = model(input, mask).logits
    y_pred = logits.argmax(dim=-1).cpu().tolist()
    return dict_classes[str(y_pred[0])]



@app.route('/', methods=['GET'])
def start():

    return 'Make your query...'


@app.route('/classify', methods=['POST'])
def predict():
    if not (request.json and 'text' in request.json):
        abort(Response("Your request should be in JSON format: {'text':[texts]}\n"))
    user_query = request.json['text']
    #:todo

    print("Data uploaded.")
    try:
        prediction = predict_class(model, tokenizer, corrector, user_query, dict_classes)
    except Exception as ex:
        prediction = 'Произошла ошибка'
    return prediction + "\n"


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")