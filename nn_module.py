import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import DistilBertModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
    model_transf = DistilBertModel.from_pretrained('distilbert-base-cased')
    model_transf = model_transf.to(device)
    return tokenizer, model_transf


@torch.inference_mode() # декократор для отключения подсчета градиентов
def take_sample(test_comment: str):
    tokenizer, model_transf = load_tokenizer()

    model_transf.eval()

    sample_encoding = tokenizer.batch_encode_plus([test_comment],
                                                  add_special_tokens=True,
                                                  return_token_type_ids=False,
                                                  truncation=True)

    sample_encoding['input_ids'] = torch.tensor(sample_encoding['input_ids']).to(device)
    sample_encoding['attention_mask'] = torch.tensor(sample_encoding['attention_mask']).to(device)
    embeddings = model_transf(**sample_encoding)['last_hidden_state'][:, 0, :].to(device)
    res = embeddings
    print(test_comment)

    return res

@torch.inference_mode()  # декократор для отключения подсчета градиентов
def predict_sample(model: nn.Module, text_input: str):
    model.eval()

    x = text_input
    preds = model(x)
    rounded_preds = torch.sigmoid(preds)
    preds_class = torch.argmax(rounded_preds, 1)
    preds_class = preds_class.cpu().numpy()[0]

    return preds_class


def predict(text: str):
    model_nn = load_model('embeddings.pt')
    test = take_sample(text)
    pred_class = predict_sample(model_nn, test)

    return (f'Ожидаемый уровень тяжести - {pred_class} (по шкале 1->5)')
