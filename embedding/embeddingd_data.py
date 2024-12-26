from transformers import AutoTokenizer, AutoModel, default_data_collator
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle



def preprocess_supervised_data(examples,tokenizer,max_length=512):
    model_inputs = {'input_ids': [],
                    'attention_mask':[],
                    }
    
    embedding_key = ""

    if 'question' in examples.keys():
        embedding_key = 'question'
    elif 'instruction' in examples.keys():
        embedding_key = 'instruction'

    for  i in examples[embedding_key]:
        res = tokenizer(i,padding='max_length',max_length=max_length,truncation=True)
        model_inputs['input_ids'].append(res['input_ids'])
        model_inputs['attention_mask'].append(res['attention_mask'])

    return model_inputs


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def hook_fn(module, input, output):
    dims = torch.tensor(output.size(0) * output.size(1))
    mag_norm = 500 / torch.sqrt(dims)
    output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    print(f"Forward hook: {module}")
    print(f"Input: {input}")
    print(f"Output: {output}")
    return output

def main():
    import argparse

# 创建解析器对象
    parser = argparse.ArgumentParser(description="sentence embedding")

    # 添加命令行参数
    parser.add_argument('--model_path', type=str, required=True, help='')
    parser.add_argument('--data_path', type=str, required=True, help='should be json list')
    parser.add_argument('--save_path', type=str, required=True, help='save to pkl file')
    parser.add_argument('--device', type=str, required=True, help='')   
    parser.add_argument('--batch_size', type=int, required=True, help='')   
    parser.add_argument('--max_length', type=int, required=True, help='')   
    parser.add_argument('--noise', type=bool, required=True, help='') 
    parser.add_argument('--use_cls', type=bool, required=True, help='')
    # 解析命令行参数
    args = parser.parse_args()
    # base setting
    model_path = args.model_path
    data_path = args.data_path
    save_path = args.save_path
    batch_size = args.batch_size
    max_length = args.max_length
    device = args.device
    noise = args.noise
    use_cls = args.use_cls

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path,)
    if noise:
        model.embeddings.register_forward_hook(hook_fn)

    # error check
    assert save_path.endswith('.pkl')


    # load data
    dataset = load_dataset("json", data_files=data_path)
    dataset2 = dataset.map(lambda examples: preprocess_supervised_data(examples, tokenizer, max_length), batched=True)
    the_dataloader = DataLoader(dataset2['train'],batch_size=batch_size, pin_memory=True,collate_fn=default_data_collator,shuffle=False)
    
    # get embedding / model inference
    embedding_list = []
    idx = 0
    model.eval()
    model.to(device)

    if use_cls:
        for step, batch in enumerate(tqdm(the_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(input_ids = batch['input_ids'],
                            attention_mask = batch['attention_mask'])
                sentence_embeddings = outputs[0][:, 0]
                res = F.normalize(sentence_embeddings, p=2, dim=1).tolist()
            for embed in res:
                embedding_list.append({"index":idx,"embedding":embed})
                idx+=1
    else:
        for step, batch in enumerate(tqdm(the_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(input_ids = batch['input_ids'],
                            attention_mask = batch['attention_mask'])
                res = F.normalize(mean_pooling(outputs,batch['attention_mask'])).tolist()
            for embed in res:
                embedding_list.append({"index":idx,"embedding":embed})
                idx+=1

    # save embedding
    # '/data/git_clone_project/RAG_for_sentence_embedding/embedding_vaild.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(embedding_list, f)


if __name__ == "__main__":
    main()
