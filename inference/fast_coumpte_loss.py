from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,default_data_collator
import torch
from peft import PeftModel
from transformers import GenerationConfig
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

abs_path = "/data/home/chenpz/git_clone_project"
model_path = f"{abs_path}/All_base_model/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/d3aa29f914761e8ea0298051fbaf8dd173e94db5"
data_path = f"/data/home/chenpz/git_clone_project/nlpData/p3/anli_r3_json_file/anli_r3_train_FFP_all.json"
# adpter_path = f"/data/home/chenpz/git_clone_project/LLaMA-Factory/saves/llama3-8b-anli_r3_train_kcg_addHighPPL_first5000_data_gas=5_lr=1e-4/checkpoint-60"
output_file = '/data/home/chenpz/git_clone_project/jupyter_notebook_test/output/anli_r3_loss_FFP_all_int4.jsonl'

output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


device = 'cuda:1'
prompt = \
'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction} {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>'''



nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype= None
)


model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=nf4_config,
                                             device_map = device)

# model = PeftModel.from_pretrained(
#                     model, adpter_path, is_trainable=False
#                     )


tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False,split_special_tokens=False,
                                          padding_side="left",
                                          **{'trust_remote_code': True, 'cache_dir': None, 'revision': 'main', 'use_auth_token': None})
tokenizer.pad_token = '<|eot_id|>'

dataset = load_dataset("json", data_files= data_path)

# def preprocess_supervised_data(examples):
#     model_inputs = {'input_ids': [],
#                     'attention_mask':[],
#                     # 'prompt':[]
#                     }
#     for  instruction,input,output in zip(examples['instruction'],examples['input'],examples['output']):
#            text = prompt.format(instruction = instruction, input = input, output = output)
#            res = tokenizer(text,padding='max_length',max_length=1300,truncation=True)
#            model_inputs['input_ids'].append(res['input_ids'])
#            model_inputs['attention_mask'].append(res['attention_mask'])
#           #  model_inputs['prompt'].append(text)
#     return model_inputs


IGNORE_INDEX = -100

question_prompt = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

answer_prompt = '''{output}<|eot_id|>'''

def preprocess_supervised_data(examples) :
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        max_length = 1300
        for  instruction,input,output in zip(examples['instruction'],examples['input'],examples['output']):

            query = question_prompt.format(instruction = instruction, input = input)
            response= answer_prompt.format(output = output)
            
            input_ids, labels = [], []
            source_ids, target_ids = tokenizer([query,response])['input_ids']
            input_ids += source_ids + target_ids
            if len(input_ids) > max_length:
                 print('warning')
            input_ids = input_ids[:max_length]

            labels += [IGNORE_INDEX] * len(source_ids) + target_ids
            labels = labels[:max_length]
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        # Padding: Ensure that all input sequences have the same length
        for i in range(len(model_inputs["input_ids"])):
            padding_length = max_length - len(model_inputs["input_ids"][i])
            
            # Left padding input_ids and attention_mask
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * padding_length + model_inputs["input_ids"][i]
            model_inputs["attention_mask"][i] = [0] * padding_length + model_inputs["attention_mask"][i]

            # Left padding labels with IGNORE_INDEX
            model_inputs["labels"][i] = [IGNORE_INDEX] * padding_length + model_inputs["labels"][i]
        
        return model_inputs



dataset2 = dataset.map(preprocess_supervised_data,batched=True,remove_columns=['output', 'input', 'instruction'],num_proc=16)

print(tokenizer.decode(dataset2['train'][0]['input_ids']))

eval_dataloader = DataLoader(dataset2['train'],batch_size=4, pin_memory=True,collate_fn=default_data_collator,shuffle=False)


from tqdm import tqdm
model.eval()
loss_list =[]
loss_fn = torch.nn.CrossEntropyLoss(
    reduction='none'
    ) # 不要平均，保留每个 token 的 loss
with open(output_file, 'w') as file:  # Open the file in write mode before the loop
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            # outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()  # 移动 logits 以对齐预测
            shift_labels = batch['labels'][:, 1:].contiguous()  # 移动 labels
            
            # 计算每个 token 的 loss
            loss_per_token = loss_fn(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
            # reshape 回到 batch 大小
            loss_per_token = loss_per_token.view(shift_labels.size())
            
            means = []
            for row in loss_per_token:
                non_zero_elements = row[row != 0]  # 提取非零元素
                if non_zero_elements.numel() > 0:  # 确保非零元素存在
                    mean_non_zero = non_zero_elements.mean()
                    means.append(mean_non_zero.item())  # 将均值添加到列表中
                else:
                    means.append(0)  # 如果该行全为零，均值为0
            
            for i, loss in enumerate(means):
                sample_id = step * eval_dataloader.batch_size + i  # 用于区分每个样本的索引
                add_item = {'id':sample_id,'loss':loss}
                file.write('%s\n' % add_item)