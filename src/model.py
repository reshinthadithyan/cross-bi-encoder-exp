from unittest.util import strclass
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)
import random
import json
from accelerate import Accelerator
import numpy as np
import torch
import torch.nn as nn
import logging
from datasets import load_dataset
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(random_seed):
    """
    Random number fixed
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)

def preprocess_function_aligned_data(datapoint):
    """
    Specific preprocessing function for the aligned dataset
    args:
        datapoint (dict) : A datapoint dict would contain all the correponding dicts.
    """
    prompt_string = ""
    for key in datapoint:
        prompt_string += "[" + key + "]"
        prompt_string += datapoint[key]
    return prompt_string

def preprocess_function_unaligned_data(datapoint):
    pass

#      "dummy_data" : {
#     "title": "Database connection in constructor and destructor",
#     "question_desc": "I am playing with different ways to do database interaction in PHP, and one of the ideas I have been playing with is connecting to the DB in the constructor and disconnecting in the destructor. This is the code from my Database class.",
#     "question_code": "function __construct()\n{\n  $this->link = mysql_connect($this->server.':'.$this->port, $this->username);\n  if(!$this->link)\n    die('Could not connect: '.mysql_error());\n\n  if(!mysql_select_db($this->database, $this->link))\n    die('Could not select database: '.mysql_error());\n}    \n\nfunction __destruct()\n{\n  if(mysql_close($this->link))\n    $this->link = null; \n}\n",
#     "answer_desc": "You could use MySQLi (PHP extension) which is class based by default instead of MySQL. It \nis very easy to set up multiple connections. You are, however, required to know the connection you are querying always.\nCongrats with the first question.",
#     "answer_code_block": "",
#     "norm_score": 1.0,
#     "score": "19",
#     "is_accepted": 0,
#     "id": "1",
#     "answer_id": "3"
#   }

def preprocess_ce_data(datapoint):
    """
    Preprocess CrossEncoder Data
    """
    prompt_string = f"""<title>{datapoint['title']}
<question> : {datapoint['question_desc']}
<answer_desc> : {datapoint['answer_desc']}
<code> : {datapoint['question_code']}"""
    output = datapoint["norm_score"]
    return prompt_string, [output]

def squeeze_tree(tensor_data):
    return {k: tensor_data[k].squeeze(0) for k in tensor_data}
class CEDataset(Dataset):
    def __init__(self,tokenizer,dataset_path:str,dataset_flag:str) -> None:
        self.dataset = json.load(open(dataset_path,"r"))
        logging.info(f"Succesfully loaded the dataset from {dataset_path} of length {len(self.dataset)}")
        self.tokenizer = tokenizer
        self.dataset_flag =  dataset_flag

    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        datapoint = self.dataset[idx]
        if self.dataset_flag == "aligned":
            input,output = preprocess_function_aligned_data(datapoint)
        elif self.dataset_flag == "ce":
            input,output = preprocess_ce_data(datapoint)
        input = squeeze_tree(self.tokenizer(input, padding="max_length",
                         truncation=True, return_tensors='pt',return_token_type_ids=True))
        output = torch.Tensor(output)
        return input,output





class CrossEncoder(nn.Module):
    def __init__(self,model_name:str="microsoft/codebert-base") -> None:
        super().__init__()
        self.config  = AutoConfig.from_pretrained(model_name) 
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.classifier = torch.nn.Linear(self.config.hidden_size,1)

    def forward(self,input_ids, attention_mask,token_type_ids):
        """
        Forward function that returns pooled_output and logits.
        """
        outputs = self.encoder(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return pooled_output,logits



def train(args):


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)    
    ce_model = CrossEncoder(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    logger.info(f"Successfully loaded the model to memory")

    ce_dataset = CEDataset(tokenizer=tokenizer,dataset_path=args.dataset_file_path,dataset_flag=args.dataset_flag)
    train_len = int(len(ce_dataset)*0.9)
    trainset, testset = random_split(ce_dataset, [train_len,len(ce_dataset)-train_len])
    trainloader = DataLoader(trainset,args.batch_size,shuffle = True)
    testloader = DataLoader(testset,args.batch_size,shuffle = True)

    # print(next(iter(trainloader))[0]["input_ids"].size())
    # print(next(iter(trainloader))[1].size())
    #Accelerate Init

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in ce_model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in ce_model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        # eps=args.adam_epsilon
    )
    t_total = (
        len(trainloader)
        // args.gradient_accumulation_steps
        * args.num_epochs
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.if_wandb:
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(project_name="crossencoder")
    else:
        accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Model : {args.model_name}")


    trainloader, testloader, ce_model, optimizer = accelerator.prepare(trainloader, testloader, ce_model, optimizer)
    accelerator.print(f"Converted everything to {device}")
    loss_fn = nn.MSELoss()
    #Start Training
    ce_model.train()
    progress_bar = tqdm(range(args.num_epochs),desc="Epoch", position=0)
    train_progress_bar = tqdm(trainloader,desc="Train Loop", position=1)
    test_progress_bar = tqdm(testloader,desc="Eval Loop", position=1)
    steps = 0
    for epoch in progress_bar:
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0

        ce_model.train()
        for batch in train_progress_bar:
            pooled_output,logits = ce_model(**batch[0])
            loss = loss_fn(logits, batch[1])
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()         
            optimizer.zero_grad()
            epoch_train_loss += loss.item()
            if args.if_wandb:
                accelerator.log({"train_loss":loss})


        ce_model.eval()  
        for batch in test_progress_bar:
            with torch.no_grad():
                pooled_output,logits = ce_model(**batch[0])
                loss = loss_fn(logits, batch[1])
                epoch_test_loss += loss.item()    
                if args.if_wandb:
                    accelerator.log({"eval_loss":loss})


    
        accelerator.print(f"Epoch {epoch}/{args.num_epochs} Training Loss :  {epoch_train_loss/len(trainloader)} Eval Loss : {epoch_test_loss/len(trainloader)} Eval Loss : {epoch_test_loss/len(testloader)}")

        if args.if_wandb:
            accelerator.log({
                "train_epoch_loss" : epoch_train_loss/len(trainloader),
                "eval_epoch_loss": epoch_test_loss/len(testloader)
            })
        progress_bar.update(1)               
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(ce_model)

        torch.save(unwrapped_model,args.output_dir+"/ce_model.pt")
        if accelerator.is_main_process:
            accelerator.print(f"Sucessfully saved model and tokenizer in {args.output_dir}")
            tokenizer.save_pretrained(args.output_dir)
            config.save_pretrained(args.output_dir)
        if args.if_wandb:
            accelerator.end_training()





             





    





    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__name__)
    
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--dataset_file_path",type=str,default="dataset/CodeReviewSE_Trial.json")
    parser.add_argument("--dataset_flag",type=str,default="ce")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--if_wandb", type=bool, default=False)
    
    parser.add_argument("--warmup_steps", type=int, default=10)

    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate (default: 1e-5)")

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="gradient accumulation steps (default: 1)")
    parser.add_argument("--output_dir",type=str,default="trained_models")
    parser.add_argument("--batch_size",type=int,default=2)


    args = parser.parse_args()
    
    #TODO(reshinth) : Set seed function
    set_seed(args.random_seed)
    train(args)
    


