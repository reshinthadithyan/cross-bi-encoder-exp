from torch.utils.data import DataLoader, Dataset 
import logging
import json
from bs4 import BeautifulSoup
from pprint import pprint
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_file(file_path:str):
    with open(file_path, 'r') as f:
        return json.load(f)

def parse_html_to_context(body:str) -> list:
    """
    Return the list of <p> tags in the given body
    """
    html_parsed = BeautifulSoup(body, 'html.parser')
    strings = []
    children = html_parsed.children
    for child in children:
        if child.name == "p":
            strings.append(child.text)
    return strings

def clean_parsed_code(code_block:str):
    code_block = code_block.replace("<code>","").replace("</code>","")
    if " " not in code_block:
        return ""
    else:
        return code_block

def parse_html_block_for_code(html_block:str)->list[str]:

    parser = BeautifulSoup(html_block,"html.parser")
    parsed_code_list : list[str] =  parser.find_all("code")
    parsed_code_list = [clean_parsed_code(code.text) for code in parsed_code_list]
    parsed_code_list = [code for code in parsed_code_list if code != ""]
    return parsed_code_list


class CodeReviewDataset(Dataset):
    def __init__(self,dataset_path,config_str:str="Review"):
        """
        Class for Dataloading with Code Review Stack Exchange Dataset.
        args:
            dataset_path (str): path to the CodeReviewSE Dataset.
            config_str (str): config file type. One of ['Review','Explain','Diff']
        """
        self.dataset = load_json_file(dataset_path)
        self.dataset_keys = list(self.dataset.keys())
        self.dataset_list = [] #List of processed_datapoints
        self.pass_count = 0
        self.total_process_count = 0
        logger.info(f"The dataset is of {len(self.dataset_keys)} size.")
        logger.info("Successully loaded the raw dataset..")
    
    def __len__(self):
        return len(self.dataset)


    def preprocess_ind(self,datapoint:dict):
        """
        Function to preprocess individual datapoint.
        args:
            datapoint (dict): Datapoint is a dictionary.
        """
        try:
            if datapoint["meta_data"]["AcceptedAnswerId"]:
                accepted_id = datapoint["meta_data"]["AcceptedAnswerId"]
            else:
                accepted_id = None
            title = datapoint["meta_data"]["Title"]
            question_body = datapoint["body"] #Question body from the datapoint
            question_body_contexts = parse_html_to_context(question_body)        
            question_desc = question_body_contexts[0]

            question_code = parse_html_block_for_code(question_body)
            if len(question_code) > 0:
                question_code_block =  max(question_code,key=len) #The longest code block from the question contains full context.
            else:
                question_desc = "\n".join(question_body_contexts)
                question_code_block = ""
            score_list = []
            for answer in datapoint["answers"]:
                score = answer["meta_data"]["Score"]
                score_list.append(int(score))

            score_avg = max(score_list)
            for answer in datapoint["answers"]:
                self.total_process_count += 1
                answer_body = answer["body"]
                answer_body_contexts = parse_html_to_context(answer_body)
                answer_desc = "\n".join(answer_body_contexts)

                answer_code = parse_html_block_for_code(answer_body)
                if len(answer_code) > 0:
                    answer_code_block =  max(answer_code,key=len) #The longest code block from the answer contains full context.
                else:
                    answer_code_block = ""
                norm_score = int(answer["meta_data"]["Score"]) / score_avg
                if accepted_id != None:
                    if answer["meta_data"]["Id"] == accepted_id:
                        is_accepted = 1
                    else:
                        is_accepted = 0
                else:
                    is_accepted = 0

                datapoint_dict = {
                    "title": title,
                    "question_desc" : question_desc,
                    "question_code": question_code_block,
                    "answer_desc" : answer_desc,
                    "answer_code_block":  answer_code_block,
                    "norm_score": norm_score,
                    "score" : answer["meta_data"]["Score"],
                    "is_accepted": is_accepted,
                    "id" : datapoint["meta_data"]["Id"],
                    "answer_id" : answer["meta_data"]["Id"]
                }
                self.dataset_list.append(datapoint_dict)
                self.pass_count += 1
        except:
            pass


    def process_dataset(self,output_path:str="dataset/CodeReviewSE_CrossEncoder.json"):

        for datapoint_index in tqdm(self.dataset_keys):
            datapoint_dict = self.dataset[datapoint_index]
            self.preprocess_ind(datapoint_dict)
        logger.info(f"{self.total_process_count}, {self.pass_count}")
        
        with open(output_path, "w") as f:
            json.dump(self.dataset_list, f, indent=2)










if __name__ == '__main__':
    dataset_path = "/Users/reshinthadithyan/master/research/code-research/eai/CodeReviewSE/dataset/CodeReviewSE_clean.json"
    code_review_dataset = CodeReviewDataset(dataset_path)
    code_review_dataset.process_dataset()
    # code_review_dataset = CodeReviewDataset(dataset_path,None)
    # print(code_review_dataset.preprocess_ind(0))