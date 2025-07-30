import os
import sys
sys.path.append(".")

import argparse
import csv
from transformers import GenerationMixin
from re import template
from process_data import load_dataset
from dect_verbalizer import DecTVerbalizer
from dect_trainer import DecTRunner
from openprompt.prompts import ManualTemplate
from openprompt.pipeline_base import PromptForClassification
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader
from openprompt.data_utils import FewShotSampler
from openprompt.plms import load_plm, LMTokenizerWrapper, T5LMTokenizerWrapper, T5TokenizerWrapper
from openprompt.data_utils.utils import InputExample
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, AutoModelForSeq2SeqLM, AutoConfig
import torch
from transformers import DebertaTokenizer, DebertaForMaskedLM
parser = argparse.ArgumentParser("")


parser.add_argument("--model", type=str, default='roberta', help="plm name")
parser.add_argument("--size", type=str, default='large', help="plm size")
parser.add_argument("--type", type=str, default='mlm', help="plm type")
parser.add_argument("--model_name_or_path", default='roberta-large', help="default load from Huggingface cache")
parser.add_argument("--shot", type=int, default=16, help="number of shots per class")
parser.add_argument("--seed", type=int, default=0, help="data sampling seed")
parser.add_argument("--template_id", type=int, default=0)
parser.add_argument("--dataset", type=str, default='sst2')
parser.add_argument("--max_epochs", type=int, default=30, help="number of training epochs for DecT")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for train and test")
parser.add_argument("--proto_dim", type=int, default=128, help="hidden dimension for DecT prototypes")
parser.add_argument("--model_logits_weight", type=float, default=1, help="weight factor (\lambda) for model logits")
parser.add_argument("--lr", default=0.01, type=float, help="learning rate for DecT")
args = parser.parse_args()

def load_model(name, size, path):
    if name == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(path, force_download=False)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
           
        )
        wrapper = LMTokenizerWrapper
        if size == "7b":
            hidden_size = 4096
        elif size == "13b":
            hidden_size = 5120
    elif name == "alpaca":
        tokenizer = LlamaTokenizer.from_pretrained(path, force_download=True)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(path, force_download=True)
        wrapper = LMTokenizerWrapper
        hidden_size = 4096
    elif name == "vicuna":
        tokenizer = LlamaTokenizer.from_pretrained(path, force_download=True)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(path, force_download=True)
        wrapper = LMTokenizerWrapper
        hidden_size = 5120
    
    elif name == "deberta":
        # Xác định phiên bản DeBERTa (v2 hay v3)
        if "v3" in path.lower():
            from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM
            tokenizer = DebertaV2Tokenizer.from_pretrained(path, use_auth_token=True)
            model = DebertaV2ForMaskedLM.from_pretrained(path, use_auth_token=True)
        else:
            from transformers import DebertaTokenizer, DebertaForMaskedLM
            tokenizer = DebertaTokenizer.from_pretrained(path)
            model = DebertaForMaskedLM.from_pretrained(path)

        hidden_size = model.config.hidden_size

        # Sử dụng MLMTokenizerWrapper cho DeBERTa
        from openprompt.plms import MLMTokenizerWrapper
        wrapper = MLMTokenizerWrapper

        # Tối ưu hóa bộ nhớ
        model.gradient_checkpointing_enable()

        # Sử dụng FP16 nếu GPU hỗ trợ
        if torch.cuda.is_available():
            model = model.half()
            print("Using FP16 precision for DeBERTa")

        print(f"Loaded DeBERTa model: {path}")
        print(f"Hidden size: {hidden_size}, Wrapper: {wrapper.__name__}")
    elif name == "distilbert":
        from transformers import DistilBertTokenizer, DistilBertForMaskedLM
        tokenizer = DistilBertTokenizer.from_pretrained(path)
        model = DistilBertForMaskedLM.from_pretrained(path)
        hidden_size = model.config.dim
        from openprompt.plms import MLMTokenizerWrapper
        wrapper = MLMTokenizerWrapper  
    elif name == "bge":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path)
        hidden_size = model.config.hidden_size

        from openprompt.plms import MLMTokenizerWrapper
        wrapper = MLMTokenizerWrapper

        model.gradient_checkpointing_enable()
        if torch.cuda.is_available():
            model = model.half()
            print("Using FP16 precision for BGE")
        
        print(f"Loaded BGE model: {path}")
        print(f"Hidden size: {hidden_size}, Wrapper: {wrapper.__name__}")

    elif name == "gemma":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            path,#  "D:/gemma",         
            torch_dtype=torch.float16,
            attn_implementation="sdpa"   
                       
        )

        from openprompt.plms import LMTokenizerWrapper
        wrapper = LMTokenizerWrapper

        # ✅ Lấy hidden size đúng cách
        if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
            hidden_size = model.config.hidden_size
        else:
            hidden_size = getattr(model.config, "hidden_size", 4096)

        print(f"Loaded Gemma model: {path}")
        print(f"Hidden size: {hidden_size}, Wrapper: {wrapper.__name__}")

    else:
        model, tokenizer, model_config, wrapper = load_plm(args.model, args.model_name_or_path)
        hidden_size = model_config.hidden_size
    
    
    return model, tokenizer, hidden_size, wrapper

def build_dataloader(dataset, template, verbalizer, tokenizer, tokenizer_wrapper_class, batch_size):
    label_map = verbalizer.label_words

    for example in dataset:
        if not hasattr(example, "meta"):
            example.meta = {}

        if "label" not in example.meta:
            label_idx = example.label
            label_token = label_map[label_idx][0] if isinstance(label_map[label_idx], list) else label_map[label_idx]
            example.meta["label"] = label_token

    dataloader = PromptDataLoader(
        dataset=dataset,
        template=template,
        verbalizer=verbalizer,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=tokenizer_wrapper_class,
        decoder_max_length=128,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader





def main():
    os.makedirs("logger", exist_ok=True)
    # set hyperparameter
    datasets = args.dataset.split(',')
    data_path = datasets[0].split('-')[0]
    if data_path == "mnli":
        args.model_logits_weight = 1
    elif data_path == "fewnerd":
        args.model_logits_weight = 1/16
    else:
        args.model_logits_weight = 1/args.shot

    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(datasets[0])
    
    set_seed(123)
    # sample data
    sampler = FewShotSampler(
        num_examples_per_label = args.shot,
        also_sample_dev = True,
        num_examples_per_label_dev = args.shot)

    train_sampled_dataset, valid_sampled_dataset = sampler(
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        seed = args.seed
    )

    # Load model
    plm, tokenizer, hidden_size, plm_wrapper_class = load_model(args.model, args.size, args.model_name_or_path)
    
    # Tự động điều chỉnh tham số cho DeBERTa-xxlarge
    if args.model == "deberta" and "xxlarge" in args.model_name_or_path.lower():
        # Giảm batch size
        original_batch_size = args.batch_size
        args.batch_size = max(4, args.batch_size // 4)
        
        # Điều chỉnh proto_dim nếu cần
        if args.proto_dim > hidden_size:
            args.proto_dim = hidden_size
            
        print(f"Adjusted parameters for DeBERTa-xxlarge:")
        print(f"  Batch size: {original_batch_size} -> {args.batch_size}")
        print(f"  Proto dim: {args.proto_dim}")
    
    # make dir to save hiddens and logits
    save_dir = f"vectors/{args.model}/{args.size}/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    # define template and verbalizer
    # define prompt
    dataset_name = args.dataset.lower()
    if args.type == "lm":
        if dataset_name in ["agnews"]:
            template_text = 'Classify the topic: {"placeholder":"text_a"}\nTopic: {"meta":"label"}'

        elif dataset_name in ["dbpedia"]:
            template_text = 'Classify the topic: {"placeholder":"text_a"}\nTopic: {"meta":"label"}'

        elif dataset_name in ["imdb"]:
            template_text = 'Classify the sentiment of the movie review: {"placeholder":"text_a"}\nThe sentiment is {"meta":"label"}.'

        elif dataset_name in ["yelp"]:
            template_text = 'Review: {"placeholder":"text_a"}\nSummary: This was {"meta":"label"}'


        elif dataset_name in ["sst2"]:
            # Optimized for sentiment analysis - combines instruction clarity with summary pattern
            template_text = 'Review: {"placeholder":"text_a"}\nSummary: This was {"meta":"label"}'

        elif dataset_name in ["yahoo"]:
        # Enhanced template for Yahoo with explicit category list
            template_text = (
                'Classify the question: {"placeholder":"text_a"}\n'
                'Choose from [society, science, health, education, computers, sports, business, entertainment, family, politics]\n'
                'Category: {"meta":"label"}'
            )

        elif dataset_name in ["mnli-m","mnli-mm", "rte", "snli"]:
           template_text = '{"placeholder":"text_a"} {"placeholder":"text_b"} Does the first sentence entail the second? The answer is {"meta":"label"}.'


        elif dataset_name == "fewnerd":
            template_text = 'Identify entity type: {"placeholder":"text_a"}\nEntity Type: {"meta":"label"}'
    

        # =======================================================================

        else:
            raise ValueError(f"No template defined for dataset: {dataset_name}")

        template = ManualTemplate(
            tokenizer=tokenizer,
            text=template_text
        )



    else:
        template = ManualTemplate(
        tokenizer=tokenizer).from_file(f"scripts/{args.type}/{data_path}/manual_template.txt", choice=args.template_id)
    verbalizer = DecTVerbalizer( 

        tokenizer=tokenizer, 
        classes=Processor.labels, 
        hidden_size=hidden_size, 
        lr=args.lr, 
        mid_dim=args.proto_dim, 
        epochs=args.max_epochs, 
        model_logits_weight=args.model_logits_weight,
        save_dir=save_dir).from_file(f"scripts/{args.type}/{data_path}/manual_verbalizer.json")
    print("Verbalizer.label_words:", verbalizer.label_words)
    print("Verbalizer.label_words_ids:", verbalizer.label_words_ids)

    # load prompt’s pipeline model
    prompt_model = PromptForClassification(plm, template, verbalizer)
            
    # process data and get data_loader
    train_dataloader = build_dataloader(train_sampled_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size) if train_dataset else None
    valid_dataloader = build_dataloader(valid_sampled_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size) if valid_dataset else None

    subset_ratio = 0.1
    subset_size = int(len(test_dataset) * subset_ratio)
    test_subset = test_dataset[:subset_size]
    test_dataloader = build_dataloader(test_subset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size)

    #test_dataloader = build_dataloader(test_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size)

    calibrate_dataloader = PromptDataLoader(
        dataset = [InputExample(
            guid=str(0),
            text_a="",
            text_b="",
            meta={"label": verbalizer.label_words[0][0]},  # 👈 Thêm dòng này
            label=0
        )],
        template = template,
        tokenizer = tokenizer,
        tokenizer_wrapper_class=plm_wrapper_class,
        decoder_max_length=128,
    )


    runner = DecTRunner(
        model = prompt_model,
        train_dataloader = train_dataloader,
        valid_dataloader = valid_dataloader,
        test_dataloader = test_dataloader,
        calibrate_dataloader = calibrate_dataloader,
        id2label = Processor.id2label,
        verbalizer = verbalizer
    )                                   

        
    res = runner.run()
    print('Dataset: {} | Shot: {} |'.format(args.dataset, args.shot))
    print(res)
    log_file = f"logger/{args.dataset}_shot{args.shot}_log.txt"
    with open(log_file, "a") as f:
        f.write(f"Dataset: {args.dataset} | Shot: {args.shot} | Seed: {args.seed}\n")
        for metric, value in res.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("------\n")
    return res["dect acc"]
    


if __name__ == "__main__":
    res = []
    for seed in range(1):
        args.seed = seed
        res.append(main())
    print('Average: {}'.format(sum(res)/len(res)*100))
    with open(f"logger/{args.dataset}_shot{args.shot}_log.txt", "a") as f:
        f.write(f"Average Accuracy over {len(res)} seeds: {sum(res)/len(res)*100:.2f}%\n")
        f.write("=========\n")
    
