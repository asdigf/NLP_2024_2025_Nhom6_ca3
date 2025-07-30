''' Script for downloading all GLUE data.
Note: for legal reasons, we are unable to host MRPC.
You can either use the version hosted by the SentEval team, which is already tokenized, 
or you can download the original data from (https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi) and extract the data from it manually.
For Windows users, you can run the .msi file. For Mac and Linux users, consider an external library such as 'cabextract' (see below for an example).
You should then rename and place specific files in a folder (see below for an example).
mkdir MRPC
cabextract MSRParaphraseCorpus.msi -d MRPC
cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
rm MRPC/_*
rm MSRParaphraseCorpus.msi
1/30/19: It looks like SentEval is no longer hosting their extracted and tokenized MRPC data, so you'll need to download the data from the original source for now.
2/11/19: It looks like SentEval actually *is* hosting the extracted data. Hooray!
'''

import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile
import urllib.request as URLLIB
import io

# FIX 1: Corrected URL mappings (swapped QQP/STS fixed)
TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
TASK2PATH = {
    "CoLA":'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
    "SST":'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
    "QQP":'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',  # FIXED
    "STS":'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',      # FIXED
    "MNLI":'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
    "QNLI":'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
    "RTE":'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
    "WNLI":'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
    "diagnostic":'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'
}

# FIX 2: Updated MRPC URLs (official source is dead)
MRPC_TRAIN = 'https://raw.githubusercontent.com/microsoft/GLUE-Baselines/master/data/MRPC/train.tsv'
MRPC_TEST = 'https://raw.githubusercontent.com/microsoft/GLUE-Baselines/master/data/MRPC/test.tsv'
MRPC_DEV_IDS = 'https://raw.githubusercontent.com/microsoft/GLUE-Baselines/master/data/MRPC/dev_ids.tsv'

def download_and_extract(task, data_dir):
    print(f"Downloading {task}...")
    data_file = f"{task}.zip"
    
    # FIX 3: Add User-Agent header to prevent 403 errors
    opener = URLLIB.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    URLLIB.install_opener(opener)
    
    try:
        URLLIB.urlretrieve(TASK2PATH[task], data_file)
        with zipfile.ZipFile(data_file) as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"\tExtracted {task}")
    except Exception as e:
        print(f"\tError downloading {task}: {str(e)}")
    finally:
        if os.path.exists(data_file):
            os.remove(data_file)

# FIX 4: Simplified MRPC handling with direct TSV downloads
def format_mrpc(data_dir, path_to_data):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    os.makedirs(mrpc_dir, exist_ok=True)

    try:
        # Download preprocessed TSV files directly
        URLLIB.urlretrieve(MRPC_TRAIN, os.path.join(mrpc_dir, "train.tsv"))
        URLLIB.urlretrieve(MRPC_TEST, os.path.join(mrpc_dir, "test.tsv"))
        URLLIB.urlretrieve(MRPC_DEV_IDS, os.path.join(mrpc_dir, "dev_ids.tsv"))
        print("\tMRPC files downloaded successfully")
    except Exception as e:
        print(f"\tMRPC download failed: {str(e)}")

    
def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "diagnostic")):
        os.mkdir(os.path.join(data_dir, "diagnostic"))
    data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
    urllib.request.urlretrieve(TASK2PATH["diagnostic"], data_file)
    print("\tCompleted!")
    return

def get_tasks(task_names):
    task_names = task_names.split(',')
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='glue_data')
    parser.add_argument('--tasks', help='tasks to download data for as a comma separated string',
                        type=str, default='all')
    parser.add_argument('--path_to_mrpc', help='path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt',
                        type=str, default='')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == 'MRPC':
            format_mrpc(args.data_dir, args.path_to_mrpc)
        elif task == 'diagnostic':
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(task, args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))