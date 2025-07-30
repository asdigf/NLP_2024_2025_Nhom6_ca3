#!/bin/bash

# Fix GLUE data downloader first
python3 download_glue_data.py --data_dir ./ --tasks MNLI,RTE

# Rename GLUE datasets to lowercase
[ -d "MNLI" ] && mv MNLI mnli || echo "MNLI directory missing - continuing"
[ -d "RTE" ] && mv RTE rte || echo "RTE directory missing - continuing"

# Function to handle downloads with retries and validation
download_dataset() {
    local url=$1
    local filename=$2
    echo "Downloading $filename from $url..."
    
    for i in {1..3}; do
        if wget -U "Mozilla/5.0" --no-check-certificate -O "$filename" "$url"; then
            echo "Download successful"
            return 0
        else
            echo "Attempt $i failed, retrying..."
            sleep 2
        fi
    done
    echo "ERROR: Failed to download $filename"
    return 1
}

# IMDB (working)
if [ ! -d "imdb" ]; then
    download_dataset "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz" imdb.tar.gz
    tar -zxvf imdb.tar.gz
    mv aclImdb imdb
    rm imdb.tar.gz
else
    echo "IMDB already exists, skipping"
fi

# SST-2 (handling case sensitivity)
[ -d "SST-2" ] && mv SST-2 sst2 || echo "SST-2 directory missing - continuing"

# FewNERD (fixed Hugging Face download)
if [ ! -d "fewnerd" ]; then
    pip install -q datasets pandas
    mkdir -p fewnerd
    python -c "from datasets import load_dataset; \
               ds = load_dataset('DFKI-SLT/few-nerd', 'supervised'); \
               ds['train'].to_csv('fewnerd/train.csv'); \
               ds['validation'].to_csv('fewnerd/dev.csv'); \
               ds['test'].to_csv('fewnerd/test.csv')"
    echo "FewNERD downloaded via Hugging Face"
else
    echo "FewNERD already exists, skipping"
fi

# SNLI (working)
if [ ! -d "snli" ]; then
    download_dataset "https://nlp.stanford.edu/projects/snli/snli_1.0.zip" snli.zip
    unzip snli.zip
    mv snli_1.0 snli
    rm snli.zip
else
    echo "SNLI already exists, skipping"
fi

# Yelp (working)
if [ ! -d "yelp" ]; then
    download_dataset "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz" yelp.tgz
    tar -zxvf yelp.tgz
    mv yelp_review_polarity_csv yelp
    rm yelp.tgz
else
    echo "Yelp already exists, skipping"
fi

echo "All available datasets processed successfully!"
cd ..