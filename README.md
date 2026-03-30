## Relation Prediction using Revised KG-BERT

This work presents a **revised implementation of KG-BERT** for the task of **relation prediction** in Knowledge Graph Completion (KGC).  
The model leverages transformer-based contextual embeddings to classify the relation between a given head–tail entity pair.

### 🔹 Evaluation Protocol
To comprehensively evaluate the model performance, we adopt standard KGC metrics:

- **Mean Rank (MR)** ↓ (lower is better)  
- **Mean Reciprocal Rank (MRR)** ↑  
- **Hits@K (K = 1, 3, 5, 10)** ↑  

Both **Raw** and **Filtered** evaluation settings are reported:
- **Raw**: Considers all candidate relations  
- **Filtered**: Removes valid competing triples from ranking (more accurate)

---

### 🔹 Dataset Summary
- **Test Triples**: 20,466  
- **Number of Relations**: 237  

---



python run_bert_relation_prediction.py \
    --task_name kg \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir ./data/FB15K-237 \
    --bert_model bert-base-cased \
    --max_seq_length 25 \
    --train_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 20.0 \
    --output_dir ./output_FB15K-237/ \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 512 \
    --n_gpu 2



### 🔹 Results

| Metric            | Raw     | Filtered |
|------------------|--------:|---------:|
| **Mean Rank (MR)** | 1.0975 | 1.0872 |
| **MRR**            | 0.9737 | 0.9785 |
| **Hits@1**         | 0.9540 | 0.9631 |
| **Hits@3**         | 0.9937 | 0.9938 |
| **Hits@5**         | 0.9975 | 0.9975 |
| **Hits@10**        | 0.9990 | 0.9990 |

---



---

### 🔹 Detailed Metrics

```text
acc = 0.9540213036255253
eval_loss = 0.14142201337963342
filtered_Hits@1 = 0.9631095475422652
filtered_Hits@10 = 0.9990227694713183
filtered_Hits@3 = 0.9938434476693052
filtered_Hits@5 = 0.9975080621518616
filtered_MR = 1.0871689631584092
filtered_MRR = 0.9784580885333554
global_step = 21260
loss = 0.28156598808939515
raw_Hits@1 = 0.9540213036255253
raw_Hits@10 = 0.9990227694713183
raw_Hits@3 = 0.9936968630900029
raw_Hits@5 = 0.9974592006254276
raw_MR = 1.0975276067624353
raw_MRR = 0.9737347262088496



# KG-BERT: BERT for Knowledge Graph Completion

The repository is modified from [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and tested on Python 3.5+.


## Installing requirement packages

```bash
pip install -r requirements.txt
```

## Data

(1) The benchmark knowledge graph datasets are in ./data. 

(2) entity2text.txt or entity2textlong.txt in each dataset contains entity textual sequences.

(3) relation2text.txt in each dataset contains relation textual sequences.

## Reproducing results
 
### 1. Triple Classification

#### WN11

```shell
python run_bert_triple_classifier.py --task_name kg --do_train  --do_eval --do_predict --data_dir ./data/WN11 --bert_model bert-base-uncased --max_seq_length 20 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./output_WN11/  --gradient_accumulation_steps 1 --eval_batch_size 512
```

#### FB13

```shell
python run_bert_triple_classifier.py --task_name kg  --do_train  --do_eval --do_predict --data_dir ./data/FB13 --bert_model bert-base-cased --max_seq_length 200 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./output_FB13/  --gradient_accumulation_steps 1 --eval_batch_size 512
```


### 2. Relation Prediction

#### FB15K

```shell
python3 run_bert_relation_prediction.py --task_name kg  --do_train  --do_eval --do_predict --data_dir ./data/FB15K --bert_model bert-base-cased --max_seq_length 25 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 20.0 --output_dir ./output_FB15K/  --gradient_accumulation_steps 1 --eval_batch_size 512
```

### 3. Link Prediction

#### WN18RR

```shell
python3 run_bert_link_prediction.py --task_name kg  --do_train  --do_eval --do_predict --data_dir ./data/WN18RR--bert_model bert-base-cased --max_seq_length 50 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 5.0 --output_dir ./output_WN18RR/  --gradient_accumulation_steps 1 --eval_batch_size 5000
```

#### UMLS

```shell
python3 run_bert_link_prediction.py --task_name kg  --do_train  --do_eval --do_predict --data_dir ./data/umls --bert_model bert-base-uncased --max_seq_length 15 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 5.0 --output_dir ./output_umls/  --gradient_accumulation_steps 1 --eval_batch_size 135
```

#### FB15k-237

```shell
python3 run_bert_link_prediction.py --task_name kg  --do_train  --do_eval --do_predict --data_dir ./data/FB15k-237 --bert_model bert-base-cased --max_seq_length 150 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 5.0 --output_dir ./output_FB15k-237/  --gradient_accumulation_steps 1 --eval_batch_size 1500
```




> 🔗 **Reference Implementation**: This work builds upon the original KG-BERT codebase available at https://github.com/yao8839836/kg-bert
