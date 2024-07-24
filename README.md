# EMBI
Epitope-MHC-Bert-Immunogenicity (EMBI) is a comprehensive deep learning framework designed to predict the immunogenicity of epitope peptides. This prediction paradigm utilizes two pre-trained Bidirectional Encoder Representations from Transformers (BERTs), one trained on epitope sequences and the other on MHC-I pseudo-sequences. The models are subsequently fine-tuned on task-specific data, such as peptide-MHC binding, pMHC presentation, and TCR recognition.
![image](https://github.com/TencentAILabHealthcare/EMBI/blob/master/workflow/workflow.jpg)
## Setup and Installation
To ensure the successful execution of the EMBI, the installation of necessary packages is crucial. This can be achieved with the following command:
```bash
conda create -n EMBI python=3.10
conda activate EMBI
git clone https://github.com/TencentAILabHealthcare/EMBI.git
cd EMBI
pip install -r requirements.txt
```
## Hardware and Software Requirements
The proposed model's computational effectiveness was assessed on a high-performance computing workstation comprising the following specifications: dual RTX 3090 graphics processing units, an i9-10920X processor, 128 GB of system memory, and a cumulative 48 GB graphics processing memory. This model operates under the Ubuntu operating system, version 18.04.
## Training and Utilization of EMBI
EMBI's training for peptide immunogenicity prediction involves the Masked Amino Acid (MAA) task. The specifics of each training are found in the `./config` folder.
### 1. Masked Amino Acid task
The Masked Amino Acid task constitutes a self-supervised learning process for EpitopeBert and MHCBert, using a common tokenizer (each unique amino acid represents a token). This must be executed initially. 
Use the following command to generate the pre-trained model of epitope sequences:
```bash
python bert_pretrain_maa_main.py --config config/bert_pretrain_maa_common_Epitope.json
```
To obtain the pre-trained model of MHC pseudo sequences, use this command:
```bash
python bert_pretrain_maa_main.py --config config/bert_pretrain_maa_common_MHC.json
```
The pre-trained model will be saved in `../Result/checkpoints/MHCBert_Pretrian` and `../Result/checkpoints/EpitopeBert_Pretrain` directory. 
### 2. Utilization
We recommend the use of EMBI on a cancer peptide dataset after fine-tuning. Alternatively, EMBI can predict exogenous peptides directly. The fine-tuning steps are detailed below:
#### 1). Fine-tune EMBI
 Prior to fine-tuning, ensure to replace some related paths in the `./config/EMBI_multimodality_training.json` and `./config/EMBI_multimodality_predict.json` files. In detail, please replace the "epitope_tokenizer_dir" and "EpitopeBert_dir" using ../Result/checkpoints/EpitopeBert_Pretrain/XXXX_XXXXXX; replace the "MHC_tokenizer_dir" and "MHCBert_dir" using ../Result/checkpoints/MHCBert_Pretrian/XXXX_XXXXXX in these two files. Prepare your dataset, ensuring it includes four columns: Epitope, Binder, Allele, HLA_pseudo_seq. Place the dataset files in the `./data/raw_data/` directory. Then, in the `./config/EMBI_multimodality_training.json` file, change the values of the `train_dir` (fine-tune train data) and `test_dir` (fine-tune test data) variables to match your dataset file names. If you have no dataset, demo data were prepared in the `./data/raw_data/` directory named `fine_tune_data_train.csv` and `fine_tune_data_test.csv` or you can skip fine-tuning and proceed to the Peptide immunogenicity prediction section. The fine-tuning command is:
 ```bash
 python EMBI_BA_AP_Immu_multimodality_main.py --config ./config/EMBI_multimodality_training.json
 ```
 Following fine-tuning, the model will be saved in the `../Result/checkpoints/EMBert-BA-AP-Immu-Training/` directory.
 #### 2). Peptide immunogenicity prediction
 Before initiating peptide prediction, download fine-tuned models for binding affinity and antigen presentation prediction from [google drive](https://drive.google.com/drive/folders/1PcfRcw0nIeUsDAg-f0AVxAgBFgqKpJ3i?usp=sharing) and place them in the current path. EMBI trained models are in `./EMBI_BA_model` and `./EMBI_AP_semi_model` directory. Please ensure that you download the entire directory. Then update the `resume` variable with the path to the EMBI model checkpoint in the `EMBI_BA_AP_Immu_multimodality_predict.py` file. The path should be ../Result/checkpoints/EMBert-BA-AP-Immu-Training/xxx.pth. The command is: 
 ```bash
 python EMBI_BA_AP_Immu_multimodality_predict.py --config ./config/EMBI_multimodality_predict.json --pf peptide_prediction_demo.csv
 ```
 If you wish to skip the pre-training process and proceed directly to utilizing the pre-trained models, download pre-trained models and fine-tuned models from [google drive](https://drive.google.com/drive/folders/1PcfRcw0nIeUsDAg-f0AVxAgBFgqKpJ3i?usp=sharing). The pre-trained models are in the `./EpitopeBert` and `./MHCBert` directories. The fine-tuned models are in `./EMBI_BA_model`, `./EMBI_AP_semi_model` and `./EMBI_multimodality_model` directories. Please replace the "epitope_tokenizer_dir" and "EpitopeBert_dir" using ./EpitopeBert; replace the "MHC_tokenizer_dir" and "MHCBert_dir" using ./MHCBert; update the `resume`  variable in the `EMBI_BA_AP_Immu_multimodality_predict.py` files with the following path `./EMBI_multimodality_model/model_best.pth`. The predict command is:
 ```bash
 python EMBI_BA_AP_Immu_multimodality_predict.py --config ./config/EMBI_multimodality_predict.json --pf peptide_prediction_demo.csv
 ```
 `peptide_prediction_demo.csv` should be placed in the `./data/raw_data` directory.
 ### 3. Expected output
 Subsequent to pre-training and fine-tuning, EMBI generates a CSV file named predict.csv. This file includes six columns: Epitope, MHC (pseudo sequence), HLA, binding_affinity_probability (ba_p), antigen_presentation_probability (ap_p), and immunogenicity_probability (Immu_pred). This file will be in the `./Result/checkpoints/EMBert-BA-AP-Immu-Predict/XXX_XXX/predict.csv`.
## Model availability
The pre-trianed model EpitopeBert and MHCBert and EMBI trained models on binding, antigen presentation and immunogenicity prediction task are available on [Zenodo](https://zenodo.org/records/12561108?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE4ODE0ZGM2LWQzMWMtNGFiYS04MDBkLTJhODc0NzFlZWE3MCIsImRhdGEiOnt9LCJyYW5kb20iOiI3MmJjYTkzMDJjYWU3MTZiMjQzZmE1OTIyODQzZGVmNCJ9.HCLhQvucZAbO1_ifHk89GlTJ6YT52w42Bw1w8dkKo8iEZI5GqPML1SUiQGx9PAhuEOiWquuPf0FN1LSQ_pcXCQ).
## Data availability
The training and test data utilized in this project are available in [datasets via Zenodo](https://zenodo.org/records/12801495?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjRlZmVlYTljLTU3MTAtNGU5YS1iYzYxLTNkNjQyZTY4MzM4OSIsImRhdGEiOnt9LCJyYW5kb20iOiJjZTFlNjVjYzAxODY4NjA0MzgxNzYyYWQzNzdmYjQ5NCJ9.wlS6wLdZday_l2gmAvvxMZIu0jwm_HRfVQa6gqc7DS-dueb898KaGywdxW91NJK9GSfKLT8XDxVr0xH9RRXp-g). Then you can directly run the model following above instructions.
## Contact
For further queries, you can reach out to us via email:
- [Yixin Guo](mailto:yixinguo.19@intl.zju.edu.cn)
- [Linnan Guo](mailto:guolinnan@zju.edu.cn)
- [Bing He](mailto:hebinghb@gmail.com)
