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
### 2. Utilization
We recommend the use of EMBI on a cancer peptide dataset after fine-tuning. Alternatively, EMBI can predict exogenous peptides directly. The fine-tuning steps are detailed below:
#### 1). Fine-tune EMBI
 Prior to fine-tuning, ensure to replace some related paths in the `./config/EMBI_multimodality_training.json` and `EMBI_multimodality_predict.json` files. Download pre-trained models from [google drive](https://drive.google.com/drive/folders/1PcfRcw0nIeUsDAg-f0AVxAgBFgqKpJ3i?usp=sharing) and place them in the current path. The pre-trained models are in ./MHCBert and ./EpitopeBert directory.The fine-tuning command is:
 ```bash
 python EMBI_BA_AP_Immu_multimodality_main.py --config ./config/EMBI_multimodality_training.json
 ```
 Following fine-tuning, the model will be saved in the `../Result/checkpoints/XXX_XXXX` directory.
 #### 2). Peptide immunogenicity prediction
 Before running peptide prediction, download fine-tuned models from [google drive](https://drive.google.com/drive/folders/1PcfRcw0nIeUsDAg-f0AVxAgBFgqKpJ3i?usp=sharing) and place them in the current path. EMBI trained models are in `./EMBI_BA_model`, `./EMBI_AP_semi_model` and `./EMBI_multimodality_model` directory. The command is: 
 ```bash
 python EMBI_BA_AP_Immu_multimodality_predict.py --config ./config/EMBI_multimodality_predict.json --pf peptide_prediction_demo.csv
 ```
 ### 3. Expected output
 Subsequent to pre-training and fine-tuning, EMBI generates a CSV file named predict.csv. This file includes five columns: peptide, MHC, binding_affinity_probability (ba_p), antigen_presentation_probability (ap_p), and immunogenicity_probability (Immu_pred).
## Model availability
The pre-trianed model EpitopeBert and MHCBert and EMBI trained models on binding, antigen presentation and immunogenicity prediction task are available on [google drive](https://drive.google.com/drive/folders/1PcfRcw0nIeUsDAg-f0AVxAgBFgqKpJ3i?usp=sharing). 
## Data availability
The training and test data utilized in this project are available in the `./data/raw_data` directory. The file `Epitope_BA_data.csv` is used for binding affinity task training, `MHCflurry_BA_training_data.csv` for antigen presentation task training and `Epitope_info_from_PRIME.csv` for immunogenicity prediction task training. Additional files are employed to evaluate our model in comparison with other models. To access the demo data, please download [the demo data via google drive](https://drive.google.com/file/d/1XoEf914xjskOHRw94afrnXSU3vJj0G47/view?usp=sharing). Then you can directly run the model following above instructions.
## Contact
For further queries, you can reach out to us via email:
- [Yixin Guo](mailto:yixinguo.19@intl.zju.edu.cn)
- [Bing He](mailto:hebinghb@gmail.com)
