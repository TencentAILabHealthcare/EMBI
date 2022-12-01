## Project: BERT model for epitope immunogenicity prediction
This project is developed and maintained by @yixinguo and @owenbhe, supervised by @jianhuayao.

## Code structure
* `./Benchmark/` includes the result of benchmark
    * `./CD8_benchamrk_pred` netMHCpan prediction result, processing steps can be found in `./benchmark.ipynb`.
    * `./Figures` contains benchmark result.
    * `./IEDB_ligand_benchmark_result` contains benchmark reuslt using IEDB test data.

* `./Data` includes some independent data from database, recent paper ...
    * `./ARTEMIS` MASS-spec data from Finton, Kathryn AK, et al. "ARTEMIS: A Novel Mass-Spec Platform for HLA-Restricted Self and Disease-Associated Peptide Discovery." Frontiers in Immunology 12 (2021): 658372.
    * `./COVID-19` covid-19 eptioep data from IEDB
    * `./data` netmhcpan training data
    * `./dbPepNeo2.0` human tumor neoantigen peptides, from Lu, Manman, et al. "dbPepNeo2. 0: A Database for Human Tumor Neoantigen Peptides From Mass Spectrometry and TCR Recognition." Frontiers in immunology (2022): 1583.
    * `./DeepImmuno` DeepImmuno project, from https://github.com/frankligy/DeepImmuno
    * `./Figure` contains model performance result, processing steps can be found in `../process_data.ipynb`
    * `./FromZhao` model prediction result using data from louisyuzhao
    * `./IEDB` data downloaded from IEDB
    * `./INeo_epp`  data from Wang, Guangzhi, et al. "INeo-Epp: a novel T-cell HLA class-I immunogenicity or neoantigenic epitope prediction method based on sequence-related amino acid features." BioMed research international 2020 (2020).
    * `./MHCflurry` data from O’Donnell, Timothy J., Alex Rubinsteyn, and Uri Laserson. "MHCflurry 2.0: improved pan-allele prediction of MHC class I-presented peptides by incorporating antigen processing." Cell systems 11.1 (2020): 42-48.
    * `./NetMHCpan_train` data from netmhcpan
    * `./TESLA` data from Wells, Daniel K., et al. "Key parameters of tumor epitope immunogenicity revealed through a consortium approach improve neoantigen prediction." Cell 183.3 (2020): 818-834.
    * `./Zhao_2018_PLoS_computational_biology` data were used to benchmark binding affinity predictor, from Zhao, Weilong, and Xinwei Sher. "Systematically benchmarking peptide-MHC binding predictors: From synthetic to naturally processed epitopes." PLoS computational biology 14.11 (2018): e1006457.
    * `./process_data.ipynb` includes scripts processing data and data visualization
    * `./produce_figure.ipynb` includes scripts generating figures in paper

* `./Result/` includes the result of each experiment.
    * `./checkpoints/` includes the checkpoint and configuration of each experiment.
    * `./datasplit/` includes the configuration, log file and results for each experiment.

* `./TcellEpitope`
    * `./bert_data_prepare` contains the tokenizer building code.
    * `./config` contains the configuration files for each experiment.
    * `./data` includes the data files from IEDB, recent studies ...
        * `./figures` contains predict probability distribution for each experiment.
        * `./processed_data` includes the processed data, the processing steps can be found in `./Data/process_data.ipynb`. These dataset were generated for model embeddings visualization
        * `./raw_data` includes the data for model, the processing steps can be found in `./Data/process_data.ipynb`.
            * `./All_epitope_HLA_pseduo_seq` contains all eptiope and HLA pairs without PRIME epitopes, these data are used to generate pseduo label data.
            * `./BA_AP_data`  these data are used to generate pseduo label data.
            * `./Cancer_antigenic` these data are downloaded from https://caped.icp.ucl.ac.be/Peptide/list.
            * `./IEDB_MS_data` these data are downloaded from IEDB(mass spectrometry), for AP predictor benchmark.
            * `./MTL_data` these data are used for multi task learning model, pseudo label data.
        * `xxxx_dataset.py` these files are used for PyTorch DataLoader builiding for each experiment.             
    * `./logger/` contains the codes for logging, no need to change if not necessary.
    * `./model/` contains the codes for the architecture of each model.
    * `./trainer/` contains the trainer design for each experiment.
    * `xxxxx_main.py` these files are used to run each experiments.



## How to run
Use the self-supervised learning on the MHC pseudo sequence with common tokenizer (unique amino acid as one token) as an example, to run the code, please use the following command
```bash
python bert_pretrain_maa_main.py --config config/bert_pretrain_maa_common_MHC.json
```

## The dependencies of multi-tasks
1. Masked Amino Acid task

The masked amino acid task is self-supervised learning task of EpitopeBert and MHCBert, which needs to be trained first.
* Using the following command to get the pre-trained model of epitope sequences, and the pre-trained model is saved in its corresponding folder under `../Result/checkpoints/` folder (such as `/aaa/louisyuzhao/project2/immuneDataSet/jasonjnyang/Epitope-receptor-generative/Result/checkpoints/BERT-Epitope-Pretrain-Common-MAA/0816_171909`).
```bash
python bert_pretrain_maa_main.py --config config/bert_pretrain_maa_common_MHC.json
```
* Using the following command to get the pre-trained model of MHC pseudo sequences, and the pre-trained model is saved in its corresponding folder under `../Result/checkpoints/` folder (such as `../Result/checkpoints/BERT-MHC-Pretrain-Common-MAA/0824_163909`).
```bash
python bert_pretrain_maa_main.py --config config/common/bert_pretrain_maa_common_beta.json
```
2. Binding Affinity Prediction task

After the MAA training, we utilize the pre-trained models to initialize EpitopeBert and MHCBert and fine-tune on the binding affinity prediction task. Using the following command to get the fine-tuned model. Note that you need to change the settings `"EpitopeBert_dir"` and `"ReceptorBert_dir"` in the config file (such as `config/common/bert_finetuning_er_main.json`) using the path of the pre-trained models of EpitopeBert and MHCBert. The fine-tuned model is saved in its corresponding folder under `../Result/checkpoints/`, such as `../Result/checkpoints/Epitope-MHC-Debug/0909_125043`.
```bash
python EMBert_ba_main.py --config config/EMBert_ba.json
```

3. Antigen Presentation Prediction task

After the MAA training, we utilize the pre-trained models to initialize EpitopeBert and MHCBert and fine-tune on the antigen presentation prediction task. Using the following command to get the fine-tuned model. Note that you need to change the settings `"EpitopeBert_dir"` and `"ReceptorBert_dir"` in the config file (such as `config/common/bert_finetuning_er_main.json`) using the path of the pre-trained models of EpitopeBert and MHCBert. The fine-tuned model is saved in its corresponding folder under `../Result/checkpoints/`, such as `../Result/checkpoints/EMBert-Pre-Debug/0828_214943`.
```bash
python EMBert_ap_main.py --config config/EMBert_ap.json
```

4. Immunogenicity Prediction task
After the BA and AP predictor training, we utilize the pre-trained models to initialize EpitopeBert and MHCBert and fine-tune on the immunogenicity prediction task with BA and AP embedding. Using the following command to get the fine-tuned model. Note that you need to change the settings `"EpitopeBert_dir"` and `"ReceptorBert_dir"` in the config file (such as `config/common/bert_finetuning_er_main.json`) using the path of the pre-trained models of EpitopeBert and MHCBert. The fine-tuned model is saved in its corresponding folder under `../Result/checkpoints/`, such as `../Result/checkpoints/EMBert-BA-AP-Immu/Multimodality/1018_144215`.
```bash
python EMBert_BA_AP_Immu_multimodality_main.py --config config/ba_ap_immu_multimodality.json
```

## Environment
* All the codes and experiments are developed under `python==3.10.4`.
* Full dependencies are saved in `requirement.txt`. Using `pip` to install all these packages is recommended.
* GPU support: Taiji platform.