# Pulmonary Embolism (PE) Detection and Identification
Xiaotian Ma, Emma C. Ferguson, Xiaoqian Jiang, Sean I. Savitz & Shayan Shams. **A multitask deep learning approach for pulmonary embolism detection and identification**. *Scientific Reports*, 2022.
[Paper link](https://www.nature.com/articles/s41598-022-16976-9)

### Abstract
Pulmonary embolism (PE) is a blood clot traveling to the lungs and is associated with substantial morbidity and mortality. Therefore, rapid diagnoses and treatments are essential. Chest computed tomographic pulmonary angiogram (CTPA) is the gold standard for PE diagnoses. Deep learning can enhance the radiologists’workflow by identifying PE using CTPA, which helps to prioritize important cases and hasten the diagnoses for at-risk patients. In this study, we propose a two-phase multitask learning method that can recognize the presence of PE and its properties such as the position, whether acute or chronic, and the corresponding right-to-left ventricle diameter (RV/LV) ratio, thereby reducing false-negative diagnoses. Trained on the RSNA-STR Pulmonary Embolism CT Dataset, our model demonstrates promising PE detection performances on the hold-out test set with the window-level AUROC achieving 0.93 and the sensitivity being 0.86 with a specificity of 0.85, which is competitive with the radiologists’sensitivities ranging from 0.67 to 0.87 with specificities of 0.89–0.99. In addition, our model provides interpretability through attention weight heatmaps and gradient-weighted class activation mapping (Grad-CAM). Our proposed deep learning model could predict PE existence and other properties of existing cases, which could be applied to practical assistance for PE diagnosis.

### Dataset
First download the data into `data` folder. The dataset is publicly available on Kaggle: https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/data

### Run
(`torchsummary`: modified from https://github.com/sksq96/pytorch-summary/tree/master/torchsummary; `lungmask`: modified from https://github.com/JoHof/lungmask)

1. Run `sh run_bash/run_r3d_18_chunk10_batch16_epoch20.sh` to train the first stage.
2. Run `sh run_features_r3d_18_chunk10_batch16.sh` to extract features from the first stage.
3. Run `sh run_sequence_tcn_or_gru_r3d_18.sh` to train the second stage.
4. Adapt `load_feature_extractor` and `feature_dir` in `test_save_features.ipynb`, and run the notebook to extract first-stage features of the test set.
5. Adapt `load_model` and `feature_dir` in `test_step1.ipynb`, and run the notebook to get first-stage test results.
6. Adapt `load_sequential_model` and `feature_dir` in `test_save_features.ipynb`, and run the notebook to get second-stage test results.
