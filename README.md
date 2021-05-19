# Acoustic-to-articulatory inversion of dysarthric speech by using cross-corpus acoustic-articulatory data (ICASSP 2021)
This repository houses the official Keras implementation of our paper accepted to 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). If you find the [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9413625) useful or reproduce any of the code's modules for your research, please consider citing our paper. 

```
@inproceedings{maharana2021acoustic,
  title={Acoustic-to-Articulatory Inversion for Dysarthric Speech by Using Cross-Corpus Acoustic-Articulatory Data},
  author={Maharana, Sarthak Kumar and Illa, Aravind and Mannem, Renuka and Belur, Yamini and Shetty, Preetie and Kumar, Veeramani Preethish and Vengalil, Seena and   Polavarapu, Kiran and Atchayaram, Nalini and Ghosh, Prasanta Kumar},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6458--6462},
  year={2021}
}
```
All the modules have been well-documented and are self-explanatory. However, for any clarifications regarding the code published here, please feel free to reach out to me at <sarthakmaharana9811@gmail.com> with the subject line as "(GitHub) ALS-AAI", for consideration. 
To get access to the datasets we have used and any other discussions regarding potential research, please contact Dr. Prasanta Kumar Ghosh at <prasantg@iisc.ac.in>

## Abstract 
In this work, we focus on estimating articulatory movements from acoustic features, known as acoustic-to-articulatory inversion (AAI), for dysarthric patients with amyotrophic lateral sclerosis (ALS). Unlike healthy subjects, there are two potential challenges involved in AAI on dysarthric speech. Due to speech impairment, the pronunciation of dysarthric patients is unclear and inaccurate, which could impact the AAI performance. In addition, acoustic-articulatory data from dysarthric patients is limited due to the difficulty in the recording. These challenges motivate us to utilize cross-corpus acoustic-articulatory data. In this study, we propose an AAI model by conditioning speaker information using x-vectors at the input, and multi-target articulatory trajectory outputs for each corpus separately. Results reveal that the proposed AAI model shows relative improvements of the Pearson correlation coefficient (CC) by ∼13.16% and ∼16.45% over a randomly initialized baseline AAI model trained with only dysarthric corpus in seen and unseen conditions, respectively. In the seen conditions, the proposed AAI model outperforms the three baseline AAI models, that utilize the cross-corpus, by ∼3.49%, ∼6.46%, and ∼4.03% in terms of CC.

## Installation 
    $ git clone https://github.com/sarthaxxxxx/AAI-ALS.git
    $ cd AAI-ALS/
    $ pip install requirements.txt
  
## Training 
  ### Configuring cfg.yaml
  To train the xSC and xMC AAI models, set x_vectors: True, and change the name of the model to xSC/xMC. Set subject conditions to "seen" or "unseen" to reproduce   the results as in the paper. To train any other AAI models, set x_vectors: False and change the model's name (RI/MC). 
  
  ### To begin training (single GPU)
  ```
  python3 train.py --config (path to config file in your system) --gpu (gpu_id)
  ```
  Best models and the respective weights to be saved at ./ckpt/
  
  ### Regarding the GBM and GBM-FT AAI models
  Using ./utils/Get_SPIRE_EMA_data_Full.py, extract training and validation data from the cross-corpus. Train the RI AAI model on this data (experiment by varying the BLSTM units). The resulting best model (256 units) is the GBM AAI. Fine-tune the best weights of the GBM on the dysarthric data and retrain to obtain the GBM-FT AAI model. 
  
## Testing
   To print out speech task based results for healthy controls and patients:
   ```
   python3 test.py --config (path to config file in your system)
   ```
## Presentation and Poster
   [Presentation](https://drive.google.com/file/d/1BkNZ1QMl1UM9ivvUNMVxLee7U2RC-Kzj/view?usp=sharing) <br>
   [Poster](https://drive.google.com/file/d/188MDKXdYPgAHxiyvGJt11-i0OjF1-8Pq/view?usp=sharing)
  
## License
MIT License

## Acknowledgement
We thank all the subjects who participated in the EMA recordings. We also thank the Department of Science and Technology, Government of India, for their support.
