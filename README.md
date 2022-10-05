# TAD
Target Activity Detection based on speaker verification models trained/tuned on kids' speech

Wayne Ward and Rosy Southwell 2022

# Speaker recognition models
These experiments are based on Speechbrain's speaker verification recipes trained on VoxCeleb.
Here we include kids speech corpora and classroom background noise in the training fata

The model architecture we use is ECAPA-TDNN

# Speaker verification using ECAPA-TDNN embeddings
Run the following command to train speaker embeddings using [ECAPA-TDNN](https://arxiv.org/abs/2005.07143):

`python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml`

After training the speaker embeddings, it is possible to perform speaker verification using cosine similarity.  You can run it with the following command:

`python speaker_verification_cosine.py hparams/verification_ecapa.yaml`

# **Acknowledgements**
This project makes use of the Speechbrain library
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```


