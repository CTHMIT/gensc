# GenSC: Generative Semantic Communication Systems Using BART-Like Model

We propose a new semantic communication system that considers the token-level correlation between consecutive tokens during the semantic encoding. 
This bidirectional correlation helps correct or fill in a semantically similar token at the semantic decoder when a token is missing or corrupted during transmission.

[Paper](https://ieeexplore.ieee.org/document/10648817)

# Overview

We call such a semantic communication the generative semantic communication (GenSC).
The key difference between this work and previous works is the proposed generative semantic communication system can recover the sentence from the corrupted one due to channel impairment. 
To achieve this, the BART-like model is adopted for semantic encoding and decoding. The semantic decoder first obtains the relationship between the consecutive tokens in a sentence and utilizes bidirectional encoding to encode the sentence semantically. 
With the help of token-level correlation, the corrupted tokens can be recovered or replaced with the most probable token.
![image](https://github.com/user-attachments/assets/0ba45cac-0249-4b54-b0c1-ea18ac32e02b)


For further information please contact [Chun-Tse Hsu](https://github.com/CTHMIT)

# Training environmental needs

  - Anaconda
  - Python 3.9
  - git
    
## Clone file
First, Download the project:

```bash
https://github.com/CTHMIT/gensc.git
cd GenSC
```

## Authorize and execute

```bash
chmod +x run.sh
./run.sh
```

## Or manually install the environment and run the main program

### Install environment

```bash
conda env create -f environment.yml
```

### Run main program
```bash
python GenSC_finall_all.py
```

# Citation

Please cite this paper if you want to use it in your work

```bash
@ARTICLE{10648817,
  author={Chang, Min-Kuan and Hsu, Chun-Tse and Yang, Guu-Chang},
  journal={IEEE Communications Letters}, 
  title={GenSC: Generative Semantic Communication Systems Using BART-Like Model}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Semantics;Decoding;Encoding;Signal to noise ratio;Correlation;Communication systems;Transformers;Transformer;Semantic communication;Generative model},
  doi={10.1109/LCOMM.2024.3450309}}
```

# License

[MIT License](https://github.com/CTHMIT/gensc/blob/main/LICENSE)

# Acknowledgement

The structure of this codebase is borrowed from DeepSC
