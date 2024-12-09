# ICDM 2024 (demo track)
This repository provides the code for our paper at ICDM 2024 (demo track) - "Video-based learning of sign languages: one

pre-train to fit them all".  

The paper introduces an approach for training a Foundation model capable of obtaining high-quality 
embeddings for sign languages.  Experiments have shown that the model can be quickly and
efficiently fine-tuned for other sign languages and shows significantly better metrics compared 
to a model pre-trained on a general-purpose dataset (Kinetics-400).
The code for training and fine-tuning models is located in [slt_mae](slt_mae). The code for obtaining the degree 
of signs similarity is in the [mae_smoothing](mae_smoothing) folder.  
To make sign language processing more accessible, we are publishing our foundation model so that 
others can fine-tune it for other sign languages. Also, we are publishing fine-tuned
models for American, Turkish, Greek and Russian sign languages.


| Model            | Weights                           |
|------------------|-----------------------------------|
| Foundation model | [Download](https://clck.ru/3F7jQ9) |
| WLASL fine-tuned | [Download](https://clck.ru/3F7jS2) |
| AUTSL fine-tuned | [Download](https://clck.ru/3F7jYS) |
| GSL fine-tuned   | [Download](https://clck.ru/3F7jZQ) |
| SLOVO fine-tuned | [Download](https://clck.ru/3F7jaA) |
