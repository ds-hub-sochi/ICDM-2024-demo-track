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
| Foundation model | [Download](https://sochi-hub.obs.ru-moscow-1.hc.sbercloud.ru:443/eccv_2024/pretrain.pth?AccessKeyId=RDEYAQXJBQIQUG7P1N26&Expires=1733793819&x-obs-security-token=gQtydS1tb3Njb3ctMYbUA9--IN_fC4YwW77q8LrqNlcxf3AoxgtyAzNWFouuJHFXpZFaJ3sHtWzgg9W8BD2ZaFClF-DRer3N2kC9XX0YVG-IMWVmAa5_NAhinLZ-PjVQ8bZZQqIoVpmVV3jfFZVR2XaW_TmdBFX4R5jq5b4sMTdtYH93HIaSXBNaTd2TLV0ahhQ-IxomUxxE9ce6L5MRYsrj6eQ98H9ek8ShqDhbVfmKfqGPN2ekhffijIWHattfOfC9KToTcSV56uip9zZmY5iwhtbJmi2x_G7dk7TaHidj4TzWlgGBpy2zx1UYT7WHUh1NXFlFymjvtvO1nlxXtXUwX2zBzixTEfi52LmUZgQ2vbVjoxm9oijrbBj8TcdLntQitfcB0elJ0kvMyK6XJ6qUEIqYpimjb-ssCAon6K4VppIRVbLVQJvau5C53gccJ-_Jfzfnj7IIWMsdGxwAex8ZVS-OCSWhDQ71zbttNiRG0k8l2gXN5q_6legwYxqjS8dhLp_Fpc3HIyCgFw6XGWrzJV2ZTuFLwk6992gpWuqbmLOpFDaau1n-tFh5CNQe-UMGcGybhZEW02DgQr2K7ZqxtcN4H310Eo_-0gi7kj1H1pXq7kyy_8lZY_lvv4-NKu2cUS4DmuD_Gy6nplWX7xRCBWJi_-1OMe4zATjvsLSGTHSLGt0esYj0HK52nSAH_eQRiGvnjZeigap7jgu9ac4ap9AW8aVtoS4Jlv0dCrjZx2ZgVn6oskZLw0L_DoQsoroV9TTJX0Gci9xCXZZkH7sLgU5DCbybgvVUGshv5Sdm7A4XvXkwZ9IyoKAsAfJfY1yfBZo1yQi2lTGTi8PYvESvxAvLx6ZAyMxBlp704t5nXrZYmtcFbD73EAkDcQ%3D%3D&Signature=7jtyNeC00Ck/bbn2I%2Bk7UmUT7gQ%3D) |
| WLASL fine-tuned | [Download](https://sochi-hub.obs.ru-moscow-1.hc.sbercloud.ru:443/eccv_2024/wlasl.pt?AccessKeyId=9LR42G7KC74GDPKMZLC1&Expires=1733793846&x-obs-security-token=gQtydS1tb3Njb3ctMYYwf-c0VK05RN2BhIDHt1kMD-CezyaWftPegotU_cxOOKB_iqegWvtEzisKffl84QaswecAoEgi9O48CIs5xxhciwcZaw-Nc23bD_Qn_fzuI53rIuPkYK2k14gb_EYM0_UxOyVBJMqhLa3VpuOFZYDRCMrpQJLmooCyo4Op4NGAglpMHvz0P01OapQ2dTh-0_N4Eih0L57vNU-D4lU_8NmUVLORDzVWDYouNPY5o8ryOJmi9aDgKKAxOAcVoOailYtBDJfdQSf7XvUl6TzWP2aRuDg8WSPt7hUwNU9yTd4mfXcWknry2HFeRwZNFGa9vcMUxmRzhu2WsNoROuTIUQ0qu5EgQBBHQWskVFgGt0DnW1mqFf5jf986WoxkWJ5rnMH8xuSAt0wY6Qsu6O2BGSYGKEMgkTJPyz58a689czrIoW6oRs4DOtE5rAIt8Ow11mFz4NaGK13paZrm_WxbgMSfj_WrPHdw1vDVcB-cI5xubv3DCKFa2XYO0-Ue0iToRAbW8UDe5unWW9ZqPaEHwMu2HgImaDpnf9s3cBNYvIJLahMjMEDWjTOYR43YFdRMzxhmVNW993GqbD_GQN1skxFyG_hJATdNUDBX-PTsOk0Hx4Asvr1uenjSjdvO8Oh3e2OMfsqFoUuEnxrDDPqts5AtEYjOACT0JOnBO1aB9X0QzjYS3iEk2oH_BhYarePRUep9xi-c32z55VxKZvwiDMr_fwImzPG59ZEvIpRpzWKtQM92BkiAztbqLGPH6_rIw3YzFgwm0CCjpUyeROyQrw8G7uA3jJKkSaP5R5k8EdrWQnToNGvL_OpgPn_q49vGOF267IsJNlCsYLRjb6WdRvJG-DMnpR59y6qQbLzJ0mYxLQ%3D%3D&Signature=KIP4WhVlQHUhq7fmm/jDa2R0wrg%3D) |
| AUTSL fine-tuned | [Download](https://sc.link/CJE74) |
| GSL fine-tuned   | [Download](https://sc.link/rV4OF) |
| SLOVO fine-tuned | [Download](https://sc.link/BySDp) |
