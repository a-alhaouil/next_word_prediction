# Next Word Prediction

This repository contains machine learning models and code for predicting the next word in a sequence based on a variety of algorithms and methods, such as TF-IDF, Cosine Similarity, AdaBoost, and more. The project demonstrates how to use different techniques for text prediction tasks.

## Features

- **Next Word Prediction using:**
-  **TF-IDF Multinominal Naive Bayes model / embedding with universal model encoder by Tensor Flow with  LSTM model**
- **Training and evaluating multiple models**
- **Interactive Web Application using Flask**
- **Support for different types of word embeddings and vectorizers**

## Requirements

Before running the code, ensure you have the following Python libraries installed:

- `Flask`
- `pandas`
- `numpy`
- `scikit-learn`
- `keras`
- `tensorflow`
- `nltk`
- `gensim`

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```
## additional step for universal model encoder from Tensor Flow :
- in **app.py** uncomment this line
  ```bash
   embed = (hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4"))
  ```
- make sure to make a rep in your repository named **universal_model_encoder_tf**
- navigate to here you find this rep and copy it in **universal_model_encoder_tf**
```bash
C:\\Users\\name\\AppData\\Local\\tfhub_modules
```
```bash
063d866c06683311b44b4992fd46fsfdsfdsf/
│
├── saved_model.pb                  
├── variables/                        
│   ├── variables.data-00000-of-00001             
│   └──  variables.index                        
```
- they comment the line to avoide loading the model each time you run the app
- 
### Web interface by flask framework

![Capture d'écran 2024-11-14 234309](https://github.com/user-attachments/assets/719c358f-5c0e-4264-b71b-7f730827489d)
![Capture d'écran 2024-11-14 234328](https://github.com/user-attachments/assets/95c01d25-34be-4b67-bb32-c89eb26a04ba)

