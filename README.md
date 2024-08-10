# Vietnamese hate speech detection
## 1. Introduce
The goal of this project is to classify sentiments in Vietnamese comments on social media to detect and prevent hate speech and offensive language. By doing so, the project contributes to creating a healthier and safer social media environment.

The project is implemented by training a model on the ViHSD dataset and using the TextCNN model to classify comments into three categories:

- **Clean**: Comments that do not contain any harassing behavior.

- **Offensive**: Comments that contain harassing content, including vulgar language, but are not directed at specific individuals.

- **Hate**: Comments that are hateful and offensive, directly targeting individuals or groups based on personal characteristics, religion, or nationality.

## 2. Dataset
The dataset used in this project is the ViHSD - Vietnamese Hate Speech Detection dataset.

ViHSD is a Vietnamese dataset collected from comments on popular social media platforms such as Facebook and YouTube. This dataset has been manually annotated to support research on the automatic detection of hate speech on social media platforms.

ViHSD contains 33,400 labeled comments, with a total vocabulary of 21,239 words. The comments in the dataset are divided into three categories based on the following labels: HATE (2), OFFENSIVE (1), and CLEAN (0).

```
@InProceedings{10.1007/978-3-030-79457-6_35,
author="Luu, Son T.
and Nguyen, Kiet Van
and Nguyen, Ngan Luu-Thuy",
editor="Fujita, Hamido
and Selamat, Ali
and Lin, Jerry Chun-Wei
and Ali, Moonis",
title="A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts",
booktitle="Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="415--426",
abstract="In recent years, Vietnam witnesses the mass development of social network users on different social platforms such as Facebook, Youtube, Instagram, and Tiktok. On social media, hate speech has become a critical problem for social network users. To solve this problem, we introduce the ViHSD - a human-annotated dataset for automatically detecting hate speech on the social network. This dataset contains over 30,000 comments, each comment in the dataset has one of three labels: CLEAN, OFFENSIVE, or HATE. Besides, we introduce the data creation process for annotating and evaluating the quality of the dataset. Finally, we evaluate the dataset by deep learning and transformer models.",
isbn="978-3-030-79457-6"
}
```
## 3. Method
### 3.1. Data preprocessing

The data preprocessing process was carried out through four main steps:

- **Text segmentation**: The text was segmented into individual words using the PyVi tool.

- **Stopword removal**: Words that do not carry significant meaning in a specific context, such as stopwords, were removed based on a predefined list in the **vietnamese-stopwords-dash.txt** file.

- **Emoji removal**: Emojis in the text were removed to ensure the data is suitable for subsequent processing steps.

- **Lowercasing**: All words in the text were converted to lowercase to ensure consistency.

### 3.2. Feature extraction
The feature extraction for the model was performed using the Word Embedding method. Specifically:

- **Word Embedding usage**: The word embedding used in this project is **cc.vi.300.vec**, a pre-trained .vec file generated using the fastText method.

- **Creating a tokenizer and embedding matrix**: A tokenizer object was created to build a dictionary of words appearing in the training set. Irrelevant words, special characters, and unnecessary punctuation were removed. The dictionary was limited to a maximum size of 10,000 words. For each word in the dictionary, a corresponding word embedding vector was placed into the embedding matrix with dimensions **(num_words, embedding_dim)**, where **num_words** is the number of words in the dictionary (plus one special token) and **embedding_dim** is 300. If a word is not found in the vocabulary of the .vec file, its embedding vector will be a zero vector.

  ### 3.3. Model
  ![](https://github.com/tnhi1821/Vietnamese-hate-speech-detection/blob/main/Image%20source/TextCNN.jpg)

The model used in this project is Text-CNN. The configuration of the model is as follows:
- Number of epochs: 40
- Batch size: 256
- Sequence length: 100
- Dropout rate: 0.5

The Text-CNN model includes a 1D Convolution layer with 32 filters and kernel sizes of 2, 3, and 5. The Adam optimizer was used to train the model.

## 4. Result
![](https://github.com/tnhi1821/Vietnamese-hate-speech-detection/blob/main/Image%20source/clean.jpg)
![](https://github.com/tnhi1821/Vietnamese-hate-speech-detection/blob/main/Image%20source/offensive.jpg)
![](https://github.com/tnhi1821/Vietnamese-hate-speech-detection/blob/main/Image%20source/hate.jpg)


To better understand how the website operates, please watch this [video](https://github.com/tnhi1821/Vietnamese-hate-speech-detection/blob/main/Image%20source/Results.mp4)

The deep neural network model Text-CNN achieved an accuracy of 86.56% and an F1-macro score of 62.25% on the dataset.

However, the model faces several challenges:
- **Class Imbalance**: The model tends to predict the CLEAN label due to the significant imbalance in the number of samples between the classes.
- **Limitations of Word Embeddings**: Using word embeddings from fastText allows the model to learn relationships between words in the dictionary but does not capture full context, leading to misclassification of homophones.
- **Informal Language**: The dataset contains many comments written in informal language, with abbreviations, slang, and metaphorical expressions, which makes accurate classification challenging for the model.

To improve the model's performance, I propose using contextual word embedding models such as BERT or PhoBERT to replace fastText. These models can capture word context more flexibly and accurately, helping to enhance classification accuracy, particularly for homophones and complex cases in informal language.
