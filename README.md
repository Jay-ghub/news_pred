# News classification and Keywords Extraction

## Project Goal 
The goal is to use machine learning classifiers to predict the categories for English news articles. Also extract the most important terms of an article according to their TF-IDF scores. At least two different models like Logistic Regression and Naive Bayes were trained and compared.

## Dataset
The dataset comes from this [Kaggle Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset/data), which contains news articles published between 2012 and 2022 from Huffpost. The dataset is in JSON format and the most useful columns for the models are "headline" and "short_description". To allow the models to get more information, these two columns were combined into "headline_description" which became our feature. In the original dataset, there were many categories but few (main categories) were selected. In the end, 6 categories were selected and sampled to each 2100 articles: 
1. POLITICS
2. BUSINESS 
3. SPORTS
4. TECH 
5. SCIENCE 
6. ENTERTAINMENT

## Data Preparation

Load the JSON file.
- inspect the dataset to have an overview of it.
- check the number of categories (initially there were 42)
- select a few of them --> some categories overlap
- combine headline and short_description
- balance the categories --> Why ? : because it prevents large categories from dominating training/evaluation (class imbalance)
- Then train / test split

## Methods

The "NewsClassifier" class allows to reproduce the train and evaluation steps for each model without having to duplicate the same steps. There is also a method "extract_keywords" for the keyword-extraction steps.
1. "train()" method: 
      - fits the TF-IDF vectorizer on the training texts, transform them into TF-IDF scores and train the classifier using those scores. 
2. "predict()" method: 
      - Transforms the test set using the already fitted vectorizer and uses the trained model to make the predictions.
3. "evaluate()" method: 
      - calculates evaluation results by outputing the accuracy score, f1 score, classification report and the confusion matrix.
4. "extract_keywords()" method: 
      - returns the top 5 terms with the highest TF-IDF scores for a particular instance of the test set. 

## Models

1. Logistic Regression: 
      - A discriminative linear classifier that categorize the article using the feature weights learned during training.
2. Multinomial Naive Bayes: 
      - probabilistic generative classifier, that used the Bayes rule to select the most probable category.

## Evaluation

For Evaluation, four things were used:
1. Accuracy:
      - Overall proportion of correctly classified articles.
      - Because the dataset was balanced, accuracy was useful here.

2. Macro F1:
      - Computes F1 score for each category or class and then averages them equally. 
      - Useful for checking if the model performs well accross the 6 classes.

3. Classification report:
      - Provides for each class:
          - precision
          - recall
          - F1
          - support

4. Confusion Matrix:

      - Show which classes are mistaken for each class.


## Results:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.78 | 0.78 |
| Multinomial Naive Bayes | 0.79 | 0.80 |

    The models have approximately the same results. Based on the F1 scores, SPORT is the strongest class for both models: 0.85 with Logistic Regression and 0.88 with MultinomialNB. POLITICS and SCIENCE did also perform well. BUSINESS is the weakest class for both models with 0.69 for LR and 0.70 for MultinomialNB, while TECH is around 0.75-0.76. Future experiments could be to increase the size of the training dataset, experiments with different n_grams range, or hyperparameter-tuning.


## Code reproduction steps

1. Clone the project
2. Download the [Kaggle Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset/data) and place inside the "data" folder
3. create a virtual environment with:
   - python3 -m venv /.venv
   - Then activate with source .venv/bin/activate
5. Install dependencies
    - pip install -r requirements.txt
6. Start Jupyter Notebook
7. open main.ipynb
8. Run all cells

