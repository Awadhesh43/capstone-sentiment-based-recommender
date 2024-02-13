# Sentiment Based Product Recommendation

### Problem Statement

The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

### Solution
* GitHub link: https://github.com/Awadhesh43/capstone-sentiment-based-recommender

* Heroku (Application is Live): https://product-recommendation-nlp-app-29e391d3dfea.herokuapp.com
  
#### Google colab Execution
##### Steps to run the Sentiment-Based-Product-Recommendation-Model.ipynb Jupyter file for model files creation 
* Create *"Colab Notebooks"* folder in mydrive. 
* Within *"Colab Notebooks"* add *"data"* and *"model_files"* folder
* Add sample30.csv data file in the *"data"* folder 
* Run the file Sentiment-Based-Product-Recommendation-Model.ipynb

### Toolkit/Libraries

* Python 3.11.2
* scikit-learn 1.2.2
* xgboost 2.0.3
* numpy 1.23.5
* nltk 3.8.1
* pandas 1.5.3
* FastAPI 0.109.0
* Flask 3.0.1


### Solution Guide

* Dataset and Attribute description file are available in "data" folder.
* Data Cleaning, Visualization and Text Preprocessing (NLP) are applied on the dataset. TF-IDF Vectorizer is used to vectorize the textual data(review_title+review_text). It measures the relative importance of the word w.r.t other documents.
* Dataset is having Class Imbalance and SMOTE Oversampling technique is used before applying the model.
* Machine Learning Classification Models (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, XGBoost are applied on the vectorized data and the target column (user_sentiment). The objective of this ML model is to classify the sentiment to positive(1) or negative(0). Best Model is selected based on the various ML classification metrics (Accuracy, Precision, Recall, F1 Score, AUC). XGBoost is selected to be a better model based on the evaluation metrics.
* Recommendation model is created using User-Based and Item-Based collaborative filtering methods. RMSE evaluation metric is used for the evaluation and comparison.
* Sentiment-Based-Product-Recommendation.ipynb Jupyter notebook contains the code for Sentiment Classification NLP Model and Recommendation ML Model.
* Top 20 products are filtered using the better Recommendation ML Model (User-Based) and recommedations for the 5 best products based on the sentiments of the 20 recommended product reviews are made after combining with the Sentiment Classification model.(model.py)
* Machine Learning models are saved in the "model_files" directory. FastAPI and Flask APP (app.py) are used as interface to run the Machine Learning models.
* FastAPI jinja2 templates (templates\index.html) and Bootstrap Style CSS are used for setting up the User interface.
* End to End application is deployed in Heroku PaaS Platform.
