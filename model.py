import pandas as pd
import numpy as np
from constants import *
import pickle

def load_datamodels(filename):
    print(ML_MODELS_DIRECTORY + filename)
    return pickle.load(open(ML_MODELS_DIRECTORY + filename, 'rb'))

# Now combine the classification model positive sentiment and from the top 20 from the recommendation system pick the top 5 who have the positive sentiment of the reviews
def product_recommendations(user, top_n):
    user_ratings_data = load_datamodels(RECOMMENDATION_MODEL)
    if (user in user_ratings_data.index):
        tfidf_vectorizer = load_datamodels(WORD_VECTORIZER)
        processed_data = load_datamodels(PROCESSED_DATA)
        sentiment_classification_model = load_datamodels(SENTIMENT_CLASSIFICATION_MODEL)
        
        # Get the top 20 recommended products using the trained ML model for the specified user
        top_20_recommendations = list(user_ratings_data.loc[user].sort_values(ascending=False)[0:20].index)
        
        # Get the processed data for the top 20 recommended products
        df_classify_recommend = processed_data[processed_data.id.isin(top_20_recommendations)]
        
        # create the tfidf vectorizer for the records
        X =  tfidf_vectorizer.transform(df_classify_recommend[FEATURE_REVIEW_TEXT_LEMMA].values.astype(str))
        
        # predict the review sentiment values
        df_classify_recommend[FEATURE_PREDICTED_SENTIMENT] = sentiment_classification_model.predict(X)
        df_classify_recommend = df_classify_recommend[[FEATURE_NAME, FEATURE_PREDICTED_SENTIMENT]]
        
        # group the data for each product
        df_classify_recommend_grouped = df_classify_recommend.groupby(FEATURE_NAME, as_index=False).count()
        
        # get the positive review count, total count and the overall percentage of positive reviews
        df_classify_recommend_grouped[FEATURE_POS_REVIEW_COUNT] = df_classify_recommend_grouped.name.apply(lambda x: df_classify_recommend[(df_classify_recommend.name==x) & (df_classify_recommend.predicted_sentiment==1)][FEATURE_PREDICTED_SENTIMENT].count())
        df_classify_recommend_grouped[FEATURE_TOTAL_REVIEW_COUNT] = df_classify_recommend_grouped[FEATURE_PREDICTED_SENTIMENT]
        df_classify_recommend_grouped[FEATURE_POS_SENTIMENT_PERCENT] = np.round(df_classify_recommend_grouped[FEATURE_POS_REVIEW_COUNT]/df_classify_recommend_grouped[FEATURE_TOTAL_REVIEW_COUNT]*100, 2)
        
        # return the grouped values with the product based on percentage of positive reviews count
        return df_classify_recommend_grouped.sort_values(FEATURE_POS_SENTIMENT_PERCENT, ascending=False)[:top_n]
    else:
        # if the user is not present in the recommendation data flag user not found
        print(f"User {user} not found. Please try again.")
        return None

def convertJson(dataFrame: pd.DataFrame):
    return dataFrame.to_json()

def get_users():
    user_ratings_data = load_datamodels(RECOMMENDATION_MODEL)
    return list(user_ratings_data.index.values.tolist())
