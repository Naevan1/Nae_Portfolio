# Nae_Portfolio
AI/Data Science Portfolio


# Project 1 : Customer Call Prediction
* Created a from-scratch GPT-1 inspired model which is pretrained on sequences of packages, from warehouse to client. It tries to generate the next most possible checkpoint a package will be in. Instead of a natural language vocabulary, we use checkpoint abbreviations. Features include time, time difference between checkpoints, station/facility/events etc.
* Fine-Tuned on top to predict whether a customer will call or not. Each sequence/data point has a label of whether and when a customer called.
* Implemented Uncertainty Estimation in the Transformer, in order to be able to reject data points which are OoD or too ambiguous.
* Creating a pipeline which clusters and calculates sentiment of the comments left by customers.


# Project 2 : Website Recommender System + Nudging to lead(account creation) (without individual user data)
*  Created a network with nodes as URL and links as 'paths' of users between the URL.
*  Used Graph Neural Networks and Node2Vec to get node embeddings. Link Prediction from GNN was used to fill high probable links which dont exist.
*  BERT similarity to automatically recommend similar articles with other articles within the website. Bert Embeddings were concatenated with Node2Vec embeddings from paths for further information.
*  Used Beam Search Algorithm with Markov Chain transition probabilties to recommend a URL/path to 'nudge' users towards lead conversion (account creation)

# Project 3 : Chatbot Agent-Customer Text Insights
* Delivered to business a pipeline which does the following:
  * Clustering
  * Sentiment Analysis
  * Summarization of dialogue using LLM's
  * Keyword Extraction
  * Visualizations


# Project 4 : Volume Prediction on Warehouses
* Utilizing Dynamic Graph Neural Networks( SpatioTemporal - Time variying features, static graph) to try to model after shipment volume movement worldwide
* Inspired from TGN(Temporal Graph Network) paper by the Twitter team which modeled after twitter users.
* Still in Progress


# Project 5 : 
