# Athanasios Rakitzis Portfolio
AI - Data Science Portfolio
<br>
After each project a list of keywords is included indicating libraries, frameworks and Theoretical concepts used.
<br>


# Project 1 : Customer Call Prediction
* Created a from-scratch GPT-1 inspired model trained on package routing sequences to predict the next probable checkpoint for a package, using features such as time, time difference between checkpoints , current facilities etc. Instead of a natural language vocabulary, we use checkpoint abbreviations.
* Fine-tuned the model to also predict if and when a customer will call, based on each sequence's label.
* Incorporated Uncertainty Estimation in the Transformer model by combining state-of-the-art methods to filter out ambiguous or Out-of-Distribution (OoD) data points.
* Built a pipeline for clustering and sentiment analysis of customer comments, providing visualizations and various insights on the text data relevant to the package.
* Keywords : Pytorch, PyLightning, Gaussian Mixture Models, Transformers Uncertainty Estimation, Regex, BERTopic.



# Project 2 : Website Recommender System & Lead Conversion (without individual user data)
*  Constructed a graph network with URLs as nodes and user navigation paths as edges to understand website user behavior.
*  Leveraged Graph Neural Networks and Node2Vec for node embeddings. Utilized  Link Prediction methodology from GNN to create probable yet non-existent connections between URL.
*  Employed BERT-based semantic similarity for automated article recommendations, enriched with Node2Vec embeddings. Bert Embeddings were concatenated with Node2Vec embeddings for enhanced reccomendations.
*  Applied Beam Search Algorithm in conjunction with Markov Chain transition probabilities to nudge users toward lead conversion/account registration.
*  Keywords : Networkx, Pytorch Geometric, Node2Vec,Semantic Similarity.

# Project 3 : Conversational Chatbot Analytics for Customer Support
* Delivered to business a pipeline which does the following:
  * Text Clustering
  * Sentiment Analysis
  * Chat summarization using Large Language Models
  * Keyword Extraction
  * Data Visualizations, Clustering Visualizations, Topic Rivers/Stream Graphs etc.
* Keywords : Ascect Semantic Similarity, KeyBert, UMAP, HDBSCAN, Yake.


# Project 4 : Volume Prediction on Warehouses
* Utilizing Dynamic Graph Neural Networks(SpatioTemporal - Time variying features, static graph) to try to model after shipment volume movement worldwide
* Inspired from TGN(Temporal Graph Network) paper by the Twitter team which modeled after twitter users.
* Still in Progress
* Keywords : Spacio-Temporal Graph Neural Networks.


# Project 5 : Speech to Text from Calls
* Built a pipeline to transcribe calls made to customer service into text.
* Applied techniques from Project 3 such as summarization, keyword analysis, clustering, and visualization to the transcribed text.
* Future work involves integrating text-generating Language Models like LLaMMa-2.
* Keywords : Whisper, BERTopic.


# Project 6 : Card Classification + XAI
* Developed an image classification model using a Convolutional Neural Network trained on Kaggle data.
* Created a dataset on my own with photos taken of cards on various angles, distortions, objects and drawings around etc.
* Fine-tuned the model using this dataset and applied LIME and SHAP for model interpretability and error analysis.
* Keywords : Tensorflow, LIME, SHAP,

# Project 7 :  GDPR Fine Amount Prediction
* Conducted data analysis on GDPR fines imposed on Universities and Public Institutions, utilizing data from the [GDPR Enforcement Tracker website](https://www.enforcementtracker.com/), which includes summaries of issued fines, country, institution, date of decision, and fine amount.
* Employed text mining techniques on the summary of court decisions to extract features that could serve as predictors in the fine amount prediction models.
* Built and compared the performance of two regression models and a decision tree for predicting the fine amount, all showing strong predictive capabilities.
* Identified violated articles and the type of data involved as the most important predictors of fine amount. Specifically, violation of Article 32 (Implementing sufficient security measures) was found to be the most influential factor.
* Identified factors leading to fines and to make accurate fine predictions for public organizations yet to be fined.
* Feature Selection done  using PCA to select features which explain most variance + Relevance
* Keywords : PCA, Scikit Learn, Regex.


# Project 8 : Binge Eating Disorder Prediction
* Conducted a predictive analysis on Binge Eating Episodes among 120 participants using a variety of statistical models including mixed models and MERF(Mixed Effects Random Forests), with a focus on model explainability. 
* Achieved the highest AUC score of 0.7529 with a mixed model based on emotions and time variables.
* Identified key predictors influencing BEEs as emotions "calm," "stress," "boredom," "guilt," time of day and year, and participants’ willingness to restrict eating behavior. 
* Data taken from a mobile application (mEMA) app which gathered real-time emotional and behavioral data from participants, aiming for real-world application use of the data.
* Keywords : MERF, Mixed Effects models.


# Project 9 : Homer's Odyssey Text Mining
* Analyzed the classical Ancient Greek book using NLP algorithms
* Named Entity Recognition & Topic Modeling & Sentiment Analysis
* EDA on frequent phrases, entities etc.
* Character - Location co-occurence and timeline creation
* Animated map of character movement based on time/page
* Keywords : Text Mining

