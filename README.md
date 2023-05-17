# Chat-Analysis-Using-Langchain-Library
# Chat Conversations Analysis

This project focuses on analyzing chat conversations to extract insights such as sentiment analysis, topic modeling, and named entity recognition. It provides two implementations: one using the `nltk`, `sklearn`, and `spacy` libraries, and another using the `langchain` library.

## Code 1: Using NLTK, sklearn, and spaCy

The code in this section demonstrates how to perform analysis on chat conversations using NLTK, sklearn, and spaCy libraries.

### Preprocessing

The `preprocess_text` function is responsible for removing irrelevant data from the chat conversations, such as emojis, images, and links. It then splits the text into individual words or tokens, returning the preprocessed text.

### Sentiment Analysis

The `perform_sentiment_analysis` function uses the `SentimentIntensityAnalyzer` from NLTK to perform sentiment analysis on the chat conversations. It calculates the sentiment score for each message using a polarity scoring mechanism and returns the average sentiment score.

### Topic Modeling

The `perform_topic_modeling` function utilizes the `CountVectorizer` and `LatentDirichletAllocation` classes from sklearn to perform topic modeling on the preprocessed chat conversations. It creates a document-term matrix using the vectorizer, and then applies Latent Dirichlet Allocation to identify main topics in the conversations.

### Named Entity Recognition

The `perform_named_entity_recognition` function uses the spaCy library to perform named entity recognition on the chat conversations. It extracts important entities such as people, organizations, and locations from the text, returning a list of entities along with their corresponding labels.

## Code 2: Using the langchain Library

The code in this section demonstrates how to achieve similar chat conversation analysis using the `langchain` library.

### Data Preprocessing

The `preprocess` function from the `langchain` library is used to preprocess the chat conversations by removing stop words, punctuation, and other irrelevant information.

### Tokenization and Bag of Words

The `tokenize` function is applied to the preprocessed conversations to tokenize them into individual words. The `create_bag_of_words` function is then used to create a bag of words model from the tokenized conversations.

### Sentiment Analysis and Identification

The `sentiment_analysis` function performs sentiment analysis on the bag of words model generated earlier. The `identify_sentiment` function is used to determine the overall sentiment of the chat conversations.

### Topic Modeling

The `topic_modeling` function applies topic modeling using the `tokenized_conversations` from earlier. It identifies the main topics being discussed in the conversations.

### Named Entity Recognition and Identification

The `named_entity_recognition` function performs named entity recognition on the preprocessed conversations, extracting important entities such as people, organizations, and locations. The `identify_entities` function is then used to identify the most important entities and their frequency of occurrence.

## Example Usage

An example usage is provided for both code implementations, showcasing how to apply the analysis functions to a list of chat conversations. The extracted insights, such as average sentiment, main topics, and named entities, are accessed and displayed.


