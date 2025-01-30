# 🎬 Explainable AI for Deep Sentiment Analysis of Movie Reviews

## 📌 Project Overview
This project aims to enhance sentiment analysis by providing **explainability** in **movie reviews**. Instead of just predicting whether a review is positive or negative, we break down *what specific aspects of the movie contributed to the sentiment*—whether it was the **music, cast, story, cinematography, or direction**.

We achieve this using:
- **Named Entity Recognition (NER)** fine-tuned on IMDb movie reviews to extract key aspects.
- **LIME (Local Interpretable Model-Agnostic Explanations)** to understand the impact of different aspects on the sentiment.
- **Attention Visualization** to see how our deep learning model focuses on different parts of the text.

## 📂 Dataset
We used the **IMDb movie review dataset**, which consists of **50,000 reviews** labeled as positive or negative.

We preprocess these reviews to:
- Identify aspects (e.g., *music*, *acting*, *story*).
- Extract relevant keywords.
- Associate sentiment scores to aspects.

## 🏗️ Architecture
![Image](https://github.com/user-attachments/assets/b942296f-179e-48f8-9c95-d603e8beb8d9)
1. **Fine-Tuned NER Model**: BERT based NER model parses movie reviews to classify words under aspects (e.g., "music", "acting").
2. **Sentiment Classifier**: Uses attention-based deep learning to classify the review.
3. **LIME XAI Model**: Highlights words that contribute most to the review’s sentiment.
4. **Attention Visualization**: Displays heatmaps showing which words influenced predictions.

## 📊 Key Features
- 🔍 **Aspect-Based Sentiment Analysis**: Extracts what people liked or disliked in a movie.
- 🧠 **Explainable AI (XAI) Integration**: LIME shows which words influenced the sentiment.
- 🎨 **Attention Maps**: Visualizes how the model interprets reviews.
- 🏷️ **Named Entity Recognition (NER)**: Finds movie-related aspects dynamically.

## 📷 Model Interpretability & Visualizations
🔹 Evluation Metric Graph
![Image](https://github.com/user-attachments/assets/61e89b12-98ac-45e2-98de-a93837f7d543)

🔹 Attention Heatmaps
![Image](https://github.com/user-attachments/assets/467eb8f2-5764-45f9-a188-15360f3beb48)
![Image](https://github.com/user-attachments/assets/ac64b40a-1b83-419d-b658-32308900a1d8)

🔹 LIME Interpretability Output
![Image](https://github.com/user-attachments/assets/ae93c2c9-e553-47e9-9286-4fe63eb6eb91)

## 📜 Key Findings
The NER model accurately identifies aspects like cinematography, acting, and story.
LIME explanations highlight words that most influence sentiment predictions.
Attention heatmaps provide a deeper understanding of how the model interprets text.

## 🚀 Future Work
Expand dataset support.
Improve aspect extraction using transformers.
Integrate additional XAI techniques (e.g., SHAP).

## 🔗 References
- [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) - Official dataset used for movie reviews.
- [LIME: Local Interpretable Model-Agnostic Explanations](https://arxiv.org/abs/1602.04938) - Research paper explaining the LIME model for interpretability.
- [NER with SpaCy](https://spacy.io/usage/training) - Documentation on fine-tuning Named Entity Recognition (NER) models using SpaCy.
- [Attention Mechanisms in NLP](https://www.aclweb.org/anthology/N16-1174/) - Explains how attention mechanisms work in deep learning models.
- [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/) - Alternative explainability method for machine learning models.

---
