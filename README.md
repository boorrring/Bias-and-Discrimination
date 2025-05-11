# 📌 Bias and Discrimination Check Using BERT & Reddit Scraping

## 📖 Overview
This project addresses the issue of hate speech and biased content on online platforms using state-of-the-art NLP models such as BERT, DistilBERT, ALBERT, and HateBERT. It includes both training on labeled datasets and real-time hate speech detection using Reddit scraping.

## 🎯 Objectives
- Detect hate speech and discriminatory content in online text.
- Use transformer-based models for context-aware classification.
- Evaluate and compare multiple models (DistilBERT, ALBERT, HateBERT).
- Integrate Reddit scraping to test on real-world data.
- Develop an easy-to-use Gradio interface for predictions.

## 🧠 Models Used
- **DistilBERT**
- **ALBERT**
- **HateBERT**
- Ensemble model via majority voting

## 🛠️ Tools & Libraries
- `transformers`, `torch`, `pandas`, `numpy`, `sklearn`
- `praw` (for Reddit scraping)
- `gradio` (for GUI)
- `deep-translator` (for data augmentation)
- `nltk`, `keras_tuner`

## 🏗️ System Architecture

The system is designed as a modular pipeline that processes input text or live Reddit comments, classifies them using fine-tuned transformer models, and returns a prediction through a user-friendly interface.

### 🔄 Workflow

```text
[Data Collection (Kaggle Dataset / Reddit Scraper)]
        ↓
[Text Preprocessing (Cleaning, Tokenization, Stopword Removal, etc.)]
        ↓
[Data Augmentation (Back Translation)]
        ↓
[Model Inference (DistilBERT / ALBERT / HateBERT)]
        ↓
[Ensemble Voting (Majority Decision)]
        ↓
[Gradio GUI Output: Hateful / Non-Hateful]

```



## 🧪 Testing
- Tested on Kaggle's labeled hate speech dataset and live Reddit data.
- Evaluation metrics used: **Accuracy**, **Precision**, **Recall**, **F1-Score**.
- Real-time classification tested through Gradio interface with response times < 1 second.

## 💡 Features
- Real-time Reddit comment scraping and classification.
- Ensemble predictions from three transformer models.
- Intuitive Gradio web interface for testing custom inputs.
- Back-translation-based data augmentation.
- Bias and fairness analysis via cross-model comparison.

## 🧩 Limitations
- Only supports English language text.
- May not handle sarcasm or deeply implicit hate speech effectively.
- High GPU/CPU usage due to use of multiple transformer models.

## 🔮 Future Enhancements
- Add multilingual support (e.g., Hindi, code-mixed).
- Integrate explainability (e.g., SHAP, LIME).
- Add human-in-the-loop feedback for moderation.
- Deploy using cloud services (AWS, GCP, Azure).
- Build an admin dashboard for real-time monitoring and alerts.

