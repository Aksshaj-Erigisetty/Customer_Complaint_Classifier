# Customer Complaint Classifier  

A Natural Language Processing (NLP) project that classifies **customer complaints** into multiple categories using **TF-IDF vectorization** and **Logistic Regression**.  
Additionally, it performs **sentiment analysis** using TextBlob to detect whether complaints are written in a positive, negative, or neutral tone.  
Developed as part of an applied **Data Science & Machine Learning** portfolio.

---

## Project Overview  

Customer complaints contain valuable insights about service quality and recurring issues faced by users.  
This project leverages **machine learning and NLP** to automatically classify complaints and uncover overall customer sentiment.  

The primary goals of this project were to:  
- Build a text classification model that categorizes complaints by **product/service type**.  
- Perform **sentiment analysis** to understand the emotional tone of customer feedback.  
- Visualize complaint distribution and sentiment trends using **Matplotlib**.

---

## Workflow: CRISP-DM Framework  

1. **Business Understanding** – Identify how text analytics can improve customer service and complaint handling.  
2. **Data Understanding** – Explore patterns across 10+ complaint categories and thousands of textual narratives.  
3. **Data Preparation** – Clean text by removing punctuation, special characters, and converting to lowercase.  
4. **Modeling** – Apply **TF-IDF** to transform text into numeric features, and train a **Logistic Regression** classifier.  
5. **Evaluation** – Measure model accuracy, precision, recall, and F1-score to validate performance.  
6. **Deployment** – Visualize key insights to assist organizations in understanding complaint trends.

---

## Technologies Used  

- **Language:** Python  
- **Libraries:** Pandas, NumPy, scikit-learn, TextBlob, Matplotlib  
- **Environment:** Google Colab / Jupyter Notebook  
- **Methodology:** TF-IDF vectorization + Logistic Regression classification  

---

## Repository Structure  

```

Complaint_Classifier/
├── README.md
├── requirements.txt
├── Customer_Complaint_Classifier.ipynb
├── consumer_complaints.csv 
├── results/
│ ├── top_10_categories.png
│ └── sentiment_distribution.png
└── .gitignore
```


---

## Dataset  

**Source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)  
**File:** `consumer_complaints.csv`  
**Size:** ~2 million complaints, 18 columns  

### Key Columns Used  
| Column | Description |
|:--------|:-------------|
| `product` | Product category of the complaint (target label) |
| `consumer_complaint_narrative` | Text narrative of the complaint (features) |

The dataset contains complaint narratives submitted by consumers to the **Consumer Financial Protection Bureau (CFPB)**, categorized by products such as:
- Debt Collection  
- Mortgage  
- Credit Reporting  
- Bank Account or Service  
- Consumer Loan  
- Student Loan  

---

##  Data Preprocessing  

1. Selected only two columns: `product` and `consumer_complaint_narrative`  
2. Dropped rows with missing complaint text  
3. Cleaned text using regular expressions:  
   - Lowercased all text  
   - Removed punctuation, numbers, and extra whitespace  
4. Created a new column `Cleaned_Complaint` for processed text  

---

## Model Development  

### **TF-IDF Vectorization**
- Converted complaint text into numerical features using Term Frequency–Inverse Document Frequency.  
- Limited vocabulary to top **5,000** most informative features.  
- Removed English stop words.  

### **Logistic Regression**
- Chosen for its interpretability and efficiency on high-dimensional data.  
- Model hyperparameters:  
  - `max_iter = 1000`  
  - Regularization = L2 (default)  

---

## Model Evaluation  

| Metric | Score |
|:--------|:------:|
| **Accuracy** | **0.848** |
| **Macro Avg F1-Score** | **0.68** |
| **Weighted Avg F1-Score** | **0.85** |

The model achieved a strong **~85% overall accuracy** across multiple product categories.  
Major classes such as *Debt Collection*, *Mortgage*, and *Credit Reporting* showed particularly high F1-scores (0.80–0.90 range).  

---

### Note on Imbalanced Data
Some categories (like “Other Financial Service”) have very few samples, which explains the low F1-scores for those specific classes.  
Despite this, the overall classifier performs robustly across major complaint types.

---

## Sentiment Analysis  

Used the **TextBlob** library to compute sentiment polarity for each complaint:  
- Range: −1.0 (Negative) → 0.0 (Neutral) → +1.0 (Positive)  

| Example Complaint | Sentiment |
|--------------------|-----------|
| “I was overcharged on my mortgage payment.” | −0.12 |
| “The representative was very helpful and resolved my issue.” | +0.45 |
| “Due to inconsistencies in the amount owed…” | +0.08 |

Most complaints are **neutral to slightly negative**, reflecting factual tone rather than emotional writing.

---

## Visualizations  

### 1. **Top 10 Complaint Categories**
Shows which types of financial products receive the most complaints.

![Top 10 Complaint Categories](results/top_10_categories.png)

### 2. **Sentiment Distribution**
Histogram of sentiment polarity for all complaints.

![Sentiment Distribution](results/sentiment_distribution.png)

---

## Key Insights  

### Analytical Observations  
- **Debt Collection** and **Mortgage** dominate the complaint volume.  
- **Neutral sentiment** dominates, indicating customers report issues factually.  
- TF-IDF with Logistic Regression provides a balance between **performance** and **interpretability**.  

### Business Value  
- Automating complaint classification reduces manual effort for regulatory agencies.  
- Sentiment trends can guide customer service improvements and compliance monitoring.  

---

##  Limitations & Future Improvements  

| Area | Suggestion |
|:------|:------------|
| **Data Imbalance** | Apply SMOTE or class weighting for underrepresented categories. |
| **Advanced NLP Models** | Try BERT or DistilBERT for higher accuracy. |
| **Dashboard Integration** | Visualize results interactively with Power BI or Streamlit. |

---


