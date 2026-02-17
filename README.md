# 📚 Reading Level Assessment using NLP

An NLP-based web application that estimates the **reading difficulty of English text** for school students.  
The tool predicts a **readability score (0–100)** and categorizes content into **Easy, Medium, or Hard** levels suitable for **Grades 3–12**.

---

## 🚀 Features
- Predicts **continuous readability score**
- Classifies text into **grade-level difficulty bands**
- Uses **TF-IDF + Ridge Regression**
- Displays popular **readability indices**:
  - Flesch Reading Ease
  - Flesch-Kincaid Grade
  - SMOG Index
  - Dale–Chall Score
- Upload `.txt` files or paste text directly
- Interactive and clean **Streamlit UI**

---

## 🧠 Tech Stack
- **Python**
- **Streamlit**
- **Scikit-learn**
- **TF-IDF Vectorization**
- **Ridge Regression**
- **TextStat**
- Pandas, NumPy

---

## 📊 Dataset
- CommonLit / CLEAR-style readability corpus  
- ~5,000 English text excerpts  
- Grades **3–12**
- Teacher-annotated **Flesch Reading Ease** scores

---

## ⚙️ How It Works
1. Input text is vectorized using **TF-IDF (unigrams + bigrams)**
2. A **Ridge Regression model** predicts readability score
3. Score is mapped to:
   - **Easy** → Grade 3–5
   - **Medium** → Grade 6–8
   - **Hard** → Grade 9–12
4. Classical readability metrics are computed for comparison

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Aman4138/Reading-Level-NLP.git
cd Reading-Level-NLP
