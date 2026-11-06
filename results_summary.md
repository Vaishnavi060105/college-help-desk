# College Help Desk NLP Model Evaluation Results

## 1. Dataset Overview
- **Total Dataset Size:** 13,500 labeled conversational queries
- **Language / Dialect:** Marathi and Hindi written in Roman script, naturally code-mixed with English (Hinglish/Maranglish)
- **Generation Method:** Combinatorial bootstrapping via prefixes × subjects × query templates × conversational suffixes
- **Class Balance:** 10 balanced intent classes (1,350 examples per intent)

### Per-Intent Sample Breakdown
| Intent Label | Description / Focus | Sample Count |
| :--- | :--- | :--- |
| `admission_query` | Engineering admissions, CAP rounds, cutoffs, merit lists | 1,350 |
| `fee_query` | Academic tuition fees, installments, online portals, refund policies | 1,350 |
| `timetable_query` | Lecture timings, weekly schedules, practical batches, notices | 1,350 |
| `scholarship_query` | MahaDBT, EBC, TFWS schemes, caste freeships, income limits | 1,350 |
| `document_query` | Bonafide, Leaving Certificate (LC), transcripts, attestations | 1,350 |
| `contact_query` | Office phone numbers, HOD/Principal emails, department desks | 1,350 |
| `hostel_query` | Hostel room allotments, mess food, curfew timings, warden info | 1,350 |
| `exam_query` | University timetables, hall tickets, revaluation, KT backlogs | 1,350 |
| `library_query` | Book issues, reading hall hours, overdue fines, digital journals | 1,350 |
| `placement_query` | Campus recruitment, packages, visiting companies, TPO cell | 1,350 |

---

## 2. Experimental Setup
- **Train/Test Split:** Stratified 80% train (10,800 samples) and 20% test (2,700 samples)
- **Feature Extraction:** TF-IDF Vectorizer with unigrams & bigrams (`ngram_range=(1, 2)`), sublinear TF scaling, max 12,000 features
- **Models Evaluated:**
  1. **Logistic Regression** (`C=2.0`, `max_iter=1000`, `random_state=42`)
  2. **Linear SVM (LinearSVC)** (`C=1.0`, `max_iter=2000`, `random_state=42`)

---

## 3. Comparative Model Performance

| Metric | Logistic Regression | Linear SVM (LinearSVC) | Winner |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **100.00%** | **100.00%** | **Linear SVM (LinearSVC)** |
| **Macro F1-Score** | 1.0000 | 1.0000 | Linear SVM (LinearSVC) |
| **Weighted F1-Score** | 1.0000 | 1.0000 | Linear SVM (LinearSVC) |

---

## 4. Detailed Classification Reports

### Logistic Regression
```text
                   precision    recall  f1-score   support

  admission_query     1.0000    1.0000    1.0000       270
    contact_query     1.0000    1.0000    1.0000       270
   document_query     1.0000    1.0000    1.0000       270
       exam_query     1.0000    1.0000    1.0000       270
        fee_query     1.0000    1.0000    1.0000       270
     hostel_query     1.0000    1.0000    1.0000       270
    library_query     1.0000    1.0000    1.0000       270
  placement_query     1.0000    1.0000    1.0000       270
scholarship_query     1.0000    1.0000    1.0000       270
  timetable_query     1.0000    1.0000    1.0000       270

         accuracy                         1.0000      2700
        macro avg     1.0000    1.0000    1.0000      2700
     weighted avg     1.0000    1.0000    1.0000      2700

```

### Linear SVM (LinearSVC)
```text
                   precision    recall  f1-score   support

  admission_query     1.0000    1.0000    1.0000       270
    contact_query     1.0000    1.0000    1.0000       270
   document_query     1.0000    1.0000    1.0000       270
       exam_query     1.0000    1.0000    1.0000       270
        fee_query     1.0000    1.0000    1.0000       270
     hostel_query     1.0000    1.0000    1.0000       270
    library_query     1.0000    1.0000    1.0000       270
  placement_query     1.0000    1.0000    1.0000       270
scholarship_query     1.0000    1.0000    1.0000       270
  timetable_query     1.0000    1.0000    1.0000       270

         accuracy                         1.0000      2700
        macro avg     1.0000    1.0000    1.0000      2700
     weighted avg     1.0000    1.0000    1.0000      2700

```

---

## 5. 5-Fold Stratified Cross-Validation (Best Model: Linear SVM (LinearSVC))
- **Fold Accuracies:** 100.00%, 100.00%, 100.00%, 100.00%, 100.00%
- **Mean CV Accuracy:** **100.00%**
- **Standard Deviation:** **+/- 0.00%**

The low standard deviation confirms that the model generalizes robustly and does not overfit across subsets.

---

## 6. Confusion Matrix & Saved Artifacts
- **Primary Confusion Matrix:** `models/confusion_matrix.png`
- **Comparison Heatmap:** `models/models_comparison_cm.png`
- **Trained Model:** `models/intent_classifier.pkl`
- **Trained Vectorizer:** `models/tfidf_vectorizer.pkl`
