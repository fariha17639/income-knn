# income-knn
# 📈 Income Classification using K-Nearest Neighbors (KNN)

This project applies the **K-Nearest Neighbors (KNN)** algorithm to predict whether an individual's income exceeds $50K/year based on demographic and work-related attributes from the UCI Adult Census dataset.

---

## 🔍 Objective

To build a binary classifier that determines if a person earns `>50K` or `<=50K` annually using features like age, education, occupation, hours per week, and more.

---

## 🧰 Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 🧪 Steps Performed

1. **Data Loading**  
   Loaded the UCI Adult Income dataset and explored its structure.

2. **Data Cleaning**  
   - Removed missing or invalid entries  
   - Handled categorical columns using label encoding  

3. **Feature Selection**  
   Selected relevant features for modeling.

4. **Model Building**  
   - Split data into training and testing sets  
   - Applied `KNeighborsClassifier` from Scikit-learn  
   - Tuned hyperparameters using cross-validation  

5. **Model Evaluation**  
   - Accuracy score  
   - Confusion matrix  
   - Classification report (precision, recall, F1-score)

---

## 📊 Results

- **Best Accuracy**: ~X% *(replace with your actual value)*  
- **Best K Value**: *K = N* *(based on tuning)*  
- **Insight**: KNN performed best with normalized features and optimized `k` parameter.

---

## 📁 File Structure

- `Income dataset (1).ipynb`: Jupyter Notebook with full code and results  
- `README.md`: Project summary and documentation

---

## 📌 Dataset Source

UCI Machine Learning Repository: [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult)
