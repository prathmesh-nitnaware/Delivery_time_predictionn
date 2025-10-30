# 🚚 Amazon Delivery Time Prediction

An AI-powered web application that predicts **Amazon order delivery time** based on real-world factors like weather, traffic, distance, area, and delivery agent details.

The project combines **Machine Learning** and a **Streamlit** interface to deliver a prediction system that’s both accurate and easy to use — ideal for non-technical users.

---

## 🧠 Project Overview

The system predicts **how long it’ll take for a package to reach a customer** using past delivery data and influencing parameters.

This project demonstrates an end-to-end ML pipeline:
- Data cleaning and preprocessing
- Feature engineering (prep time, distance, encoding)
- Model training using Random Forest
- Web app deployment with Streamlit
- Visual analytics (bar graph + dendrogram)

---

## 🎯 Key Objectives

- Predict order **delivery time in hours**
- Identify key factors affecting delivery duration
- Provide a **non-technical, easy-to-use interface**
- Deploy the model as a **Streamlit web app**

---

## ⚙️ Tech Stack

| Layer | Tool / Library | Purpose |
|-------|----------------|----------|
| **Frontend (UI)** | [Streamlit](https://streamlit.io/) | Interactive web interface |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) | Model training and prediction |
| **Data Handling** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) | Data transformation and analysis |
| **Visualization** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [SciPy](https://scipy.org/) | Graphs, dendrograms, feature insights |
| **Model Storage** | [Joblib](https://joblib.readthedocs.io/) | Save and load trained model |
| **Language** | Python 3.10+ | Core programming language |

---

## 🧾 Dataset Description

Each row in the dataset represents a **delivery record** with attributes describing the order, environment, and agent.

| Column | Description |
|--------|-------------|
| **Order_ID** | Unique identifier for each delivery |
| **Agent_Age** | Age of the delivery partner |
| **Agent_Rating** | Average rating given by customers |
| **Store_Latitude / Longitude** | Location of the seller/store |
| **Drop_Latitude / Longitude** | Customer delivery location |
| **Order_Date / Time / Pickup_Time** | Order and pickup timestamps |
| **Weather** | Weather condition (Sunny, Rainy, etc.) |
| **Traffic** | Traffic level (Low, Medium, High) |
| **Vehicle** | Delivery vehicle type |
| **Area** | Type of area (Urban, Semi-Urban, Rural) |
| **Category** | Order size (Small, Medium, Large) |
| **Delivery_Time** | Target variable (actual delivery time) |

---

## 🔬 Machine Learning Workflow

### 🧩 1. Data Preprocessing
- Removed missing and irrelevant data  
- Encoded categorical variables using **LabelEncoder**  
- Engineered new features:
  - **Prep_Time:** Estimated order preparation time  
  - **Distance_km:** Approximate store-to-customer distance from lat/long

---

### 🤖 2. Model Training (`train_model.py`)

A **Random Forest Regressor** is used to learn patterns in data.

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "delivery_time_model.pkl")
```


The system automatically:
- Converts user input into the correct ML-ready format  
- Predicts delivery time in real-time  
- Displays feature impact visually  

---

## 🧠 Non-Technical Explanation

Imagine you’re a customer placing an order on Amazon.  
The app takes into account:
- How far the store is  
- What the traffic looks like  
- The weather  
- The size of your order  
- The type of delivery vehicle  

Then, using thousands of past deliveries as examples, it estimates how long your delivery will take — just like Amazon’s “Arriving in 2 hrs 15 min” system.  

You don’t need to understand machine learning — just fill in details and click **Predict**.

---

## 🧩 System Architecture

            ┌───────────────────────────┐
            │  cleaned_delivery_data.csv │
            └──────────────┬────────────┘
                           │
                           ▼
               Data Cleaning + Feature Engineering
                           │
                           ▼
           ┌─────────────────────────────┐
           │  Random Forest Regressor     │
           └──────────────┬──────────────┘
                           │
                  Model Saved (.pkl)
                           │
                           ▼
           ┌─────────────────────────────┐
           │       Streamlit App          │
           │  • Input Form                │
           │  • Auto-Fill Button          │
           │  • Graphs & Prediction       │
           └─────────────────────────────┘


---

## ⚡ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/prathmesh-nitnaware/Delivery_time_predictionn.git
cd Delivery_time_predictionn
```

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Train the Model
```
python train_model.py
```

### 4️⃣ Launch the Streamlit App
```
streamlit run app.py
```

Then open the provided link (usually http://localhost:8501
).

### 🧾 Requirements
```
pandas
numpy
scikit-learn
seaborn
matplotlib
scipy
streamlit
joblib
```

## 🚀 Future Enhancements

- Integrate **real-time APIs** (traffic, weather)  
- Use **Geopy** or **Google Maps API** for accurate distance calculation  
- Add **delivery time benchmarking dashboard**  
- Deploy to **Streamlit Cloud / AWS EC2**  
- Train a **Gradient Boosting Regressor** for comparison  

---

## 👨‍💻 Author

**Prathmesh Nitnaware**  
🎓 Computer Engineering Student  
💡 Passionate about AI, ML, and real-world problem solving  

📫 [GitHub](https://github.com/prathmesh-nitnaware) | [LinkedIn](https://linkedin.com/in/prathmesh-nitnaware)


---

## 🏁 Conclusion

This project represents a complete **Machine Learning lifecycle** —  
from **data preprocessing → model training → interactive deployment**.

It delivers a user-friendly prediction system that blends **technical accuracy** with **visual simplicity**,  
mirroring how real-world logistics giants like **Amazon** predict and optimize delivery times.



