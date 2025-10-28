# Predictive Delivery Optimizer

### Overview
This project is a Predictive Delivery Optimizer designed to forecast potential delivery delays and provide key insights for logistics optimization.  
It integrates multiple logistics datasets — orders, delivery performance, routes, and fleet information — into a single analytical and predictive system.

The project uses Python, Streamlit, and Machine Learning (Random Forest Classifier) to:
- Predict whether an order will be delayed.
- Analyze key performance indicators.
- Visualize delivery patterns.
- Provide interactive dashboards and insights for better decision-making.

---

### Features
- Data Cleaning & Integration: Combines multiple datasets (`orders.csv`, `delivery_performance.csv`, `routes_distance.csv`, `vehicle_fleet.csv`).
- Feature Engineering: Creates derived metrics like `Delay_Days`, `Fuel_Cost_per_KM`, `Delivery_Efficiency`, `Revenue_per_KM`, and `Satisfaction_Index`.
- Predictive Modeling: Trains a Random Forest classifier to predict delivery delays.
- Interactive Dashboard: Built using Streamlit for real-time analytics and visualization.
- Visualization Suite:
  - Delivery Delay by Priority (Box Plot)
  - Distance vs Delay (Scatter Plot)
  - Customer Satisfaction Distribution (Histogram)
  - Feature Importances (Bar Chart)
- Export Capability: Filtered datasets can be downloaded as CSV files.

---

### Tech Stack
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn, Joblib |
| Visualization | Matplotlib, Plotly |
| Web App Framework | Streamlit |
| File Export | CSV |

---

### Project Structure
📦 Predictive-Delivery-Optimizer
├── main.py
├── app.py
├── requirements.txt
├── README.md
├── Innovation_Brief.pdf
├── delivery_performance.csv
├── orders.csv
├── routes_distance.csv
├── vehicle_fleet.csv
└── processed_data.csv

---

### Installation & Setup
1. Clone this repository
   ```bash
   git clone <your-repo-url>
   cd Predictive-Delivery-Optimizer
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run preprocessing and model training
   ```bash
   python main.py
   ```

4. Launch the dashboard
   ```bash
   streamlit run app.py
   ```

---

### Model Summary
| Metric | Score |
|---------|-------|
| Model | Random Forest Classifier |
| Accuracy | ~85–90% |
| Target Variable | Delayed |
| Key Predictors | Distance, Priority, Traffic Delay, Order Value, Delivery Cost |

---

### Key Insights
- High Priority orders generally have lower delay probability.
- Longer routes correlate with higher delay risk.
- Customer Satisfaction Index is inversely proportional to delay duration.
- Delivery cost inefficiency increases with congestion.

---

### Author
**Maulik Mehrotra**  
B.Tech Computer Science Engineering  
Manipal University Jaipur
