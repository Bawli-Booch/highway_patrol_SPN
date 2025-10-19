# 🛣️ Highway Patrolling Dashboard — Streamlit App

## 📘 Overview
The **Highway Patrolling Dashboard** is an interactive Streamlit-based web application designed to visualize and analyze real-time patrolling data for highway monitoring and reporting.  
It helps in **tracking submissions**, **identifying hotspots**, **monitoring issues**, and **analyzing patterns** using map-based and chart-based insights.

---

## 🚀 Features
- 📊 Overview Dashboard with hourly and daily analytics
- 🗺️ Interactive Map with clustering and issue coloring
- 📅 Trend analysis (status and issue-wise)
- 📋 Downloadable, pivoted data tables
- 💾 Works offline or on Streamlit Cloud

---

## ⚙️ Installation & Setup (Offline)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/highway-patrolling-dashboard.git
cd highway-patrolling-dashboard
```

### 2️⃣ Create Virtual Environment
For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App Locally
```bash
streamlit run highway_v15.py
```
The app will launch at [http://localhost:8501](http://localhost:8501)

---

## 🌍 Deployment (Streamlit Cloud)
1. Push the repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Connect your GitHub repo.
4. Choose `highway_v15.py` as the main file.
5. Deploy and share your live app!

---

## 🧭 Data Columns
| Column | Description |
|--------|--------------|
| Latitude | GPS Latitude |
| Longitude | GPS Longitude |
| Status | Patrol status (`हाँ` / `नहीं`) |
| Issue | Reported issue (छुट्टा पशु / दुर्घटना / सड़क पर पार्किंग / अन्य) |
| Agent | Officer / Agent name |
| Created_At | Timestamp of submission |
| Photo | URL of uploaded photo |

---

## 📦 Folder Structure
```
📦 highway-patrolling-dashboard/
 ┣ 📜 highway_v15.py        # Main Streamlit app
 ┣ 📜 requirements.txt       # Dependencies
 ┣ 📜 README.md              # Documentation
 ┣ 📂 data/                  # (Optional) Input CSVs
 ┗ 📂 assets/                # (Optional) Icons / logos
```

---

## 📦 Example requirements.txt
```
streamlit
pandas
plotly
numpy
streamlit-plotly-events
pytz
```

---

## 👨‍💻 Author
**Dev Ved**  
📧 devendrakumarruyal@gmail.com  
🚔 Uttar Pradesh Police | Technology Projects

---

## 🪪 License
Licensed under the **MIT License** — free for modification and redistribution with attribution.
