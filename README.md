# 🌾 Fasal Mulya Bhavishyavani — Crop Price Prediction App
## फसल मूल्य भविष्यवाणी — किसान सहायक वेब ऐप

A farmer-friendly crop price prediction web application for India.

---

## 📁 Project Structure

```
crop_price_app/
├── app.py               # Main Flask application (backend + ML)
├── init_db.py           # Database initialization script
├── requirements.txt     # Python dependencies
├── crop_data.db         # SQLite database (auto-created)
├── templates/
│   └── index.html       # Main UI (3D design, Chart.js)
├── static/
│   ├── css/             # Additional stylesheets (optional)
│   └── js/              # Additional scripts (optional)
└── README.md            # This file
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Internet connection (for API calls)

### 2. Install Dependencies
```bash
cd crop_price_app
pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```
Database is auto-created on first run with 100+ crops.

### 4. Open in Browser
```
http://localhost:5000
```

---

## 🌐 API Integration

### Agmarknet (Data.gov.in)
- **Endpoint**: `https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070`
- **API Key**: Free public key included
- **Data**: Daily mandi prices across India
- **Fallback**: Simulated statistical data if API is unavailable

### Getting Your Own API Key
1. Visit: https://data.gov.in/
2. Register for free
3. Navigate to Agmarknet dataset
4. Copy your API key into `app.py` → `API_KEY` variable

---

## 🧠 Prediction Logic

The app uses **Polynomial Regression** (degree 2):

```
price = a₀ + a₁×t + a₂×t² + seasonal_factor + noise
```

1. **Historical Data**: 12 months of price data (API + simulated)
2. **Model Training**: Polynomial regression fit on historical prices
3. **Seasonal Adjustment**: Month-specific factors (kharif/rabi seasons)
4. **Future Prediction**: 6-month price forecast
5. **Trend Analysis**: Rising / Falling / Stable classification

---

## 🗄️ Database Schema

```sql
-- 100+ crops with Hindi names
CREATE TABLE crops (
    id              INTEGER PRIMARY KEY,
    name            TEXT,       -- English name
    hindi_name      TEXT,       -- Hindi name
    category        TEXT,       -- Cereals, Pulses, Vegetables, etc.
    unit            TEXT,       -- Quintal, Kg, etc.
    avg_base_price  REAL        -- Base price for simulation
);

-- Cache for API responses (reduces bandwidth)
CREATE TABLE price_cache (
    id          INTEGER PRIMARY KEY,
    crop_name   TEXT,
    state       TEXT,
    market      TEXT,
    price       REAL,
    date        TEXT,
    fetched_at  TEXT
);
```

---

## 📱 UI Features

- ✅ 3D glass-morphism card design
- ✅ Yellow/orange/saffron gradient theme
- ✅ Animated crop particles background
- ✅ Hindi + English bilingual labels
- ✅ Responsive (mobile-friendly)
- ✅ Chart.js price history + forecast graph
- ✅ Smart selling suggestions
- ✅ Live mandi prices table (when API available)
- ✅ Loading spinner during prediction
- ✅ Error handling with clear messages

---

## 🚀 Production Deployment

### Using Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Environment Variables
```bash
export FLASK_SECRET_KEY="your-secret-key"
export AGMARKNET_API_KEY="your-api-key"
```

---

## 📞 Support
Built for Indian farmers 🇮🇳 | किसान मित्र
Data source: Agmarknet, Data.gov.in
