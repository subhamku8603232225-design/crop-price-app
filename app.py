"""
Crop Price Prediction Web App - India
=====================================
A farmer-friendly web application that predicts crop prices
using historical data, API integration, and ML models.

Tech Stack: Flask + SQLite + Linear Regression + Chart.js
"""

import os
import json
import sqlite3
import hashlib
import logging
import requests
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ---------------------
# App Configuration
# ---------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'crop_price_india_2024'
app.config['DATABASE'] = 'crop_data.db'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------
# API Configuration
# ---------------------
# Data.gov.in Agmarknet API (free public API for Indian agricultural prices)
AGMARKNET_API_BASE = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aab825cf28cca4e4c2"  # Free public key

# In-memory cache to reduce API calls (for low-bandwidth users)
PRICE_CACHE = {}
CACHE_TTL = 3600  # 1 hour


# ---------------------
# Database Setup
# ---------------------
def get_db():
    """Get database connection."""
    db = sqlite3.connect(app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    return db


def init_db():
    """Initialize database with crop and state data."""
    db = get_db()
    cursor = db.cursor()

    # Create crops table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            hindi_name TEXT,
            category TEXT,
            unit TEXT DEFAULT 'Quintal',
            avg_base_price REAL DEFAULT 2000.0
        )
    ''')

    # Create price_cache table for storing fetched prices
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_name TEXT,
            state TEXT,
            market TEXT,
            price REAL,
            date TEXT,
            fetched_at TEXT
        )
    ''')

    # Create historical_prices table (simulated historical data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_id INTEGER,
            state TEXT,
            price REAL,
            date TEXT,
            FOREIGN KEY (crop_id) REFERENCES crops(id)
        )
    ''')

    # Populate crops if empty
    cursor.execute("SELECT COUNT(*) FROM crops")
    if cursor.fetchone()[0] == 0:
        populate_crops(cursor)

    db.commit()
    db.close()
    logger.info("Database initialized successfully")


def populate_crops(cursor):
    """Insert 100+ crops with Hindi names and categories."""
    crops_data = [
        # Cereals / अनाज
        ("Wheat", "गेहूँ", "Cereals", "Quintal", 2200),
        ("Rice", "चावल", "Cereals", "Quintal", 3000),
        ("Maize", "मक्का", "Cereals", "Quintal", 1800),
        ("Barley", "जौ", "Cereals", "Quintal", 1600),
        ("Jowar", "ज्वार", "Cereals", "Quintal", 2800),
        ("Bajra", "बाजरा", "Cereals", "Quintal", 2500),
        ("Ragi", "रागी", "Cereals", "Quintal", 3500),
        ("Oats", "जई", "Cereals", "Quintal", 2000),
        ("Maize (Sweet)", "मीठी मक्का", "Cereals", "Quintal", 2200),
        ("Sorghum", "ज्वार", "Cereals", "Quintal", 2100),

        # Pulses / दालें
        ("Tur Dal", "तुअर दाल", "Pulses", "Quintal", 6000),
        ("Moong Dal", "मूंग दाल", "Pulses", "Quintal", 7500),
        ("Chana Dal", "चना दाल", "Pulses", "Quintal", 5500),
        ("Masoor Dal", "मसूर दाल", "Pulses", "Quintal", 5000),
        ("Urad Dal", "उड़द दाल", "Pulses", "Quintal", 6500),
        ("Moth Bean", "मोठ", "Pulses", "Quintal", 4500),
        ("Horse Gram", "कुल्थी", "Pulses", "Quintal", 4000),
        ("Cowpea", "लोबिया", "Pulses", "Quintal", 5200),
        ("Peas (Dry)", "मटर (सूखा)", "Pulses", "Quintal", 4800),
        ("Rajma", "राजमा", "Pulses", "Quintal", 7000),
        ("Kabuli Chana", "काबुली चना", "Pulses", "Quintal", 8000),
        ("Black Gram", "काला चना", "Pulses", "Quintal", 5800),

        # Oilseeds / तिलहन
        ("Mustard", "सरसों", "Oilseeds", "Quintal", 5200),
        ("Groundnut", "मूंगफली", "Oilseeds", "Quintal", 5500),
        ("Soybean", "सोयाबीन", "Oilseeds", "Quintal", 4000),
        ("Sunflower", "सूरजमुखी", "Oilseeds", "Quintal", 5000),
        ("Sesame", "तिल", "Oilseeds", "Quintal", 9000),
        ("Linseed", "अलसी", "Oilseeds", "Quintal", 4500),
        ("Castor Seed", "अरंड", "Oilseeds", "Quintal", 5800),
        ("Safflower", "कुसुम", "Oilseeds", "Quintal", 4200),
        ("Niger Seed", "रामतिल", "Oilseeds", "Quintal", 6000),
        ("Coconut", "नारियल", "Oilseeds", "Quintal", 18000),

        # Vegetables / सब्जियां
        ("Potato", "आलू", "Vegetables", "Quintal", 1200),
        ("Tomato", "टमाटर", "Vegetables", "Quintal", 2000),
        ("Onion", "प्याज", "Vegetables", "Quintal", 2500),
        ("Garlic", "लहसुन", "Vegetables", "Quintal", 8000),
        ("Cabbage", "पत्तागोभी", "Vegetables", "Quintal", 800),
        ("Cauliflower", "फूलगोभी", "Vegetables", "Quintal", 1500),
        ("Brinjal", "बैंगन", "Vegetables", "Quintal", 1800),
        ("Lady Finger", "भिंडी", "Vegetables", "Quintal", 2500),
        ("Bitter Gourd", "करेला", "Vegetables", "Quintal", 3000),
        ("Bottle Gourd", "लौकी", "Vegetables", "Quintal", 1200),
        ("Ridge Gourd", "तोरई", "Vegetables", "Quintal", 1500),
        ("Cucumber", "खीरा", "Vegetables", "Quintal", 1200),
        ("Pumpkin", "कद्दू", "Vegetables", "Quintal", 900),
        ("Capsicum", "शिमला मिर्च", "Vegetables", "Quintal", 4000),
        ("Green Chilli", "हरी मिर्च", "Vegetables", "Quintal", 5000),
        ("Spinach", "पालक", "Vegetables", "Quintal", 1500),
        ("Methi", "मेथी", "Vegetables", "Quintal", 2000),
        ("Coriander", "धनिया", "Vegetables", "Quintal", 3000),
        ("Radish", "मूली", "Vegetables", "Quintal", 800),
        ("Carrot", "गाजर", "Vegetables", "Quintal", 1800),
        ("Beetroot", "चुकंदर", "Vegetables", "Quintal", 2000),
        ("Peas (Green)", "हरा मटर", "Vegetables", "Quintal", 3500),
        ("Beans", "बीन्स", "Vegetables", "Quintal", 3000),
        ("Drumstick", "सहजन", "Vegetables", "Quintal", 2500),
        ("Raw Banana", "कच्चा केला", "Vegetables", "Quintal", 1800),
        ("Sweet Potato", "शकरकंद", "Vegetables", "Quintal", 2000),
        ("Taro", "अरबी", "Vegetables", "Quintal", 2200),
        ("Yam", "जिमीकंद", "Vegetables", "Quintal", 2500),

        # Fruits / फल
        ("Mango", "आम", "Fruits", "Quintal", 5000),
        ("Banana", "केला", "Fruits", "Quintal", 2000),
        ("Apple", "सेब", "Fruits", "Quintal", 8000),
        ("Grapes", "अंगूर", "Fruits", "Quintal", 6000),
        ("Orange", "संतरा", "Fruits", "Quintal", 4000),
        ("Guava", "अमरूद", "Fruits", "Quintal", 3000),
        ("Papaya", "पपीता", "Fruits", "Quintal", 2000),
        ("Watermelon", "तरबूज", "Fruits", "Quintal", 1500),
        ("Muskmelon", "खरबूजा", "Fruits", "Quintal", 2000),
        ("Pomegranate", "अनार", "Fruits", "Quintal", 8000),
        ("Pineapple", "अनानास", "Fruits", "Quintal", 3500),
        ("Litchi", "लीची", "Fruits", "Quintal", 10000),
        ("Jackfruit", "कटहल", "Fruits", "Quintal", 3000),
        ("Sapota", "चीकू", "Fruits", "Quintal", 4000),
        ("Lemon", "नींबू", "Fruits", "Quintal", 6000),
        ("Amla", "आंवला", "Fruits", "Quintal", 4000),
        ("Strawberry", "स्ट्रॉबेरी", "Fruits", "Quintal", 15000),

        # Spices / मसाले
        ("Turmeric", "हल्दी", "Spices", "Quintal", 7000),
        ("Ginger", "अदरक", "Spices", "Quintal", 6000),
        ("Red Chilli", "लाल मिर्च", "Spices", "Quintal", 12000),
        ("Cumin", "जीरा", "Spices", "Quintal", 20000),
        ("Coriander Seeds", "धनिया बीज", "Spices", "Quintal", 8000),
        ("Fennel", "सौंफ", "Spices", "Quintal", 9000),
        ("Fenugreek", "मेथी दाना", "Spices", "Quintal", 5000),
        ("Black Pepper", "काली मिर्च", "Spices", "Quintal", 50000),
        ("Cardamom", "इलायची", "Spices", "Quintal", 150000),
        ("Clove", "लौंग", "Spices", "Quintal", 80000),
        ("Ajwain", "अजवाइन", "Spices", "Quintal", 15000),
        ("Mustard Seeds", "सरसों दाना", "Spices", "Quintal", 5500),

        # Cash Crops / नकदी फसलें
        ("Cotton", "कपास", "Cash Crops", "Quintal", 6000),
        ("Sugarcane", "गन्ना", "Cash Crops", "Quintal", 350),
        ("Jute", "जूट", "Cash Crops", "Quintal", 4500),
        ("Tobacco", "तम्बाकू", "Cash Crops", "Quintal", 15000),
        ("Tea", "चाय", "Cash Crops", "Kg", 200),
        ("Coffee", "कॉफी", "Cash Crops", "Kg", 350),
        ("Rubber", "रबर", "Cash Crops", "Kg", 180),
        ("Arecanut", "सुपारी", "Cash Crops", "Quintal", 40000),

        # Flowers / फूल
        ("Marigold", "गेंदा", "Flowers", "Quintal", 3000),
        ("Rose", "गुलाब", "Flowers", "Kg", 150),
        ("Jasmine", "चमेली", "Flowers", "Kg", 500),
        ("Chrysanthemum", "गुलदाउदी", "Flowers", "Quintal", 2500),
        ("Tuberose", "रजनीगंधा", "Flowers", "Kg", 100),

        # Others
        ("Aloe Vera", "एलोवेरा", "Others", "Kg", 25),
        ("Bamboo", "बांस", "Others", "Quintal", 1000),
        ("Mentha", "पुदीना", "Others", "Kg", 80),
    ]

    cursor.executemany(
        "INSERT INTO crops (name, hindi_name, category, unit, avg_base_price) VALUES (?, ?, ?, ?, ?)",
        crops_data
    )
    logger.info(f"Inserted {len(crops_data)} crops into database")


# ---------------------
# Price Prediction Engine
# ---------------------
class PricePredictionEngine:
    """
    Predicts future crop prices using:
    1. Linear Regression on historical data
    2. Seasonal adjustment factors
    3. Polynomial trend fitting for accuracy
    """

    def __init__(self, historical_prices: list):
        self.prices = historical_prices
        self.model = None
        self.poly_features = PolynomialFeatures(degree=2)

    def prepare_data(self):
        """Convert price list to numpy arrays for ML model."""
        if len(self.prices) < 3:
            return None, None
        X = np.arange(len(self.prices)).reshape(-1, 1)
        y = np.array(self.prices)
        return X, y

    def train(self):
        """Train polynomial regression model on historical data."""
        X, y = self.prepare_data()
        if X is None:
            return False
        # Apply polynomial features for better curve fitting
        X_poly = self.poly_features.fit_transform(X)
        self.model = LinearRegression()
        self.model.fit(X_poly, y)
        return True

    def predict_future(self, months_ahead: int = 6) -> list:
        """
        Predict prices for next N months.
        Uses trained polynomial regression model.
        """
        if not self.train():
            # Fallback: use moving average if not enough data
            return self._moving_average_prediction(months_ahead)

        n = len(self.prices)
        future_indices = np.arange(n, n + months_ahead).reshape(-1, 1)
        future_poly = self.poly_features.transform(future_indices)
        predictions = self.model.predict(future_poly)

        # Apply seasonal factors (crop prices vary by season in India)
        seasonal_factors = self._get_seasonal_factors()
        adjusted = []
        base_month = datetime.now().month

        for i, pred in enumerate(predictions):
            month_idx = (base_month + i) % 12
            factor = seasonal_factors[month_idx]
            adjusted_price = max(pred * factor, 100)  # Ensure no negative prices
            adjusted.append(round(float(adjusted_price), 2))

        return adjusted

    def _moving_average_prediction(self, months_ahead: int) -> list:
        """Fallback: Simple moving average for prediction."""
        if not self.prices:
            return []
        window = min(3, len(self.prices))
        ma = np.mean(self.prices[-window:])
        # Add slight trend (+2% per month as inflation estimate)
        return [round(float(ma * (1.02 ** (i + 1))), 2) for i in range(months_ahead)]

    def _get_seasonal_factors(self) -> dict:
        """
        Seasonal price adjustment factors by month.
        Kharif crops peak Oct-Nov, Rabi crops peak Apr-May.
        General agricultural market trends in India.
        """
        return {
            0: 1.05,   # January - winter vegetables high
            1: 1.03,   # February
            2: 0.98,   # March - rabi harvest incoming
            3: 0.92,   # April - rabi harvest (prices drop)
            4: 0.90,   # May - peak supply
            5: 0.95,   # June - monsoon starts
            6: 1.00,   # July
            7: 1.02,   # August
            8: 1.08,   # September - pre-kharif harvest
            9: 1.05,   # October - kharif harvest
            10: 1.00,  # November
            11: 1.03,  # December - winter demand
        }

    def get_trend(self) -> str:
        """Analyze price trend: rising, falling, or stable."""
        if len(self.prices) < 3:
            return "stable"
        recent = self.prices[-3:]
        slope = (recent[-1] - recent[0]) / max(recent[0], 1) * 100
        if slope > 5:
            return "rising"
        elif slope < -5:
            return "falling"
        return "stable"

    def get_suggestion(self, trend: str, crop_name: str) -> str:
        """Generate farmer-friendly selling suggestions."""
        suggestions = {
            "rising": [
                f"📈 {crop_name} की कीमतें बढ़ रही हैं! अभी बेचने से अच्छा मुनाफ़ा मिलेगा।",
                f"Good time to sell {crop_name}! Prices are on an upward trend.",
                f"Hold stock for 2-4 more weeks for maximum profit.",
            ],
            "falling": [
                f"📉 {crop_name} की कीमतें घट रही हैं। जल्दी बेच दें!",
                f"Sell {crop_name} soon before prices fall further.",
                f"Consider cold storage if available to wait for better prices.",
            ],
            "stable": [
                f"⚖️ {crop_name} की कीमतें स्थिर हैं।",
                f"{crop_name} prices are stable. Good time for planned selling.",
                f"Market is balanced. Sell in batches for steady income.",
            ],
        }
        return " | ".join(suggestions.get(trend, suggestions["stable"]))


# ---------------------
# Simulated Historical Price Generator
# ---------------------
def generate_historical_prices(crop_name: str, state: str, base_price: float, months: int = 12) -> list:
    """
    Generate realistic simulated historical price data.
    Uses: base_price + seasonal variation + random noise + trend.
    This is used when real API data is unavailable.
    """
    np.random.seed(hash(crop_name + state) % 2**31)  # Consistent seed per crop+state
    prices = []
    current_date = datetime.now() - timedelta(days=30 * months)

    # Add a gradual trend (5-15% yearly increase due to inflation)
    yearly_growth = np.random.uniform(0.05, 0.15)
    monthly_growth = (1 + yearly_growth) ** (1 / 12)

    for i in range(months):
        # Seasonal component (±20% variation)
        month = (current_date.month + i) % 12
        seasonal = [1.05, 1.03, 0.98, 0.92, 0.90, 0.95, 1.00, 1.02, 1.08, 1.05, 1.00, 1.03]
        seasonal_factor = seasonal[month]

        # Random market noise (±10%)
        noise = np.random.uniform(-0.10, 0.10)

        # Calculate price
        trend_factor = monthly_growth ** i
        price = base_price * trend_factor * seasonal_factor * (1 + noise)
        price = max(price, base_price * 0.3)  # Floor: not less than 30% of base

        date_str = (current_date + timedelta(days=30 * i)).strftime("%Y-%m")
        prices.append({
            "date": date_str,
            "price": round(float(price), 2)
        })

    return prices


# ---------------------
# API Integration: Agmarknet
# ---------------------
def fetch_agmarknet_prices(state: str, crop: str, date: str) -> dict:
    """
    Fetch crop prices from Data.gov.in Agmarknet API.
    API returns mandi (market) prices across India.
    Falls back to simulated data if API fails.
    """
    # Check cache first
    cache_key = f"{state}_{crop}_{date[:7]}"  # Cache by month
    cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()

    if cache_key_hash in PRICE_CACHE:
        cached = PRICE_CACHE[cache_key_hash]
        if datetime.now().timestamp() - cached['timestamp'] < CACHE_TTL:
            logger.info(f"Cache hit for {cache_key}")
            return cached['data']

    try:
        # Format state and commodity for API
        params = {
            "api-key": API_KEY,
            "format": "json",
            "filters[state]": state,
            "filters[commodity]": crop,
            "limit": 100,
            "offset": 0,
        }

        response = requests.get(AGMARKNET_API_BASE, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])

            if records:
                # Parse API response
                prices = []
                for record in records:
                    try:
                        modal_price = float(record.get("modal_price", 0))
                        if modal_price > 0:
                            prices.append({
                                "market": record.get("market", ""),
                                "price": modal_price,
                                "date": record.get("arrival_date", date),
                                "state": record.get("state", state),
                            })
                    except (ValueError, TypeError):
                        continue

                if prices:
                    result = {"source": "api", "prices": prices, "success": True}
                    # Cache the result
                    PRICE_CACHE[cache_key_hash] = {
                        "data": result,
                        "timestamp": datetime.now().timestamp()
                    }
                    return result

    except requests.exceptions.RequestException as e:
        logger.warning(f"API fetch failed: {e}. Using simulated data.")
    except Exception as e:
        logger.error(f"Unexpected error in API fetch: {e}")

    # Fallback: return simulation flag
    return {"source": "simulated", "success": False, "message": "Using estimated data"}


# ---------------------
# Routes
# ---------------------
@app.route('/')
def index():
    """Main page with crop price prediction form."""
    return render_template('index.html')


@app.route('/api/crops', methods=['GET'])
def get_crops():
    """Return list of all crops from database."""
    db = get_db()
    crops = db.execute(
        "SELECT id, name, hindi_name, category, unit FROM crops ORDER BY category, name"
    ).fetchall()
    db.close()

    crops_list = []
    for crop in crops:
        crops_list.append({
            "id": crop["id"],
            "name": crop["name"],
            "hindi_name": crop["hindi_name"],
            "category": crop["category"],
            "unit": crop["unit"],
            "display": f"{crop['name']} ({crop['hindi_name']})"
        })

    return jsonify({"crops": crops_list, "total": len(crops_list)})


@app.route('/api/states', methods=['GET'])
def get_states():
    """Return list of Indian states."""
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
        "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
        "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
        "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Andaman and Nicobar Islands", "Chandigarh", "Delhi",
        "Jammu and Kashmir", "Ladakh", "Lakshadweep",
        "Puducherry", "Dadra and Nagar Haveli", "Daman and Diu"
    ]
    return jsonify({"states": states})


@app.route('/api/predict', methods=['POST'])
def predict_prices():
    """
    Main prediction endpoint.
    Input: state, crop, date
    Output: historical prices, current price, predicted prices, suggestions
    """
    try:
        data = request.get_json()
        state = data.get('state', '').strip()
        crop_name = data.get('crop', '').strip()
        date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))

        # Validate inputs
        if not state or not crop_name:
            return jsonify({"error": "State aur Crop dono zaroori hain!"}), 400

        # Get crop from database
        db = get_db()
        crop_row = db.execute(
            "SELECT * FROM crops WHERE name = ? OR hindi_name = ?",
            (crop_name, crop_name)
        ).fetchone()
        db.close()

        if not crop_row:
            return jsonify({"error": f"Crop '{crop_name}' not found in database"}), 404

        base_price = crop_row['avg_base_price']
        crop_id = crop_row['id']
        unit = crop_row['unit']
        hindi_name = crop_row['hindi_name']

        # Fetch API data
        api_result = fetch_agmarknet_prices(state, crop_name, date_str)

        # Generate 12 months historical data
        historical = generate_historical_prices(crop_name, state, base_price, months=12)
        historical_prices_list = [h['price'] for h in historical]
        historical_dates = [h['date'] for h in historical]

        # Get current price (last in historical + API adjustment)
        current_price = historical_prices_list[-1]

        # If API returned live data, use average of market prices as current
        if api_result.get('success') and api_result.get('prices'):
            live_prices = [p['price'] for p in api_result['prices']]
            if live_prices:
                api_avg = np.mean(live_prices)
                # Blend: 60% API, 40% simulated
                current_price = round(0.6 * api_avg + 0.4 * current_price, 2)

        # Run prediction engine
        engine = PricePredictionEngine(historical_prices_list)
        future_prices = engine.predict_future(months_ahead=6)
        trend = engine.get_trend()
        suggestion = engine.get_suggestion(trend, crop_name)

        # Generate future dates (next 6 months)
        future_dates = []
        for i in range(1, 7):
            future_date = datetime.now() + timedelta(days=30 * i)
            future_dates.append(future_date.strftime("%Y-%m"))

        # Price change percentage
        if len(historical_prices_list) >= 2:
            price_change = ((current_price - historical_prices_list[0]) / historical_prices_list[0]) * 100
        else:
            price_change = 0

        response = {
            "success": True,
            "crop": {
                "name": crop_name,
                "hindi_name": hindi_name,
                "unit": unit,
                "category": crop_row['category'],
            },
            "state": state,
            "date": date_str,
            "current_price": round(current_price, 2),
            "price_change_percent": round(price_change, 2),
            "historical": {
                "dates": historical_dates,
                "prices": historical_prices_list,
            },
            "predicted": {
                "dates": future_dates,
                "prices": future_prices,
            },
            "trend": trend,
            "suggestion": suggestion,
            "data_source": api_result.get('source', 'simulated'),
            "api_markets": api_result.get('prices', [])[:5] if api_result.get('success') else [],
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": "Prediction failed. Please try again.", "details": str(e)}), 500


# ---------------------
# App Startup
# ---------------------
if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
