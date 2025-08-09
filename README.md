# 🏏 T20 Score Predictor

A sophisticated machine learning web application that predicts final scores in T20 cricket matches using advanced cricket-aware features and real-time match situations.

![T20 Score Predictor](https://img.shields.io/badge/T20-Score%20Predictor-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-2.0+-red)
![ML](https://img.shields.io/badge/ML-Cricket%20Aware-orange)

## 🖥️ Application Preview

![T20 Score Predictor Interface](assets/screenshot.png)

*Modern, responsive web interface with real-time predictions and cricket intelligence*

## ✨ Features

### 🎯 **Intelligent Predictions**
- **Cricket-Aware Algorithm**: Advanced machine learning model with 35+ cricket-specific features
- **Real-Time Analysis**: Considers current match situation, momentum, and game phases
- **Multi-Factor Assessment**: Analyzes batting form, wicket pressure, and strategic positioning

### 📊 **Comprehensive Match Analysis**
- **Game Phase Recognition**: Powerplay, middle overs, death overs, and super death detection
- **Momentum Tracking**: Recent performance analysis over last 5 overs
- **Pressure Indicators**: Wicket pressure and time pressure calculations
- **Strategic Insights**: Batting depth assessment and resource management

### 🎨 **Modern Web Interface**
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Real-Time Validation**: Instant feedback on input validity
- **Interactive UI**: Smooth animations and hover effects
- **Cricket Context**: Phase-specific tooltips and guidance

### 🔍 **Advanced Features**
- **Cross-Validation**: Ensures data consistency and realistic scenarios
- **Cricket Intelligence**: Game-specific insights and recommendations
- **API Support**: RESTful endpoints for integration with other applications
- **Detailed Analytics**: Comprehensive breakdown of prediction factors

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/t20-score-predictor.git
   cd t20-score-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your model**
   - Place your trained `score_predictor.pkl` file in the root directory
   - Or train a new model using the provided features

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   ```
   http://localhost:5000
   ```

## 📋 Requirements

Create a `requirements.txt` file with:

```txt
Flask==2.3.3
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
```

## 🎮 Usage

### Web Interface

1. **Select Teams**: Choose batting and bowling teams from international cricket teams
2. **Choose Venue**: Select the city where the match is being played
3. **Current Situation**: Input current score, balls left, and wickets remaining
4. **Recent Form**: Enter run rate and runs scored in last 5 overs
5. **Get Prediction**: Click "Predict Final Score" for intelligent analysis

### API Usage

**Endpoint**: `POST /api/predict`

**Request Body**:
```json
{
  "batting_team": "India",
  "bowling_team": "Australia",
  "city": "Mumbai",
  "current_score": 120,
  "balls_left": 60,
  "wickets_left": 6,
  "run_rate": 8.5,
  "last_five_run": 45,
  "last_five_over_wicket": 1
}
```

**Response**:
```json
{
  "predicted_score": 165,
  "insights": [
    "Need 45 runs from 10.0 overs (Required RR: 4.50)",
    "🔥 Death overs - time for big hitting and calculated risks!",
    "⚖️ Balanced situation - need to manage wickets carefully"
  ],
  "cricket_intelligence": {
    "game_phase": "death_overs",
    "wicket_situation": "moderate",
    "recent_form": "accelerating",
    "pressure_level": "medium"
  }
}
```

## 🧠 Cricket-Aware Features

Our model uses sophisticated cricket intelligence with these key features:

### **Behavioral Analysis**
- `recent_wicket_pressure`: Adjustment for recent wicket losses
- `wicket_adjusted_aggression`: Modified aggression based on wicket situation
- `collapse_indicator`: Detection of potential batting collapses

### **Game Phase Intelligence**
- `powerplay`, `middle_overs`, `death_overs`: Phase-specific behavior
- `death_overs_aggression`: Enhanced aggression in final overs
- `phase_expected_rr`: Realistic targets for each phase

### **Strategic Assessment**
- `resources_remaining`: DLS-style resource calculation
- `scoring_potential`: Expected runs based on wickets and phase
- `contextual_aggression`: Multi-factor aggression index

### **Momentum & Form**
- `momentum_score`: Recent performance considering runs and wickets
- `form_stability`: Stability assessment based on recent wickets
- `performance_vs_expected`: Comparison with phase expectations

## 🏗️ Architecture

```
t20-score-predictor/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface (if using templates)
├── static/               # CSS, JS, images (if separated)
├── assets/               # Screenshots and documentation images
│   └── screenshot.png    # Application interface preview
├── score_predictor.pkl   # Trained ML model
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🎯 Model Performance

The model incorporates:
- **35+ Cricket-Specific Features**
- **Multi-Phase Game Understanding**
- **Behavioral Pattern Recognition**
- **Realistic Constraint Validation**
- **Cross-Situational Analysis**

## 🌟 Supported Teams & Venues

### **International Teams**
Australia, Bangladesh, England, India, Ireland, New Zealand, Pakistan, Scotland, South Africa, Sri Lanka, West Indies, Zimbabwe

### **Major Venues**
60+ international cricket venues including:
- **Asia**: Mumbai, Delhi, Kolkata, Dhaka, Colombo, Dubai, Abu Dhabi
- **Australia/NZ**: Melbourne, Sydney, Adelaide, Auckland, Wellington
- **England**: London, Manchester, Birmingham, Cardiff
- **Caribbean**: Barbados, Trinidad, Kingston, St Lucia
- **Africa**: Cape Town, Johannesburg, Durban, Harare

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔧 Customization

### Adding New Teams or Venues
Modify the `teams` and `cities` lists in `app.py`:

```python
teams = ['Your Team', ...existing teams]
cities = ['Your City', ...existing cities]
```

### Enhancing Features
Add new cricket-aware features in the `create_cricket_aware_features()` function:

```python
def create_cricket_aware_features(df):
    # Your custom features here
    df['your_custom_feature'] = your_calculation
    return df
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Cricket data providers and statisticians
- T20 cricket leagues for inspiration
- Machine learning community for algorithms and techniques
- Flask community for the excellent web framework

## 📞 Support

For support, email me at satyam2610saurabh@gmail.com or create an issue on GitHub.

---

*Disclaimer: Predictions are based on historical data and current match situations. Actual cricket matches may vary due to various unpredictable factors.*
