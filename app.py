from flask import Flask, render_template, request, jsonify, flash
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load model
try:
    model = joblib.load("score_predictor.pkl")
except FileNotFoundError:
    print("Model file not found. Please ensure score_predictor.pkl is in the correct directory.")
    model = None

# Teams and cities for dropdown
teams = ['Australia', 'India', 'South Africa', 'New Zealand', 'Sri Lanka',
       'West Indies', 'Ireland', 'Pakistan', 'Bangladesh', 'England',
       'Scotland', 'Zimbabwe']

cities = ['Cape Town', 'Lauderhill', 'Cardiff', 'Wellington', 'St Lucia',
       'Mount Maunganui', 'Bridgetown', 'Mirpur', 'Auckland', 'Colombo',
       'Pallekele', 'Tarouba', 'Trinidad', 'Durban', 'Dhaka', 'Dambulla',
       'Kingston', 'Barbados', 'Johannesburg', 'Hamilton', 'Chittagong',
       'Delhi', 'Harare', 'Birmingham', 'Edinburgh', 'Abu Dhabi',
       'Melbourne', 'Mumbai', 'Manchester', 'Adelaide', 'Perth', 'Sydney',
       'Kolkata', 'Dubai', 'Centurion', 'Dublin', 'Karachi', 'Hobart',
       'Southampton', 'Chattogram', 'Lahore', 'Hambantota', 'Kandy',
       'Chandigarh', 'London', 'Providence', 'Nottingham', 'Ahmedabad',
       'Gros Islet', 'Christchurch', 'Chester-le-Street', 'Rajkot',
       'Brisbane', 'Pune', "St George's", 'Guyana', 'Napier', 'Sharjah',
       'Belfast', 'Basseterre', 'Nagpur', 'Bangalore', 'Bristol']

def create_cricket_aware_features(df):
    """Create cricket-aware features that match the training model exactly"""
    df = df.copy()
    
    # Ensure no division by zero
    df['wickets_lost'] = 10 - df['wickets_left']
    df['overs_left'] = df['balls_left'] / 6
    df['overs_bowled'] = (120 - df['balls_left']) / 6
    
    # === BASIC ENHANCED FEATURES ===
    
    # Your original features with improvements
    df['score_per_wicket'] = df['current_score'] / (df['wickets_lost'] + 1)
    df['aggression_index'] = df['last_five_run'] / (df['balls_left'] + 1)
    df['pressure_index'] = df['balls_left'] / (df['wickets_left'] + 1)
    
    # === WICKET-BASED BEHAVIORAL FEATURES ===
    
    # Recent wicket pressure - teams become cautious after losing wickets
    df['recent_wicket_pressure'] = np.where(
        df['last_five_over_wicket'] >= 2,  # 2+ wickets in last 5 overs
        2.0,  # High pressure
        np.where(
            df['last_five_over_wicket'] == 1,
            1.5,  # Medium pressure  
            1.0   # No recent pressure
        )
    )
    
    # Wicket-adjusted aggression - teams play more cautiously after losing wickets
    df['wicket_adjusted_aggression'] = df['aggression_index'] / df['recent_wicket_pressure']
    
    # Collapse indicator - multiple wickets recently suggests a batting collapse
    df['collapse_indicator'] = (df['last_five_over_wicket'] >= 3).astype(int)
    
    # === GAME PHASE AWARENESS ===
    
    # Define clear game phases
    df['powerplay'] = (df['balls_left'] >= 84).astype(int)  # First 6 overs
    df['middle_overs'] = ((df['balls_left'] < 84) & (df['balls_left'] > 36)).astype(int)  # 7-14 overs
    df['death_overs'] = (df['balls_left'] <= 36).astype(int)  # Last 6 overs
    df['super_death'] = (df['balls_left'] <= 18).astype(int)  # Last 3 overs
    
    # === DEATH OVERS AGGRESSION ===
    
    # In death overs, teams try to maximize runs regardless of wickets
    df['death_overs_aggression'] = np.where(
        df['death_overs'] == 1,
        df['aggression_index'] * 1.5,  # Boost aggression in death overs
        df['aggression_index']
    )
    
    # Super death multiplier - even more aggressive in last 3 overs
    df['death_overs_aggression'] = np.where(
        df['super_death'] == 1,
        df['death_overs_aggression'] * 1.3,
        df['death_overs_aggression']
    )
    
    # === WICKET-IN-HAND STRATEGY ===
    
    # Teams with more wickets can afford to be more aggressive
    df['wickets_in_hand_ratio'] = df['wickets_left'] / 10
    df['wicket_buffer'] = np.where(
        df['wickets_left'] >= 7, 'plenty',    # 7+ wickets - can be aggressive
        np.where(df['wickets_left'] >= 4, 'moderate',  # 4-6 wickets - balanced
                np.where(df['wickets_left'] >= 2, 'few',      # 2-3 wickets - cautious
                        'critical'))                            # 0-1 wickets - very cautious
    )
    
    # Aggression modifier based on wickets in hand
    wicket_aggression_map = {'plenty': 1.2, 'moderate': 1.0, 'few': 0.8, 'critical': 0.6}
    df['wicket_aggression_modifier'] = df['wicket_buffer'].map(wicket_aggression_map)
    
    # === CONTEXTUAL AGGRESSION ===
    
    # Combine all factors for realistic aggression
    df['contextual_aggression'] = (
        df['aggression_index'] * 
        df['wicket_aggression_modifier'] * 
        (1 / df['recent_wicket_pressure']) *
        (1 + df['death_overs'] * 0.5)  # Extra boost in death overs
    )
    
    # === MOMENTUM AND FORM ===
    
    # Recent form considering both runs and wickets
    df['last_five_run_rate'] = df['last_five_run'] * 6 / 30  # Convert to run rate
    df['form_stability'] = np.where(
        df['last_five_over_wicket'] == 0,
        1.2,  # Stable - no wickets lost
        np.where(
            df['last_five_over_wicket'] == 1,
            1.0,  # Normal
            0.7   # Unstable - multiple wickets
        )
    )
    
    df['momentum_score'] = df['last_five_run_rate'] * df['form_stability']
    
    # === PRESSURE AND SITUATION ===
    
    # Time pressure increases as overs decrease
    df['time_pressure'] = 1 - (df['balls_left'] / 120)
    
    # Wicket pressure considering recent losses
    df['wicket_pressure'] = (
        (10 - df['wickets_left']) / 10 * 0.7 +  # Overall wickets lost
        (df['last_five_over_wicket'] / 5) * 0.3  # Recent wicket losses
    )
    
    # Combined pressure (higher means more pressure)
    df['combined_pressure'] = (df['time_pressure'] + df['wicket_pressure']) / 2
    
    # === STRATEGIC FEATURES ===
    
    # Remaining resources (DLS-style thinking)
    df['resources_remaining'] = (df['wickets_left'] * df['balls_left']) / (10 * 120)
    
    # Scoring potential based on situation
    df['scoring_potential'] = np.where(
        df['death_overs'] == 1,
        df['wickets_left'] * 8,  # In death overs, each wicket worth ~8 runs
        np.where(
            df['middle_overs'] == 1,
            df['wickets_left'] * 6,  # In middle overs, each wicket worth ~6 runs
            df['wickets_left'] * 4   # In powerplay, each wicket worth ~4 runs
        )
    )
    
    # === BATTING DEPTH ASSESSMENT ===
    
    # More nuanced batting position
    df['batting_situation'] = np.where(
        df['wickets_left'] >= 8, 'top_order_intact',
        np.where(df['wickets_left'] >= 6, 'top_order_batting',
                np.where(df['wickets_left'] >= 4, 'middle_order',
                        np.where(df['wickets_left'] >= 2, 'lower_middle',
                                'tail_enders')))
    )
    
    # === PHASE-SPECIFIC FEATURES ===
    
    # Expected run rate for each phase (realistic targets)
    df['phase_expected_rr'] = np.where(
        df['powerplay'] == 1, 7.5,   # Powerplay target
        np.where(df['middle_overs'] == 1, 6.5,  # Middle overs target
                np.where(df['death_overs'] == 1, 9.5,   # Death overs target
                        7.0))  # Default
    )
    
    # How actual performance compares to phase expectation
    df['performance_vs_expected'] = (df['last_five_run_rate'] / df['phase_expected_rr'])
    
    # === WICKET LOSS IMPACT ===
    
    # Expected final score reduction due to recent wickets
    df['wicket_loss_penalty'] = df['last_five_over_wicket'] * np.where(
        df['death_overs'] == 1, 8,   # Each wicket costs ~8 runs in death overs
        np.where(df['middle_overs'] == 1, 12,  # Each wicket costs ~12 runs in middle
                15)   # Each wicket costs ~15 runs in powerplay
    )
    
    # === FINAL SCORE INDICATORS ===
    
    # Projected run rate for remaining overs
    df['projected_rr'] = np.where(
        df['balls_left'] > 0,
        (df['momentum_score'] * df['wicket_aggression_modifier'] * 
         (1 + df['death_overs'] * 0.3)),
        0
    )
    
    # Conservative estimate (if team plays safe)
    df['conservative_projection'] = (
        df['current_score'] + 
        (df['balls_left'] / 6) * df['phase_expected_rr'] * 0.8
    )
    
    # Aggressive estimate (if team goes for big finish)
    df['aggressive_projection'] = (
        df['current_score'] + 
        (df['balls_left'] / 6) * df['projected_rr'] * 1.2
    )
    
    return df

def validate_input(data):
    """Validate input data for T20 cricket constraints"""
    errors = []
    
    # Check if batting and bowling teams are different
    if data.get('batting_team') == data.get('bowling_team'):
        errors.append("Batting and bowling teams cannot be the same")
    
    # Validate current score (0-400)
    current_score = data.get('current_score', 0)
    if not (0 <= current_score <= 400):
        errors.append("Current score must be between 0 and 400")
    
    # Validate balls left (0-120)
    balls_left = data.get('balls_left', 0)
    if not (0 <= balls_left <= 120):
        errors.append("Balls left must be between 0 and 120")
    
    # Validate wickets left (0-10)
    wickets_left = data.get('wickets_left', 0)
    if not (0 <= wickets_left <= 10):
        errors.append("Wickets left must be between 0 and 10")
    
    # Validate run rate (0-36, theoretical max is 36 runs per over)
    run_rate = data.get('run_rate', 0)
    if not (0 <= run_rate <= 36):
        errors.append("Run rate must be between 0 and 36")
    
    # Validate last 5 overs runs (0-180, theoretical max is 36*5)
    last_five_run = data.get('last_five_run', 0)
    if not (0 <= last_five_run <= 180):
        errors.append("Runs in last 5 overs must be between 0 and 180")
    
    # Validate last 5 overs wickets (0-10)
    last_five_over_wicket = data.get('last_five_over_wicket', 0)
    if not (0 <= last_five_over_wicket <= 10):
        errors.append("Wickets in last 5 overs must be between 0 and 10")
    
    # Cross-validation: Check if run rate is consistent with current score and balls faced
    balls_faced = 120 - balls_left
    if balls_faced > 0:
        overs_faced = balls_faced / 6
        if overs_faced > 0:
            calculated_run_rate = current_score / overs_faced
            # Allow some tolerance for rounding
            if abs(run_rate - calculated_run_rate) > 3:
                errors.append(f"Run rate ({run_rate:.2f}) seems inconsistent with current score and balls faced. Expected around {calculated_run_rate:.2f}")
    
    # Validate that current score is reasonable for the stage of the game
    if balls_left < 120:  # Game has started
        balls_faced = 120 - balls_left
        overs_faced = balls_faced / 6
        
        # Check if score is too high for early overs
        if overs_faced <= 6 and current_score > 120:  # Powerplay
            errors.append("Current score seems unusually high for the powerplay period")
        
        # Check if score is too low for late overs (unless many wickets lost)
        if overs_faced >= 15 and current_score < 80 and wickets_left >= 6:
            errors.append("Current score seems low for this stage of the innings")
    
    # Validate wicket losses in last 5 overs
    if last_five_over_wicket > (10 - wickets_left):
        errors.append("Wickets lost in last 5 overs cannot exceed total wickets lost")
    
    return errors

def get_prediction_insights(form_data, prediction):
    insights = []
    return insights 

@app.route('/')
def home():
    return render_template('index.html', teams=teams, cities=cities)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 500
        
        # Get form data - Updated to match training features
        form_data = {
            'batting_team': request.form.get('batting_team'),
            'bowling_team': request.form.get('bowling_team'),
            'city': request.form.get('city'),
            'current_score': int(request.form.get('current_score', 0)),
            'balls_left': int(request.form.get('balls_left', 0)),
            'wickets_left': int(request.form.get('wickets_left', 0)),
            'run_rate': float(request.form.get('run_rate', 0)),
            'last_five_run': float(request.form.get('last_five_run', 0)),  # Updated field name
            'last_five_over_wicket': int(request.form.get('last_five_over_wicket', 0))  # New field
        }
        
        # Validate input
        validation_errors = validate_input(form_data)
        if validation_errors:
            return render_template('index.html', 
                                 errors=validation_errors,
                                 teams=teams, 
                                 cities=cities,
                                 form_data=form_data)
        
        # Create input dataframe for prediction
        input_df = pd.DataFrame([form_data])
        
        # Add cricket-aware features that match your training exactly
        input_df_with_features = create_cricket_aware_features(input_df)
        
        # Select the exact features that your model expects (from your training code)
        feature_cols = [
            'batting_team', 'bowling_team', 'city',
            'current_score', 'balls_left', 'wickets_left',
            'run_rate', 'last_five_run', 'last_five_over_wicket',
            
            # Original enhanced features
            'score_per_wicket', 'aggression_index', 'pressure_index',
            
            # Cricket-aware features
            'recent_wicket_pressure', 'wicket_adjusted_aggression', 'collapse_indicator',
            'powerplay', 'middle_overs', 'death_overs', 'super_death',
            'death_overs_aggression', 'wickets_in_hand_ratio', 'wicket_buffer',
            'contextual_aggression', 'momentum_score', 'form_stability',
            'combined_pressure', 'resources_remaining', 'scoring_potential',
            'batting_situation', 'performance_vs_expected', 'wicket_loss_penalty',
            'projected_rr', 'conservative_projection'
        ]
        
        # Make sure we only pass features that exist and the model expects
        available_features = [col for col in feature_cols if col in input_df_with_features.columns]
        model_input = input_df_with_features[available_features]
        
        # Make prediction
        prediction = model.predict(model_input)[0]
        
        # Ensure prediction is reasonable
        prediction = max(form_data['current_score'], int(prediction))
        prediction = min(prediction, 450)  # T20 maximum realistic score
        
        # Get comprehensive cricket insights
        insights = get_prediction_insights(form_data, prediction)
        
        return render_template('index.html', 
                             prediction_text=f"Predicted Final Score: {int(prediction)} runs",
                             insights=insights,
                             teams=teams, 
                             cities=cities,
                             form_data=form_data)
    
    except ValueError as e:
        error_msg = "Invalid input format. Please check your entries."
        return render_template('index.html', 
                             errors=[error_msg],
                             teams=teams, 
                             cities=cities)
    
    except Exception as e:
        error_msg = f"An error occurred during prediction: {str(e)}"
        return render_template('index.html', 
                             errors=[error_msg],
                             teams=teams, 
                             cities=cities)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON predictions with cricket-aware features"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Validate input
        validation_errors = validate_input(data)
        if validation_errors:
            return jsonify({'errors': validation_errors}), 400
        
        # Create input dataframe
        input_df = pd.DataFrame([data])
        
        # Add cricket-aware features
        input_df_with_features = create_cricket_aware_features(input_df)
        
        # Select features for model (same as training)
        feature_cols = [
            'batting_team', 'bowling_team', 'city',
            'current_score', 'balls_left', 'wickets_left',
            'run_rate', 'last_five_run', 'last_five_over_wicket',
            'score_per_wicket', 'aggression_index', 'pressure_index',
            'recent_wicket_pressure', 'wicket_adjusted_aggression', 'collapse_indicator',
            'powerplay', 'middle_overs', 'death_overs', 'super_death',
            'death_overs_aggression', 'wickets_in_hand_ratio', 'wicket_buffer',
            'contextual_aggression', 'momentum_score', 'form_stability',
            'combined_pressure', 'resources_remaining', 'scoring_potential',
            'batting_situation', 'performance_vs_expected', 'wicket_loss_penalty',
            'projected_rr', 'conservative_projection'
        ]
        
        available_features = [col for col in feature_cols if col in input_df_with_features.columns]
        model_input = input_df_with_features[available_features]
        
        # Make prediction
        prediction = model.predict(model_input)[0]
        prediction = max(data['current_score'], int(prediction))
        prediction = min(prediction, 450)
        
        # Get insights
        insights = get_prediction_insights(data, prediction)
        
        # Return comprehensive response
        response = {
            'predicted_score': int(prediction),
            'current_score': data['current_score'],
            'balls_left': data['balls_left'],
            'wickets_left': data['wickets_left'],
            'last_five_run': data['last_five_run'],
            'last_five_over_wicket': data['last_five_over_wicket'],
            'insights': insights,
            'cricket_intelligence': {
                'game_phase': 'powerplay' if data['balls_left'] >= 84 else 'death_overs' if data['balls_left'] <= 36 else 'middle_overs',
                'wicket_situation': 'plenty' if data['wickets_left'] >= 7 else 'moderate' if data['wickets_left'] >= 4 else 'critical',
                'recent_form': 'accelerating' if data.get('last_five_run', 0) > 50 else 'steady' if data.get('last_five_run', 0) > 35 else 'struggling',
                'pressure_level': 'high' if data.get('last_five_over_wicket', 0) >= 2 else 'medium' if data.get('last_five_over_wicket', 0) == 1 else 'low'
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)