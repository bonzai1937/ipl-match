from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class XGBoostCricketPredictor:
    def __init__(self):
        self.score_model = None
        self.over_model = None
        self.preprocessing_data = None
        self.data = None
        self.dynamic_batter_factors = {}
        self.dynamic_bowler_factors = {}
        self.load_models()
        self.load_data()
        self.calculate_dynamic_factors()  # NEW: Calculate factors for all players
    
    def load_models(self):
        """Load XGBoost models and preprocessor"""
        try:
            logger.info("Loading XGBoost models...")
            
            # Check if model files exist
            model_files = {
                'remaining_runs': 'xgboost_remaining_runs_model.json',
                'next_over': 'xgboost_next_over_model.json',
                'preprocessor': 'xgboost_preprocessor.pkl'
            }
            
            missing_files = []
            for name, file_path in model_files.items():
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                logger.error(f"Missing model files: {missing_files}")
                return
            
            # Load XGBoost models
            self.score_model = xgb.XGBRegressor()
            self.score_model.load_model("xgboost_remaining_runs_model.json")
            
            self.over_model = xgb.XGBRegressor()
            self.over_model.load_model("xgboost_next_over_model.json")
            
            # Load preprocessor
            self.preprocessing_data = joblib.load("xgboost_preprocessor.pkl")
            
            logger.info("✅ XGBoost models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            self.score_model = None
            self.over_model = None
            self.preprocessing_data = None
    
    def load_data(self):
        """Load training data for options"""
        try:
            possible_files = [
                'final_dataset_with_pitch.csv'
            ]
            
            for filename in possible_files:
                if os.path.exists(filename):
                    self.data = pd.read_csv(filename)
                    logger.info(f"✅ Training data loaded: {len(self.data)} rows")
                    return
            
            logger.warning("❌ No training data found - using static options")
            self.data = None
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.data = None
    
    def calculate_dynamic_factors(self):
        """Calculate and store dynamic factors for all players"""
        self.dynamic_batter_factors, self.dynamic_bowler_factors = self.calculate_dynamic_player_factors()
        logger.info(f"🎯 Dynamic factors ready: {len(self.dynamic_batter_factors)} batters, {len(self.dynamic_bowler_factors)} bowlers")

    def calculate_dynamic_player_factors(self):
        """🚀 Calculate factors for ALL players dynamically from dataset"""
        if self.data is None:
            return {}, {}
        
        try:
            logger.info("🔄 Calculating dynamic factors for ALL players...")
            
            # BATTER FACTORS - Based on actual performance
            batter_stats = self.data.groupby('batter').agg({
                'total_runs': 'sum',
                'batter_runs': 'mean',
                'strike_rate': 'mean',
                'ball': 'count'  # Number of balls faced
            }).reset_index()
            
            # Filter batters with minimum 50 balls faced for reliability
            batter_stats = batter_stats[batter_stats['ball'] >= 50]
            
            # Calculate batter performance score
            avg_strike_rate = batter_stats['strike_rate'].mean()
            avg_runs_per_ball = batter_stats['batter_runs'].mean()
            
            batter_factors = {}
            for _, row in batter_stats.iterrows():
                player = row['batter']
                
                # Performance relative to average
                sr_ratio = row['strike_rate'] / avg_strike_rate if avg_strike_rate > 0 else 1.0
                runs_ratio = row['batter_runs'] / avg_runs_per_ball if avg_runs_per_ball > 0 else 1.0
                
                # Combined performance score (70% strike rate, 30% runs per ball)
                performance_score = (sr_ratio * 0.7) + (runs_ratio * 0.3)
                
                # Convert to factor (range 0.6 to 1.4)
                if performance_score >= 1.3:
                    factor = 1.35  # Elite batters
                elif performance_score >= 1.15:
                    factor = 1.20  # Very good batters
                elif performance_score >= 1.05:
                    factor = 1.10  # Good batters
                elif performance_score >= 0.95:
                    factor = 1.00  # Average batters
                elif performance_score >= 0.85:
                    factor = 0.90  # Below average batters
                elif performance_score >= 0.75:
                    factor = 0.80  # Poor batters
                else:
                    factor = 0.70  # Very poor batters (likely bowlers)
                
                batter_factors[player] = factor
            
            # BOWLER FACTORS - Based on actual performance
            bowler_stats = self.data.groupby('bowler').agg({
                'total_runs': 'sum',
                'ball': 'count',
                'player_dismissed': lambda x: (x.notna()).sum()  # Wickets taken
            }).reset_index()
            
            # Filter bowlers with minimum 50 balls bowled
            bowler_stats = bowler_stats[bowler_stats['ball'] >= 50]
            
            # Calculate bowler performance
            bowler_stats['economy'] = (bowler_stats['total_runs'] * 6) / bowler_stats['ball']
            bowler_stats['wicket_rate'] = bowler_stats['player_dismissed'] / bowler_stats['ball'] * 100
            
            avg_economy = bowler_stats['economy'].mean()
            avg_wicket_rate = bowler_stats['wicket_rate'].mean()
            
            bowler_factors = {}
            for _, row in bowler_stats.iterrows():
                player = row['bowler']
                
                # Lower economy is better for bowlers
                economy_ratio = avg_economy / row['economy'] if row['economy'] > 0 else 1.0
                wicket_ratio = row['wicket_rate'] / avg_wicket_rate if avg_wicket_rate > 0 else 1.0
                
                # Combined bowling performance (60% economy, 40% wicket rate)
                bowling_performance = (economy_ratio * 0.6) + (wicket_ratio * 0.4)
                
                # Convert to factor (range 0.6 to 1.0 - good bowlers reduce scores)
                if bowling_performance >= 1.3:
                    factor = 0.65  # Elite bowlers
                elif bowling_performance >= 1.15:
                    factor = 0.72  # Very good bowlers
                elif bowling_performance >= 1.05:
                    factor = 0.78  # Good bowlers
                elif bowling_performance >= 0.95:
                    factor = 0.85  # Average bowlers
                elif bowling_performance >= 0.85:
                    factor = 0.90  # Below average bowlers
                else:
                    factor = 0.95  # Poor bowlers
                
                bowler_factors[player] = factor
            
            logger.info(f"✅ Generated factors for {len(batter_factors)} batters and {len(bowler_factors)} bowlers")
            
            # Log some examples
            top_batters = sorted(batter_factors.items(), key=lambda x: x[1], reverse=True)[:5]
            top_bowlers = sorted(bowler_factors.items(), key=lambda x: x[1])[:5]
            
            logger.info(f"🏏 Top batters: {top_batters}")
            logger.info(f"🎯 Top bowlers: {top_bowlers}")
            
            return batter_factors, bowler_factors
            
        except Exception as e:
            logger.error(f"Error calculating dynamic factors: {str(e)}")
            return {}, {}
    
    def get_dynamic_batter_factor(self, batter_name, dynamic_batter_factors):
        """Get batter factor - ALWAYS prioritize dynamic factors from dataset"""
        
        # 🚀 FIRST PRIORITY: Dynamic factors from YOUR dataset
        if batter_name in dynamic_batter_factors:
            factor = dynamic_batter_factors[batter_name]
            logger.info(f"📊 DATASET factor for {batter_name}: {factor:.3f}")
            return factor
        
        # 🚀 SECOND PRIORITY: Check if it's a known bowler (major penalty)
        known_bowlers = {
            'J Bumrah', 'B Kumar', 'T Boult', 'K Rabada', 'R Ashwin', 'Y Chahal',
            'A Nehra', 'AR Patel', 'P Kumar', 'Z Khan', 'I Sharma', 'Arshdeep Singh'
        }
        
        if batter_name in known_bowlers:
            factor = 0.65
            logger.info(f"🚨 Known bowler batting {batter_name}: {factor:.3f}")
            return factor
        
        # 🚀 LAST RESORT: Default for completely unknown players
        factor = 1.00
        logger.info(f"❓ Unknown player {batter_name}: {factor:.3f} (neutral default)")
        return factor
    
    def get_dynamic_bowler_factor(self, bowler_name, dynamic_bowler_factors):
        """Get bowler factor - dynamic if available, fallback to hardcoded"""
        
        # First try dynamic factors
        if bowler_name in dynamic_bowler_factors:
            factor = dynamic_bowler_factors[bowler_name]
            logger.info(f"📊 Dynamic bowler factor for {bowler_name}: {factor:.3f}")
            return factor
        
        # Fallback to hardcoded elite bowlers
        elite_bowlers = {
            'J Bumrah': 0.65, 'B Kumar': 0.72, 'T Boult': 0.68, 'K Rabada': 0.70,
            'R Ashwin': 0.76, 'Y Chahal': 0.78, 'A Nehra': 0.73, 'AR Patel': 0.76,
            'P Kumar': 0.75, 'Z Khan': 0.74, 'I Sharma': 0.80, 'Arshdeep Singh': 0.76
        }
        
        if bowler_name in elite_bowlers:
            factor = elite_bowlers[bowler_name]
            logger.info(f"🎯 Hardcoded elite bowler {bowler_name}: {factor:.3f}")
            return factor
        
        # Default for unknown bowlers
        factor = 0.88
        logger.info(f"❓ Unknown bowler {bowler_name}: {factor:.3f} (default)")
        return factor
    
    def get_options(self):
        """Get available options for frontend"""
        if self.data is not None:
            try:
                return {
                    'teams': sorted(self.data['batting_team'].unique().tolist()),
                    'batters': sorted(self.data['batter'].unique().tolist()),
                    'bowlers': sorted(self.data['bowler'].unique().tolist()),
                    'venues': sorted(self.data['venue'].unique().tolist()),
                    'pitch_types': sorted(self.data['pitch_type'].unique().tolist())
                }
            except Exception as e:
                logger.error(f"Error extracting options: {str(e)}")
        
        # Fallback static options
        return {
            'teams': [
                'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
                'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
                'Rajasthan Royals', 'Sunrisers Hyderabad', 'Lucknow Super Giants',
                'Gujarat Titans', 'Royal Challengers Bengaluru'
            ],
            'batters': [
                'V Kohli', 'AB de Villiers', 'MS Dhoni', 'R Sharma', 'KL Rahul',
                'S Dhawan', 'D Warner', 'S Iyer', 'R Pant', 'H Pandya',
                'A Nehra', 'SC Ganguly', 'RT Ponting', 'ML Hayden', 'CH Gayle',
                'JC Buttler', 'SK Raina'
            ],
            'bowlers': [
                'J Bumrah', 'B Kumar', 'T Boult', 'K Rabada', 'R Ashwin',
                'Y Chahal', 'Mohammed Shami', 'A Nehra', 'AR Patel', 'P Kumar',
                'Z Khan', 'I Sharma', 'S Sreesanth', 'Arshdeep Singh'
            ],
            'venues': [
                'M Chinnaswamy Stadium', 'Wankhede Stadium', 'Eden Gardens',
                'Feroz Shah Kotla', 'MA Chidambaram Stadium', 'Rajiv Gandhi International Stadium',
                'Sawai Mansingh Stadium', 'Punjab Cricket Association Stadium',
                'De Beers Diamond Oval', 'Narendra Modi Stadium', 'Barabati Stadium'
            ],
            'pitch_types': ['Unknown', 'Sluggish', 'Balanced', 'Batting-friendly', 'Spin-friendly']
        }
    
    def get_team_players(self, batting_team, bowling_team):
        """Get players for specific teams"""
        if self.data is None:
            return {'batters': [], 'bowlers': []}
        
        try:
            team_batters = []
            team_bowlers = []
            
            if batting_team:
                batting_data = self.data[self.data['batting_team'] == batting_team]
                if not batting_data.empty:
                    team_batters = sorted(batting_data['batter'].unique().tolist())
                    logger.info(f"✅ Found {len(team_batters)} batters for {batting_team}")
                else:
                    logger.warning(f"❌ No batters found for {batting_team}")
            
            if bowling_team:
                bowling_data = self.data[self.data['bowling_team'] == bowling_team]
                if not bowling_data.empty:
                    team_bowlers = sorted(bowling_data['bowler'].unique().tolist())
                    logger.info(f"✅ Found {len(team_bowlers)} bowlers for {bowling_team}")
                else:
                    logger.warning(f"❌ No bowlers found for {bowling_team}")
            
            return {'batters': team_batters, 'bowlers': team_bowlers}
            
        except Exception as e:
            logger.error(f"Error getting team players: {str(e)}")
            return {'batters': [], 'bowlers': []}
    
    def get_phase(self, over):
        """🚀 ENHANCED: Get match phase with death over sub-phases"""
        if over <= 6:
            return 'Powerplay'
        elif over <= 10:
            return 'Middle1'
        elif over <= 15:
            return 'Middle2'
        elif over <= 17:
            return 'Death_Early'  # Overs 16-17
        elif over <= 19:
            return 'Death_Middle'  # Overs 18-19
        else:
            return 'Death_Final'   # Over 20
    
    def get_death_over_boost(self, current_over, current_run_rate):
        """🚀 BALANCED: Calculate death over acceleration factor"""
        if current_over <= 15:
            return 1.0
        elif current_over == 16:
            return 1.20  # 25% boost in 16th over (reduced from 40%)
        elif current_over == 17:
            return 1.30  # 30% boost in 17th over (reduced from 50%)
        elif current_over == 18:
            return 1.35  # 35% boost in 18th over (reduced from 60%)
        elif current_over == 19:
            return 1.40  # 40% boost in 19th over (reduced from 70%)
        else:  # Over 20
            return 1.25  # 25% boost in final over (reduced from 45%)
    
    def get_wicket_penalty(self, current_over, current_wickets):
        """🚀 ENHANCED: More lenient wicket penalties in death overs"""
        if current_over <= 10:
            # Early overs - original penalties
            penalties = {
                0: 1.00, 1: 0.92, 2: 0.83, 3: 0.72, 4: 0.60,
                5: 0.48, 6: 0.36, 7: 0.25, 8: 0.16, 9: 0.10, 10: 0.06
            }
        elif current_over <= 15:
            # Middle overs - moderate penalties
            penalties = {
                0: 1.00, 1: 0.88, 2: 0.75, 3: 0.61, 4: 0.47,
                5: 0.34, 6: 0.23, 7: 0.15, 8: 0.09, 9: 0.05, 10: 0.03
            }
        else:
            # 🚀 DEATH OVERS - More balanced penalties (not too lenient)
            penalties = {
                0: 1.00, 1: 0.92, 2: 0.84, 3: 0.74, 4: 0.62,  # More balanced
                5: 0.48, 6: 0.35, 7: 0.24, 8: 0.16, 9: 0.10, 10: 0.07
            }
        
        return penalties.get(current_wickets, 0.05)
    
    def prepare_features(self, input_data):
        """Prepare features for XGBoost prediction"""
        try:
            df = pd.DataFrame([input_data])
            
            # Calculate derived features
            over = df['over'].iloc[0]
            cumulative_runs = df['cumulative_runs'].iloc[0]
            cumulative_wickets = df['cumulative_wickets'].iloc[0]
            
            # Add calculated features
            df['phase'] = df['over'].apply(self.get_phase)
            df['run_rate'] = cumulative_runs / (over + 0.1)
            df['wickets_remaining'] = 10 - cumulative_wickets
            df['overs_remaining'] = 20 - over
            df['resources_remaining'] = df['overs_remaining'] * (df['wickets_remaining'] / 10)
            df['is_death_over'] = (over >= 16).astype(int)
            
            # Set default values for missing features
            defaults = {
                'strike_rate': df['run_rate'] * 100,
                'batter_avg': 25.0,
                'bowler_avg': 8.0,
                'partnership_runs': cumulative_runs * 0.3,
                'partnership_balls': (over * 6) * 0.3,
                'bowler_balls_bowled': 12,
                'bowler_recent_economy': 8.0,
                'runs_last_5_overs': cumulative_runs * 0.4,
                'wickets_last_5_overs': cumulative_wickets * 0.4,
                'partnership_momentum': 6.0,
                'batter_last5': 25.0,
                'bowler_last5': 8.0,
                'balls_faced_current_over': 3,
                'runs_current_over': 6,
                'current_over_rate': 6.0,
                'phase_avg_runs': 8.0
            }
            
            for feature, value in defaults.items():
                df[feature] = value
            
            # Required features in correct order
            required_features = [
                'venue', 'batting_team', 'bowling_team', 'batter', 'bowler',
                'over', 'cumulative_runs', 'cumulative_wickets', 'phase', 'pitch_type',
                'run_rate', 'strike_rate', 'batter_avg', 'bowler_avg',
                'partnership_runs', 'partnership_balls', 'bowler_balls_bowled',
                'bowler_recent_economy', 'wickets_remaining', 'resources_remaining',
                'is_death_over', 'runs_last_5_overs', 'wickets_last_5_overs',
                'partnership_momentum', 'batter_last5', 'bowler_last5',
                'balls_faced_current_over', 'runs_current_over', 'current_over_rate',
                'phase_avg_runs', 'overs_remaining'
            ]
            
            return df[required_features]
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise e
    
    def preprocess_features(self, features_df):
        """Apply XGBoost preprocessing"""
        try:
            if self.preprocessing_data is None:
                raise Exception("Preprocessing data not loaded")
            
            label_encoders = self.preprocessing_data['label_encoders']
            scaler = self.preprocessing_data['scaler']
            categorical_features = self.preprocessing_data['categorical_features']
            numeric_features = self.preprocessing_data['numeric_features']
            
            X_encoded = features_df.copy()
            
            # Encode categorical features
            for col in categorical_features:
                if col in X_encoded.columns and col in label_encoders:
                    le = label_encoders[col]
                    X_encoded[col] = X_encoded[col].astype(str)
                    
                    # Handle unknown categories
                    unknown_mask = ~X_encoded[col].isin(le.classes_)
                    if unknown_mask.any():
                        X_encoded.loc[unknown_mask, col] = le.classes_[0]
                    
                    X_encoded[col] = le.transform(X_encoded[col])
            
            # Scale numeric features
            X_encoded[numeric_features] = scaler.transform(X_encoded[numeric_features])
            
            return X_encoded
            
        except Exception as e:
            logger.error(f"Error preprocessing: {str(e)}")
            raise e
    
    def predict_final_score_fallback(self, input_data):
        """🚀 ENHANCED: More aggressive fallback with death over acceleration"""
        logger.warning("🚨 Using ENHANCED fallback prediction")
        
        current_runs = input_data['cumulative_runs']
        current_over = input_data['over']
        wickets = input_data['cumulative_wickets']
        overs_left = 20 - current_over
        
        if overs_left <= 0:
            return current_runs
        
        current_run_rate = current_runs / current_over if current_over > 0 else 8
        
        # 🚀 ENHANCED: Much more aggressive phase adjustments
        phase = self.get_phase(current_over)
        if phase == 'Powerplay':
            # Early powerplay - much more aggressive projection
            if current_over <= 3:
                projected_rate = current_run_rate * 1.25  # 25% boost for very early
            else:
                projected_rate = current_run_rate * 1.15  # 15% boost for late powerplay
        elif phase == 'Middle1':
            projected_rate = current_run_rate * 1.05  # Slight boost
        elif phase == 'Middle2':
            projected_rate = current_run_rate * 1.02  # Slight boost
        elif phase == 'Death_Early':  # Overs 16-17
            projected_rate = max(current_run_rate * 1.45, 11.0)  # At least 11 RPO
        elif phase == 'Death_Middle':  # Overs 18-19
            projected_rate = max(current_run_rate * 1.65, 12.0)  # At least 12 RPO
        else:  # Death_Final - Over 20
            projected_rate = max(current_run_rate * 1.50, 13.0)  # At least 13 RPO
        
        # 🚀 ENHANCED: Use new wicket penalty function
        wicket_penalty = self.get_wicket_penalty(current_over, wickets)
        
        # 🚀 APPLY FACTORS HERE IN FALLBACK TOO!
        pitch_factors = {
            'Batting-friendly': 1.30, 'Balanced': 1.10, 'Unknown': 1.0,
            'Spin-friendly': 0.88, 'Sluggish': 0.80
        }
        pitch_factor = pitch_factors.get(input_data.get('pitch_type', 'Unknown'), 1.0)
        
        batter_name = input_data.get('batter', '')
        batter_factor = self.get_dynamic_batter_factor(batter_name, self.dynamic_batter_factors)
        
        # Calculate final score with factors
        enhanced_rate = projected_rate * wicket_penalty * pitch_factor * batter_factor
        projected_runs = enhanced_rate * overs_left
        estimated_final = current_runs + projected_runs
        
        # 🚀 ENHANCED: More realistic bounds for death overs
        if current_over >= 16:  # Death overs
            if wickets <= 3:
                min_final = current_runs + (overs_left * 8)
                max_final = current_runs + (overs_left * 15)
            elif wickets <= 5:
                min_final = current_runs + (overs_left * 7)
                max_final = current_runs + (overs_left * 13)
            else:
                min_final = current_runs + (overs_left * 6)
                max_final = current_runs + (overs_left * 11)
        elif current_over <= 6:  # Very early - be optimistic
            min_final = current_runs + (overs_left * 6)
            max_final = current_runs + (overs_left * 14)
        elif current_over <= 10:  # Early middle
            min_final = current_runs + (overs_left * 5)
            max_final = current_runs + (overs_left * 12)
        else:  # Later overs
            min_final = current_runs + (overs_left * 4)
            max_final = current_runs + (overs_left * 10)
        
        estimated_final = max(min_final, min(estimated_final, max_final))
        
        # 🚀 BALANCED: Higher caps for death overs but not excessive
        if current_over >= 16:  # Death overs
            if wickets >= 7:
                estimated_final = min(estimated_final, 190)  # Modest increase from 170
            elif wickets >= 5:
                estimated_final = min(estimated_final, 225)  # Modest increase from 200
            elif wickets >= 3:
                estimated_final = min(estimated_final, 255)  # Modest increase from 230
            elif wickets >= 1:
                estimated_final = min(estimated_final, 285)  # Modest increase from 260
            else:
                estimated_final = min(estimated_final, 310)  # Modest increase from 280
        else:  # Earlier overs
            if wickets >= 7:
                estimated_final = min(estimated_final, 170)
            elif wickets >= 5:
                estimated_final = min(estimated_final, 200)
            elif wickets >= 3:
                estimated_final = min(estimated_final, 230)
            elif wickets >= 1:
                estimated_final = min(estimated_final, 260)
            else:
                estimated_final = min(estimated_final, 280)
        
        logger.info(f"🎯 ENHANCED FALLBACK: {current_runs}/{wickets} in {current_over} → {estimated_final:.1f}")
        return estimated_final
    
    def predict_final_score(self, input_data):
        """🚀 ENHANCED: Now works for ALL players dynamically with death over fixes"""
        try:
            if self.score_model is None or self.preprocessing_data is None:
                logger.info("🎯 Models not loaded, using ENHANCED fallback")
                return self.predict_final_score_fallback(input_data)
            
            logger.info("🤖 Using XGBoost with ENHANCED DEATH OVER system for ALL players")
            
            # Prepare and preprocess features
            features_df = self.prepare_features(input_data)
            features_processed = self.preprocess_features(features_df)
            
            # Get XGBoost prediction (remaining runs)
            remaining_runs_pred = float(self.score_model.predict(features_processed)[0])
            
            current_runs = input_data['cumulative_runs']
            current_wickets = input_data['cumulative_wickets']
            current_over = input_data['over']
            current_run_rate = current_runs / current_over if current_over > 0 else 8
            
            # 🚀 ENHANCED FACTOR SYSTEM WITH DEATH OVER FIXES!
            
            # 1. ENHANCED PHASE BOOST (including better middle overs)
            if current_over <= 6:
                phase_boost = 1.20
            elif current_over <= 10:
                phase_boost = 1.15  # Increased slightly from 1.10
            elif current_over <= 15:
                phase_boost = 1.30  # INCREASED from 1.05 for better middle over predictions
            else:
                phase_boost = death_boost  # Use death over boost for 16+
            
            # 1.5. DEATH OVER BOOST (for overs 16+)
            death_boost = self.get_death_over_boost(current_over, current_run_rate)
            
            # 2. ENHANCED WICKET PENALTY
            wicket_penalty = self.get_wicket_penalty(current_over, current_wickets)
            
            # 3. CONTEXT BOOST - Balanced for death overs
            if current_over >= 16:  # Death overs
                if current_run_rate >= 9:
                    context_boost = 1.15  # Team is in great rhythm (reduced from 1.25)
                elif current_run_rate >= 7.5:
                    context_boost = 1.08  # Moderate pace (reduced from 1.15)
                else:
                    context_boost = 1.03  # Slow start but death overs boost (reduced from 1.08)
            else:
                context_boost = 1.0
            
            # 4. PITCH TYPE FACTORS
            pitch_factors = {
                'Batting-friendly': 1.35, 'Balanced': 1.15, 'Unknown': 1.0,
                'Spin-friendly': 0.85, 'Sluggish': 0.75
            }
            pitch_factor = pitch_factors.get(input_data.get('pitch_type', 'Unknown'), 1.0)
            
            # 5. TEAM FACTORS
            team_factors = {
                'Mumbai Indians': 1.12, 'Chennai Super Kings': 1.10, 'Royal Challengers Bangalore': 1.16,
                'Delhi Capitals': 1.08, 'Punjab Kings': 1.10, 'Rajasthan Royals': 1.06,
                'Kolkata Knight Riders': 1.04, 'Sunrisers Hyderabad': 0.90, 'Gujarat Titans': 1.08,
                'Lucknow Super Giants': 1.06, 'Royal Challengers Bengaluru': 1.16
            }
            team_factor = team_factors.get(input_data.get('batting_team', ''), 1.0)
            
            # 6. VENUE FACTORS
            venue_factors = {
                'M Chinnaswamy Stadium': 1.18, 'Wankhede Stadium': 1.15, 'Eden Gardens': 1.08,
                'Feroz Shah Kotla': 1.05, 'MA Chidambaram Stadium': 0.88, 'Rajiv Gandhi International Stadium': 1.10,
                'Sawai Mansingh Stadium': 1.08, 'Punjab Cricket Association Stadium': 1.12,
                'De Beers Diamond Oval': 1.06, 'Narendra Modi Stadium': 1.10, 'Barabati Stadium': 1.08
            }
            venue_factor = venue_factors.get(input_data.get('venue', ''), 1.0)
            
            # 🚀 7. DYNAMIC BATTER FACTORS - FOR ALL PLAYERS!
            batter_name = input_data.get('batter', '')
            batter_factor = self.get_dynamic_batter_factor(batter_name, self.dynamic_batter_factors)
            
            # 🚀 8. DYNAMIC BOWLER FACTORS - FOR ALL PLAYERS!
            bowler_name = input_data.get('bowler', '')
            bowler_factor = self.get_dynamic_bowler_factor(bowler_name, self.dynamic_bowler_factors)
            
            # 🚀 BALANCED DEATH OVER RUN RATE PROJECTION
            if current_over >= 16:
                # Death overs - expect higher run rates but more realistic
                if current_wickets <= 3:
                    projected_death_rate = max(current_run_rate * 1.20, 10.0)  # At least 10 RPO (reduced from 11.5)
                elif current_wickets <= 5:
                    projected_death_rate = max(current_run_rate * 1.12, 9.0)   # At least 9 RPO (reduced from 10)
                else:
                    projected_death_rate = max(current_run_rate * 1.05, 7.5)   # At least 7.5 RPO (reduced from 8.5)
                
                overs_left = 20 - current_over
                remaining_runs_death = projected_death_rate * overs_left
                
                # Use the higher of XGBoost prediction or death over projection
                remaining_runs_pred = max(remaining_runs_pred, remaining_runs_death)
            
            # 🚀 APPLY ALL FACTORS
            base_final_score = current_runs + remaining_runs_pred
            
            # Calculate total factor impact (use phase_boost instead of death_boost for all phases)
            if current_over >= 16:
                total_factor = (death_boost * wicket_penalty * pitch_factor * team_factor * 
                               venue_factor * batter_factor * bowler_factor * context_boost)
            else:
                total_factor = (phase_boost * wicket_penalty * pitch_factor * team_factor * 
                               venue_factor * batter_factor * bowler_factor * context_boost)
            
            # Apply factors to remaining runs
            adjusted_remaining = remaining_runs_pred * total_factor
            adjusted_final_score = current_runs + adjusted_remaining
            
            # Additional adjustments for extreme performers
            if batter_factor <= 0.70:  # Very poor batters (likely bowlers)
                adjusted_final_score *= 0.85
                logger.info(f"🚨 Penalty for poor batter: {batter_name}")
            elif batter_factor >= 1.20:  # Elite batters
                adjusted_final_score *= 1.08
                logger.info(f"⭐ Boost for elite batter: {batter_name}")
            
            if bowler_factor <= 0.70:  # Elite bowlers
                adjusted_final_score *= 0.95
                logger.info(f"🎯 Penalty for elite bowler: {bowler_name}")
            
            # Team and venue specific adjustments
            if input_data.get('batting_team') == 'Royal Challengers Bangalore':
                adjusted_final_score *= 1.08
            elif input_data.get('batting_team') == 'Sunrisers Hyderabad':
                adjusted_final_score *= 0.92
            
            if input_data.get('venue') == 'M Chinnaswamy Stadium':
                adjusted_final_score *= 1.12
            elif input_data.get('venue') == 'MA Chidambaram Stadium':
                adjusted_final_score *= 0.90
            
            # 🚀 BALANCED REALISTIC BOUNDS FOR DEATH OVERS
            overs_left = 20 - current_over
            
            if current_over >= 16:  # Death overs - balanced higher potential
                if current_wickets <= 2:
                    max_realistic = current_runs + (overs_left * 12)  # Up to 12 RPO (reduced from 15)
                    min_realistic = current_runs + (overs_left * 7)
                elif current_wickets <= 4:
                    max_realistic = current_runs + (overs_left * 11)  # Up to 11 RPO (reduced from 13)
                    min_realistic = current_runs + (overs_left * 6)
                elif current_wickets <= 6:
                    max_realistic = current_runs + (overs_left * 9)   # Up to 9 RPO (reduced from 11)
                    min_realistic = current_runs + (overs_left * 5)
                else:
                    max_realistic = current_runs + (overs_left * 8)   # Up to 8 RPO (reduced from 9)
                    min_realistic = current_runs + (overs_left * 4)
            elif current_over <= 6:
                if current_run_rate > 12:
                    max_realistic = current_runs + (overs_left * 12)
                    min_realistic = current_runs + (overs_left * 7)
                elif current_run_rate > 8:
                    max_realistic = current_runs + (overs_left * 11)
                    min_realistic = current_runs + (overs_left * 6)
                else:
                    max_realistic = current_runs + (overs_left * 9)
                    min_realistic = current_runs + (overs_left * 5)
            elif current_over <= 10:
                max_realistic = current_runs + (overs_left * 10)
                min_realistic = current_runs + (overs_left * 4)
            else:
                max_realistic = current_runs + (overs_left * 9)
                min_realistic = current_runs + (overs_left * 3)
            
            adjusted_final_score = max(min_realistic, min(adjusted_final_score, max_realistic))
            
            # 🚀 BALANCED FINAL CAPS - Moderate increases for death overs
            if current_over >= 16:  # Death overs
                if current_wickets >= 7:
                    adjusted_final_score = min(adjusted_final_score, 190)  # Modest increase from 180
                elif current_wickets >= 5:
                    adjusted_final_score = min(adjusted_final_score, 225)  # Modest increase from 220
                elif current_wickets >= 3:
                    adjusted_final_score = min(adjusted_final_score, 255)  # Modest increase from 250
                elif current_wickets >= 1:
                    adjusted_final_score = min(adjusted_final_score, 285)  # Modest increase from 280
                else:
                    adjusted_final_score = min(adjusted_final_score, 310)  # Modest increase from 300
            else:  # Earlier overs
                if current_wickets >= 7:
                    adjusted_final_score = min(adjusted_final_score, 180)
                elif current_wickets >= 5:
                    adjusted_final_score = min(adjusted_final_score, 220)
                elif current_wickets >= 3:
                    adjusted_final_score = min(adjusted_final_score, 250)
                elif current_wickets >= 1:
                    adjusted_final_score = min(adjusted_final_score, 280)
                else:
                    adjusted_final_score = min(adjusted_final_score, 300)
            
            logger.info(f"🎯 BALANCED DEATH OVER PREDICTION FOR ALL PLAYERS:")
            logger.info(f"   Current: {current_runs}/{current_wickets} in {current_over} overs")
            logger.info(f"   XGBoost remaining: {remaining_runs_pred:.1f}")
            logger.info(f"   Phase boost: {phase_boost:.3f}")
            logger.info(f"   Death boost: {death_boost:.3f}")
            logger.info(f"   Context boost: {context_boost:.3f}")
            logger.info(f"   Dynamic batter factor ({batter_name}): {batter_factor:.3f}")
            logger.info(f"   Dynamic bowler factor ({bowler_name}): {bowler_factor:.3f}")
            logger.info(f"   Wicket penalty: {wicket_penalty:.3f}")
            logger.info(f"   Pitch factor: {pitch_factor:.3f}")
            logger.info(f"   Total factor: {total_factor:.3f}")
            logger.info(f"   Final score: {float(adjusted_final_score):.1f}")
            
            return float(adjusted_final_score)
            
        except Exception as e:
            logger.error(f"❌ Enhanced prediction error: {str(e)}")
            return self.predict_final_score_fallback(input_data)


# Initialize predictor
predictor = XGBoostCricketPredictor()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'XGBoost Cricket Score Prediction API - Enhanced Death Over System',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': predictor.score_model is not None,
        'data_loaded': predictor.data is not None,
        'dynamic_batters': len(predictor.dynamic_batter_factors),
        'dynamic_bowlers': len(predictor.dynamic_bowler_factors),
        'version': '4.0.0'
    })

@app.route('/get-options', methods=['GET'])
def get_options():
    """Get all available options"""
    try:
        options = predictor.get_options()
        return jsonify({
            'success': True,
            'options': options
        })
    except Exception as e:
        logger.error(f"Error getting options: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get-team-players', methods=['POST'])
def get_team_players():
    """Get players for specific teams"""
    try:
        data = request.get_json()
        batting_team = data.get('batting_team', '')
        bowling_team = data.get('bowling_team', '')
        
        players = predictor.get_team_players(batting_team, bowling_team)
        
        return jsonify({
            'success': True,
            'batters': players['batters'],
            'bowlers': players['bowlers']
        })
        
    except Exception as e:
        logger.error(f"Error getting team players: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get-player-factors', methods=['GET'])
def get_player_factors():
    """NEW: Get dynamic factors for debugging/review"""
    try:
        # Get top and bottom performers
        top_batters = sorted(predictor.dynamic_batter_factors.items(), key=lambda x: x[1], reverse=True)[:10]
        bottom_batters = sorted(predictor.dynamic_batter_factors.items(), key=lambda x: x[1])[:10]
        
        top_bowlers = sorted(predictor.dynamic_bowler_factors.items(), key=lambda x: x[1])[:10]
        bottom_bowlers = sorted(predictor.dynamic_bowler_factors.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return jsonify({
            'success': True,
            'dynamic_factors': {
                'total_batters': len(predictor.dynamic_batter_factors),
                'total_bowlers': len(predictor.dynamic_bowler_factors),
                'top_batters': top_batters,
                'bottom_batters': bottom_batters,
                'top_bowlers': top_bowlers,
                'bottom_bowlers': bottom_bowlers
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting player factors: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with enhanced death over system"""
    try:
        logger.info("🔥 ENHANCED DEATH OVER Prediction request received")
        
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
        
        # Validate required fields
        required_fields = [
            'venue', 'batting_team', 'bowling_team', 'batter', 'bowler',
            'over', 'cumulative_runs', 'cumulative_wickets', 'pitch_type'
        ]
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing fields: {missing_fields}'
            }), 400
        
        # Make prediction with ENHANCED DEATH OVER algorithm
        prediction = predictor.predict_final_score(input_data)
        
        # Get factor info for response
        batter_factor = predictor.get_dynamic_batter_factor(
            input_data.get('batter', ''), predictor.dynamic_batter_factors
        )
        bowler_factor = predictor.get_dynamic_bowler_factor(
            input_data.get('bowler', ''), predictor.dynamic_bowler_factors
        )
        
        # Get death over info
        current_over = input_data.get('over', 0)
        is_death_over = current_over >= 16
        death_boost = predictor.get_death_over_boost(current_over, 0) if is_death_over else 1.0
        
        logger.info(f"✅ ENHANCED DEATH OVER Prediction successful: {prediction:.1f}")
        
        return jsonify({
            'success': True,
            'predicted_score': float(round(prediction, 1)),
            'timestamp': datetime.now().isoformat(),
            'model_used': 'XGBoost-ENHANCED-DEATH' if predictor.score_model is not None else 'Fallback-ENHANCED-DEATH',
            'factors_applied': {
                'batter': input_data.get('batter', ''),
                'batter_factor': round(batter_factor, 3),
                'bowler': input_data.get('bowler', ''),
                'bowler_factor': round(bowler_factor, 3),
                'pitch_type': input_data.get('pitch_type', ''),
                'venue': input_data.get('venue', ''),
                'batting_team': input_data.get('batting_team', ''),
                'is_death_over': is_death_over,
                'death_boost': round(death_boost, 3) if is_death_over else None
            },
            'version': '4.0.0'
        })
        
    except Exception as e:
        logger.error(f"❌ ENHANCED DEATH OVER Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting ENHANCED DEATH OVER XGBoost Cricket Prediction API...")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    
    # List available files
    files = [f for f in os.listdir('.') if f.endswith(('.json', '.pkl', '.csv'))]
    logger.info(f"📋 Available files: {files}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)