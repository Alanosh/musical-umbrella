import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import xgboost as xgb
from scipy.stats import zscore, ks_2samp
from scipy.spatial.distance import euclidean
import json
import os
import warnings
from umap import UMAP
from sklearn.cluster import KMeans
from joblib import dump, load
import time
import sys
warnings.filterwarnings('ignore')

# Constants
DATA_DIR = 'memecoin_data'
PUMP_FILE = f"{DATA_DIR}/merged_pump.csv"  # updated filename
DUD_FILE = f"{DATA_DIR}/merged_dud.csv"    # updated filename
INTERVALS_PER_COIN = 360                   # updated interval count
CONFIDENCE_THRESHOLD_PUMP = 0.9
CONFIDENCE_THRESHOLD_DUD = 0.8
CACHE_FILE = 'cache.json'
ERROR_LOG = 'model_errors.log'
PREDICTIONS_FILE = 'predictions.txt'
MODEL_MAIN = 'xgboost_main.json'
MODEL_HONEYPOT = 'xgboost_honeypot.json'
ANOMALY_MODEL = 'anomaly_detector.pkl'

# 1. Data Preprocessing and Feature Engineering
def load_and_merge_data():
    try:
        pumps = pd.read_csv(PUMP_FILE)
        pumps['Is_Pump'] = 1
        duds = pd.read_csv(DUD_FILE)
        duds['Is_Pump'] = 0
        df = pd.concat([pumps, duds], ignore_index=True)
        df = df.sort_values(['Contract_Address', 'Interval']).reset_index(drop=True)
        if not all(df.groupby('Contract_Address')['Interval'].count() == INTERVALS_PER_COIN):
            raise ValueError("Invalid interval count per Contract_Address")
        return df
    except Exception as e:
        with open(ERROR_LOG, 'a') as f:
            f.write(f"Load error: {str(e)}\n")
        raise

def preprocess_data(df, is_training=True):
    features = ['Price_Surge', 'Volume_SOL', 'Insider_Ratio', 'Wallet_Diversity', 
                'Liquidity_Volume', 'Price_Volatility', 'Price_Momentum', 
                'Transaction_Frequency', 'Whale_Volume', 'Trade_Size_Variance', 
                'Sniper_Bot_Activity', 'Buy_Sell_Ratio', 'Failed_Trade_Ratio', 'TOTAL_NET_SOL']
    
    if not all(col in df.columns for col in features + ['Contract_Address', 'Interval']):
        raise ValueError("Missing required columns")
    
    scaler = RobustScaler()
    for addr in df['Contract_Address'].unique():
        mask = df['Contract_Address'] == addr
        df.loc[mask, features] = scaler.fit_transform(df.loc[mask, features])
    
    volatility = df['Price_Volatility'].rolling(window=INTERVALS_PER_COIN).mean().fillna(method='bfill')
    quantile = 0.005 if volatility.mean() > volatility.std() else 0.01
    for col in ['Volume_SOL', 'Price_Volatility', 'Whale_Volume', 'Transaction_Frequency']:
        lower, upper = df.groupby('Contract_Address')[col].transform(
            lambda x: x.quantile([quantile, 1-quantile])).unstack().values.T
        df[col] = df[col].clip(lower, upper)
    
    for col in ['Price_Surge', 'Price_Momentum', 'Transaction_Frequency']:
        volatility = df['Price_Volatility'].rolling(window=30).std().fillna(method='bfill')
        alpha = np.where(volatility > volatility.mean(), 0.05, 0.4)
        df[f'{col}_EMA'] = df.groupby('Contract_Address')[col].transform(
            lambda x: x.ewm(alpha=alpha[x.index], adjust=False).mean())
    
    df['Volatility_Spike_Indicator'] = (df['Price_Volatility'] > 
        (df['Price_Volatility'].rolling(window=30).mean() + 
         2.5 * df['Price_Volatility'].rolling(window=30).std())).astype(int) * df['Insider_Ratio']
    df['Insider_Clustering_Ratio'] = df['Insider_Ratio'] / (df['Wallet_Diversity'] + 1e-6)
    df['Liquidity_Decay_Metric'] = df.groupby('Contract_Address')['Liquidity_Volume'].transform(
        lambda x: x.pct_change(10)).fillna(0)
    df['Honeypot_Score'] = ((df['Liquidity_Decay_Metric'] < -0.6) & 
                            (df['Insider_Ratio'] > 0.75)).astype(int)
    df['Cross_Coin_Correlation'] = df.groupby('Interval')['Price_Surge'].transform(
        lambda x: x.corr(df.loc[x.index, 'Volume_SOL'])).fillna(0)
    df['Dark_Pool_Signal'] = (df['Whale_Volume'] / (df['Transaction_Frequency'] + 1e-6)).ewm(span=20).mean()
    df['Dark_Pool_Enhanced'] = df['Dark_Pool_Signal'] * (1 - df['Cross_Coin_Correlation'])
    
    for window in [6, 30, 60]:
        for col in ['Price_Surge', 'Volume_SOL', 'Sniper_Bot_Activity']:
            df[f'{col}_Max_{window}'] = df.groupby('Contract_Address')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max())
            df[f'{col}_Mean_{window}'] = df.groupby('Contract_Address')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'{col}_Std_{window}'] = df.groupby('Contract_Address')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()).fillna(0)
    
    df['Price_Surge_x_Insider_Ratio'] = df['Price_Surge'] * df['Insider_Ratio']
    df['Whale_Volume_div_Buy_Sell_Ratio'] = df['Whale_Volume'] / (df['Buy_Sell_Ratio'] + 1e-6)
    df['Sniper_Bot_x_Liquidity'] = df['Sniper_Bot_Activity'] * df['Liquidity_Volume']
    
    for col in ['Whale_Volume', 'Transaction_Frequency']:
        z_scores = df.groupby('Contract_Address')[col].transform(
            lambda x: np.abs(zscore(x.rolling(window=30, min_periods=1).mean().fillna(method='bfill'))))
        df[col] = np.where(z_scores > 3, 
                          df.groupby('Contract_Address')[col].transform(
                              lambda x: x.rolling(window=30, min_periods=1).median()), 
                          df[col])
    
    return df

# 2. Model Training with Noise Injection
def train_model(df):
    try:
        features = [col for col in df.columns if col not in ['Contract_Address', 'Interval', 'Is_Pump']]
        X = df[features]
        y = df['Is_Pump']
        
        X_noisy = X.copy()
        for col in X.columns:
            X_noisy[col] += np.random.normal(0, np.random.uniform(0.15, 0.35), X.shape[0])
        X_noisy = X_noisy.sample(frac=0.88)
        adv_mask = np.random.choice([True, False], size=len(X_noisy), p=[0.15, 0.85])
        X_noisy.loc[adv_mask, 'Volume_SOL'] *= 3
        X_noisy.loc[adv_mask, 'Insider_Ratio'] *= 1.5
        
        sample_weights = np.ones(len(y))
        sample_weights[df['Interval'] >= (INTERVALS_PER_COIN - 90)] *= 1.7
        
        noise_mask = df['Price_Volatility'] > (df['Price_Volatility'].mean() + 2.5 * df['Price_Volatility'].std())
        sample_weights[noise_mask] *= 0.4
        
        reducer = UMAP(n_components=3, random_state=42)
        clusters = reducer.fit_transform(X[['Price_Surge', 'Insider_Ratio', 'Liquidity_Volume']])
        kmeans = KMeans(n_clusters=8, random_state=42)
        cluster_labels = kmeans.fit_predict(clusters)
        
        models = []
        for cluster in range(8):
            mask = cluster_labels == cluster
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=0.67,
                max_depth=7,
                learning_rate=0.08,
                n_estimators=300,
                eval_metric='logloss',
                tree_method='hist',
                random_state=42
            )
            tscv = TimeSeriesSplit(n_splits=6)
            for train_idx, val_idx in tscv.split(X[mask]):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model.fit(X_train, y_train, sample_weight=sample_weights[train_idx])
                print("Expected output: Classification report for validation set")
                print("Required output: ", classification_report(y_val, model.predict(X_val)))
            model.fit(X[mask], y[mask], sample_weight=sample_weights[mask])
            models.append(model)
        
        main_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=0.67,
            max_depth=7,
            learning_rate=0.08,
            n_estimators=300,
            eval_metric='logloss',
            tree_method='hist',
            random_state=42
        )
        main_model.fit(X_noisy, y, sample_weight=sample_weights)
        
        honeypot_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=0.67,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42
        )
        honeypot_features = ['Honeypot_Score', 'Liquidity_Decay_Metric', 'Insider_Clustering_Ratio']
        honeypot_model.fit(X[honeypot_features], y)
        
        anomaly_detector = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
        anomaly_detector.fit(X)
        
        importance = main_model.feature_importances_
        feature_weights = {f: i for f, i in zip(features, importance)}
        noisy_features = [f for f in features if df[f].std() > 2 * df[f].std().mean()]
        for f in noisy_features:
            feature_weights[f] *= 0.5
        for col in X.columns:
            X[col] *= feature_weights.get(col, 1.0)
        
        pump_profile = X[y == 1].mean().to_dict()
        dud_profile = X[y == 0].mean().to_dict()
        cache = {
            'pump_profile': pump_profile,
            'dud_profile': dud_profile,
            'centroids': kmeans.cluster_centers_.tolist(),
            'price_surge_mean': X['Price_Surge'].mean()
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        
        return models, main_model, honeypot_model, anomaly_detector, feature_weights, kmeans
    except Exception as e:
        with open(ERROR_LOG, 'a') as f:
            f.write(f"Train error: {str(e)}\n")
        raise

# 3. Generic Data Processing
def process_data(data_chunk, models, main_model, honeypot_model, anomaly_detector, feature_weights, kmeans):
    try:
        df = pd.DataFrame(data_chunk)
        if not all(col in df.columns for col in ['Contract_Address', 'Interval'] + 
                   ['Price_Surge', 'Volume_SOL', 'Insider_Ratio', 'Wallet_Diversity', 
                    'Liquidity_Volume', 'Price_Volatility', 'Price_Momentum', 
                    'Transaction_Frequency', 'Whale_Volume', 'Trade_Size_Variance', 
                    'Sniper_Bot_Activity', 'Buy_Sell_Ratio', 'Failed_Trade_Ratio', 'TOTAL_NET_SOL']):
            raise ValueError("Invalid input data format")
        
        df = preprocess_data(df, is_training=False)
        features = [col for col in df.columns if col not in ['Contract_Address', 'Interval']]
        X = df[features]
        
        volatility = df['Price_Volatility'].mean()
        vol_mean = df['Price_Volatility'].rolling(window=INTERVALS_PER_COIN*24).mean().fillna(method='bfill').iloc[-1]
        pump_threshold = 0.85 if volatility > vol_mean else 0.9
        dud_threshold = 0.75 if volatility > vol_mean else 0.8
        
        probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
        honeypot_probs = honeypot_model.predict_proba(X[['Honeypot_Score', 'Liquidity_Decay_Metric', 'Insider_Clustering_Ratio']])[:, 1]
        final_probs = 0.7 * probs + 0.3 * (1 - honeypot_probs)
        preds = np.where(final_probs >= pump_threshold, 1, 0)
        
        tree_preds = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
        confidence_intervals = [f"{p:.2f}±{np.std(tree_preds[i]):.2f}" for i, p in enumerate(final_probs)]
        
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        pump_profile = np.array(list(cache['pump_profile'].values()))
        dud_profile = np.array(list(cache['dud_profile'].values()))
        centroids = np.array(cache['centroids'])
        similarity_scores = [1 - euclidean(X.iloc[i].values, pump_profile if p == 1 else dud_profile) 
                           for i, p in enumerate(preds)]
        cluster_distances = [min(euclidean(X.iloc[i].values, c) for c in centroids) for i in range(len(X))]
        
        recent_mask = df['Interval'] >= (INTERVALS_PER_COIN - 90)
        X.loc[recent_mask, ['Price_Momentum', 'Sniper_Bot_Activity']] *= 1.7
        
        anomalies = anomaly_detector.predict(X) == -1
        
        results = []
        for i, (pred, prob, ci, sim, dist, addr) in enumerate(zip(preds, final_probs, confidence_intervals, 
                                                                similarity_scores, cluster_distances, df['Contract_Address'])):
            if (pred == 1 and prob >= pump_threshold) or (pred == 0 and prob <= 1 - dud_threshold):
                top_features = sorted([(f, feature_weights.get(f, 0)) for f in features], 
                                    key=lambda x: x[1], reverse=True)[:3]
                result = {
                    'Contract_Address': addr,
                    'Predicted_Class': 'Pump' if pred == 1 else 'Dud',
                    'Confidence_Score': prob,
                    'Confidence_Interval': ci,
                    'Similarity_Score': sim,
                    'Cluster_Distance': dist,
                    'Key_Features': [f[0] for f in top_features],
                    'Is_Anomaly': anomalies[i]
                }
                with open(PREDICTIONS_FILE, 'a') as f:
                    f.write(json.dumps(result) + '\n')
                print("Expected output: Contract Address prediction result")
                print("Required output: ", result)
                results.append(result)
                start_time = time.time()
                while time.time() - start_time < 180:
                    print(f"Contract Address: {addr}", flush=True)
                    time.sleep(1)
                print("Stopped printing contract address after 3 minutes.")
        
        return results
    except Exception as e:
        with open(ERROR_LOG, 'a') as f:
            f.write(f"Process error: {str(e)}\n")
        return []

# 4. Model Adaptation and Monitoring (UNCHANGED except INTERVALS_PER_COIN)
def update_model_incrementally(data_chunk, models, main_model, honeypot_model, kmeans):
    try:
        df = pd.DataFrame(data_chunk)
        df = preprocess_data(df, is_training=False)
        X = df[[col for col in df.columns if col not in ['Contract_Address', 'Interval', 'Is_Pump']]]
        y = df.get('Is_Pump', pd.Series(np.zeros(len(df))))
        
        sample_weights = np.ones(len(y))
        sample_weights[df['Interval'] >= (INTERVALS_PER_COIN - 90)] *= 1.7
        noise_mask = df['Price_Volatility'] > (df['Price_Volatility'].mean() + 2.5 * df['Price_Volatility'].std())
        sample_weights[noise_mask] *= 0.4
        
        for model in models + [main_model]:
            model.fit(X, y, xgb_model=model.get_booster(), learning_rate=0.008)
        honeypot_model.fit(X[['Honeypot_Score', 'Liquidity_Decay_Metric', 'Insider_Clustering_Ratio']], y)
        
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        historical_dist = cache.get('price_surge_mean', X['Price_Surge'].mean())
        _, p_value = ks_2samp(X['Price_Surge'], 
                             np.random.normal(historical_dist, X['Price_Surge'].std(), len(X)))
        if p_value < 0.03:
            print("Drift detected, retraining...")
            df_all = pd.concat([load_and_merge_data(), df])
            return train_model(df_all)
        cache['price_surge_mean'] = X['Price_Surge'].mean()
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        
        return models, main_model, honeypot_model
    except Exception as e:
        with open(ERROR_LOG, 'a') as f:
            f.write(f"Update error: {str(e)}\n")
        return models, main_model, honeypot_model

# 5. Main Pipeline (UNCHANGED except INTERVALS_PER_COIN)
def run_pipeline():
    try:
        df = load_and_merge_data()
        models, main_model, honeypot_model, anomaly_detector, feature_weights, kmeans = train_model(df)
        main_model.save_model(MODEL_MAIN)
        for i, model in enumerate(models):
            model.save_model(f'xgboost_cluster_{i}.json')
        honeypot_model.save_model(MODEL_HONEYPOT)
        dump(anomaly_detector, ANOMALY_MODEL)
        last_mtime = 0
        while True:
            try:
                if os.path.exists('realtime_data.json'):
                    mtime = os.path.getmtime('realtime_data.json')
                    if mtime > last_mtime:
                        with open('realtime_data.json', 'r') as f:
                            data_chunk = json.load(f)
                        results = process_data(data_chunk, models, main_model, honeypot_model, 
                                             anomaly_detector, feature_weights, kmeans)
                        models, main_model, honeypot_model = update_model_incrementally(
                            data_chunk, models, main_model, honeypot_model, kmeans)
                        last_mtime = mtime
            except Exception as e:
                with open(ERROR_LOG, 'a') as f:
                    f.write(f"Pipeline loop error: {str(e)}\n")
    except Exception as e:
        with open(ERROR_LOG, 'a') as f:
            f.write(f"Pipeline error: {str(e)}\n")
        raise

if __name__ == '__main__':
    run_pipeline()
