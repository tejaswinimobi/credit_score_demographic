import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from fuzzywuzzy import process, fuzz
import os

# =========================
# Load Models & Assets
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

address_model = joblib.load(os.path.join(BASE_DIR, "lgb_model.pkl"))
address_scaler = joblib.load(os.path.join(BASE_DIR, "lgb_feature_scaler.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
demo_model = load_model(os.path.join(BASE_DIR, "ann_default_model.h5"))
best_thresh = joblib.load(os.path.join(BASE_DIR, "best_threshold.pkl"))
encoders_mappings = joblib.load(os.path.join(BASE_DIR, "encoders_and_mappings.pkl"))
industry_occupations = joblib.load(os.path.join(BASE_DIR, "industry_occupations.pkl"))

# =========================
# Helper Functions
# =========================
def match_occupation(occ):
    flat_list = []
    for industry, segments in industry_occupations.items():
        for seg in ['High', 'Medium', 'Low']:
            for job in segments.get(seg, []):
                flat_list.append((job.lower(), industry, seg))
    occ_lower = str(occ).strip().lower()
    titles = [x[0] for x in flat_list]
    best_match = process.extractOne(occ_lower, titles, scorer=fuzz.partial_ratio)
    if best_match and best_match[1] > 70:
        idx = titles.index(best_match[0])
        return flat_list[idx][1], flat_list[idx][2]
    return 'Unknown', 'Unknown'

def calculate_credit_score(prob):
    return np.round(850 - 400 * prob).astype(int)

# =========================
# Customer Group Categorization (NO days_pass_due anymore)
# =========================
def categorize_customer(prob_default):
    if prob_default < 0.2:
        return "Good"
    elif prob_default < 0.4:
        return "Average"
    elif prob_default < 0.6:
        return "Risky"
    else:
        return "Bad"

def predict_income_and_loan(df):
    results = []
    for _, row in df.iterrows():
        income = row.get("monthly_income_usd", 0)
        loan = row.get("total_credit_amount_usd", 0)
        prob_default = row.get("combined_prob", 0.0)

        # assign group (based only on probability now)
        group = categorize_customer(prob_default)

        # rules
        if group == "Good":
            predicted_income = income * 1.1
            predicted_loan = loan * 1.15
        elif group == "Average":
            predicted_income = income
            predicted_loan = loan
        elif group == "Risky":
            predicted_income = income * 0.9
            predicted_loan = loan * 0.8
        else:  # Bad
            predicted_income = income * 0.8
            predicted_loan = loan * 0.6

        results.append({
            "ph_number": row["ph_number"],
            "customer_group": group,
            "predicted_income_usd": round(predicted_income, 2),
            "predicted_loan_usd": round(predicted_loan, 2)
        })
    return pd.DataFrame(results)

# =========================
# α-based Income & Loan Prediction
# =========================
from sklearn.metrics import mean_squared_error

def loss_function(alpha, df, beta=0.5):
    df = df.copy()
    df['pred_income'] = df['monthly_income_usd'] * (1 - alpha * df['combined_prob'])
    non_defaulters = df[df['demo_pred'] == 0]
    defaulters = df[df['demo_pred'] == 1]

    mse_nd = mean_squared_error(
        non_defaulters['monthly_income_usd'], 
        non_defaulters['pred_income']
    ) if not non_defaulters.empty else 0

    penalty_d = defaulters['pred_income'].mean() if not defaulters.empty else 0
    return mse_nd + beta * penalty_d

def learn_alpha_grid(df, alpha_grid=np.linspace(0, 1.5, 31), beta=0.5):
    best_alpha, best_loss = None, float("inf")
    for alpha in alpha_grid:
        loss = loss_function(alpha, df, beta=beta)
        if loss < best_loss:
            best_loss, best_alpha = loss, alpha
    return best_alpha

def apply_income_loan_prediction(df, alpha):
    df = df.copy()
    df['alpha_predicted_income_usd'] = df['monthly_income_usd'] * (1 - alpha * df['combined_prob'])
    df['alpha_predicted_loan_usd'] = df['total_credit_amount_usd'] * (1 - alpha * df['combined_prob'])
    df['alpha_predicted_income_usd'] = df['alpha_predicted_income_usd'].clip(lower=0).round(2)
    df['alpha_predicted_loan_usd'] = df['alpha_predicted_loan_usd'].clip(lower=0).round(2)
    return df

# =========================
# Demographic Preprocessing (unchanged)
# =========================
def preprocess_and_predict_demo(df):
    today = pd.to_datetime('today')

    # Currency conversion
    exchange_rates = {
        'Ghanaian Cedi': 0.097, 'Kwacha': 0.041, 'US Dollars': 1.0,
        'West African franc': 0.0018, 'Leone': 0.04429, 'Depreciated Curr - Kwach': 0.04053
    }
    monetary_columns = ['total_credit_amount', 'monthly_income', 'monthly_installments', 'monthly_saving']
    for col in monetary_columns:
        df[col + '_usd'] = df.apply(lambda row: row[col] * exchange_rates.get(row['currency'], 1.0), axis=1)
    df.drop(columns=monetary_columns, inplace=True)

    # Age & employment duration
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['employed_since'] = pd.to_datetime(df['employed_since'], errors='coerce')
    df['age'] = (today - df['date_of_birth']).dt.days // 365
    df['age_employment'] = (today - df['employed_since']).dt.days // 365

    # Derived ratio
    df['savings_to_income'] = df['monthly_saving_usd'] / (df['monthly_income_usd'] + 1e-6)

    # Occupation mapping
    df[['industry', 'industry_segment']] = df['occupation_type'].apply(lambda x: pd.Series(match_occupation(x)))
    df['industry_segment'] = df['industry_segment'].map(encoders_mappings['segment_encoding']).fillna(-1)
    df['industry_freq_encoded'] = df['industry'].map(encoders_mappings['industry_freq_map']).fillna(encoders_mappings['global_mean'])

    # Education & location encoding
    df['education_encoded'] = df['highest_education'].map(encoders_mappings['education_order']).fillna(-1)
    df['city_freq'] = df['city'].map(encoders_mappings['city_freq_map']).fillna(0)
    df['country_freq'] = df['cc_code'].map(encoders_mappings['country_freq_map']).fillna(0)
    df['currency_encoded'] = df['currency'].map(encoders_mappings['currency_map']).fillna(encoders_mappings['global_mean'])

    # Device group encoding
    def map_device_group(brand):
        brand = str(brand).lower()
        if brand in ['nokia', 'samsung']:
            return 'Top'
        elif brand in ['tecno', 'zte', 'hmd (nokia)']:
            return 'Mid'
        else:
            return 'Low'
    df['device_group'] = df['device'].apply(map_device_group)
    df['device_group_encoded'] = encoders_mappings['device_label_encoder'].transform(df['device_group'])

    # One-hot encoding
    for col in encoders_mappings['marital_status_dummy_columns']:
        base = col.replace('marital_status_', '')
        df[col] = (df['marital_status'].str.lower() == base).astype(int)
    for col in encoders_mappings['employee_type_dummy_columns']:
        base = col.replace('employee_type_', '')
        df[col] = (df['employee_type'].str.lower() == base).astype(int)
    for col in encoders_mappings['gender_dummy_columns']:
        base = col.replace('gender_', '')
        df[col] = (df['gender'].str.lower() == base).astype(int)

    # Ensure required columns
    trained_cols = scaler.feature_names_in_
    for col in trained_cols:
        if col not in df.columns:
            df[col] = 0
    scaled_part = pd.DataFrame(scaler.transform(df[trained_cols]), columns=trained_cols, index=df.index)

    # Boolean features
    bool_cols = encoders_mappings['marital_status_dummy_columns'] + \
                encoders_mappings['employee_type_dummy_columns'] + \
                encoders_mappings['gender_dummy_columns']
    for col in bool_cols:
        if col not in df.columns:
            df[col] = 0

    df_final = pd.concat([scaled_part, df[bool_cols]], axis=1)

    # Predict
    y_prob = demo_model.predict(df_final).flatten()
    y_pred = (y_prob > best_thresh).astype(int)

    df['demo_pred_prob'] = y_prob
    df['demo_pred'] = y_pred
    df['credit_score'] = calculate_credit_score(y_prob)

    return df[['ph_number', 'demo_pred', 'demo_pred_prob', 'credit_score',
               'monthly_income_usd', 'total_credit_amount_usd']]

# =========================
# Combined Runner
# =========================
def run_combined_pipeline_single(entry_dict):
    df = pd.DataFrame([entry_dict])

    # --- Address Model ---
    addr_df = df[['ph_number', 'latitude', 'longitude']].dropna().copy()
    if not addr_df.empty:
        X_addr = addr_df[['latitude', 'longitude']]
        X_addr_scaled = address_scaler.transform(X_addr)
        addr_prob = address_model.predict_proba(X_addr_scaled)[:, 1]
        addr_pred = (addr_prob >= 0.52).astype(int)
        addr_df['address_pred'] = addr_pred
        addr_df['address_model_prob'] = addr_prob
    else:
        addr_df = pd.DataFrame([{
            'ph_number': entry_dict['ph_number'],
            'address_pred': 0,
            'address_model_prob': 0.5
        }])

    # --- Demographic Model ---
    demo_df = preprocess_and_predict_demo(df.copy())

    # --- Merge Results ---
    final_df = pd.merge(addr_df[['ph_number', 'address_pred', 'address_model_prob']], 
                        demo_df, on='ph_number', how='inner')

    # --- Weighted Combine ---
    final_df['combined_prob'] = (0.25 * final_df['address_model_prob']) + (0.75 * final_df['demo_pred_prob'])
    final_df['combined_pred'] = (final_df['combined_prob'] >= 0.5).astype(int)

    # --- Income & Loan Prediction ---
    enriched_df = predict_income_and_loan(final_df)
    final_df = final_df.merge(enriched_df, on="ph_number", how="left")

    # --- α-based Income & Loan Prediction ---
    best_alpha = 0.65
    final_df = apply_income_loan_prediction(final_df, best_alpha)

    return final_df
