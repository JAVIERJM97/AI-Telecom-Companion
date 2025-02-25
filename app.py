from flask import Flask, render_template, request, jsonify
import openai
import pandas as pd
import pickle
import re
import os
import logging

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['JSON_AS_ASCII'] = False 

logging.basicConfig(level=logging.DEBUG)

with open("key.txt", "r") as file:
    openai_api_key = file.readline().strip()
openai.api_key = openai_api_key


PLAN_TABLE = """
Available Plans (concise):
1) Mobile - Basic
   - Data: 2GB
   - Unlimited Data: No
   - TV: No
   - Fiber: No
   - Price: $10 - $20

2) Mobile - Standard
   - Data: 5GB
   - Unlimited Data: No
   - TV: No
   - Fiber: No
   - Price: $20 - $30

3) Mobile - Unlimited
   - Data: 100GB (Unlimited)
   - TV: No
   - Fiber: No
   - Price: $40 - $50

4) Fiber + TV
   - Data: 0GB
   - TV: Yes
   - Fiber: Yes
   - Price: $30 - $40

5) Fiber + TV + Mobile (Basic)
   - Data: 0GB
   - TV: Yes
   - Fiber: Yes
   - Price: $40 - $50

6) Fiber + TV + Mobile (Premium)
   - Data: 50GB
   - TV: Yes
   - Fiber: Yes
   - Price: $50 - $70

7) Fiber + TV + Mobile (Unlimited)
   - Data: 100GB (Unlimited)
   - TV: Yes
   - Fiber: Yes
   - Price: $60 - $80

8) Fiber Only (Basic)
   - TV: No
   - Fiber: Yes
   - Price: $20 - $30

9) Fiber Only (Premium)
   - TV: No
   - Fiber: Yes
   - Price: $30 - $40

10) Mobile + Fiber (Basic)
    - Data: 2GB
    - Fiber: Yes
    - TV: No
    - Price: $30 - $40

11) Mobile + Fiber (Standard)
    - Data: 5GB
    - Fiber: Yes
    - TV: No
    - Price: $40 - $50
"""

def ask_ai_about_plans(user_question):
    """
    Sends user_question to GPT along with the PLAN_TABLE context.
    Returns a short, concise answer about available plans.
    """
    prompt = f"""
You have the following table of available plans:

{PLAN_TABLE}

The user asks: \"{user_question}\"

Please provide a short, concise answer with only the necessary plan info.
No extra commentary.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant with knowledge about telecom plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        answer = response["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        app.logger.error("Error in ask_ai_about_plans: %s", str(e))
        return "Error in AI plan query."

with open("xgb_model.pkl", "rb") as churn_file:
    xgb_model = pickle.load(churn_file)

strategy_models = {}
for strategy in [
    "10% Discount", "Contract Freeze", "Dedicated Customer Support",
    "Extra Data Package", "Loyalty Bonus", "Upgrade to Better Plan", "iPhone Discount or Free Device"
]:
    model_filename = f"strategy_model_{strategy.replace(' ', '_').replace('%', 'percent')}.pkl"
    if os.path.exists(model_filename):
        with open(model_filename, "rb") as strategy_file:
            strategy_models[strategy] = pickle.load(strategy_file)

telecom = pd.read_csv("Updated_Telecom_Data.csv", dtype=str)
telecom.columns = [col.strip().lower() for col in telecom.columns]

def get_customer_info(first_name, last_name):
    customer = telecom[
        (telecom["first_name"].str.lower() == first_name.lower()) &
        (telecom["last_name"].str.lower() == last_name.lower())
    ]
    if customer.empty:
        return None
    return customer.iloc[0].to_dict()

def predict_churn(customer_info):
    churn_features = pd.DataFrame([customer_info])
    churn_features = pd.get_dummies(churn_features)
    model_columns = xgb_model.get_booster().feature_names
    churn_features = churn_features.reindex(columns=model_columns, fill_value=0)
    churn_proba = xgb_model.predict_proba(churn_features)[0][1]
    churn_label = "High" if churn_proba > 0.66 else "Medium" if churn_proba > 0.33 else "Low"
    return churn_label, churn_proba

def recommend_best_strategy(customer_info):
    import pandas as pd
    if not strategy_models:
        return "No recommended strategy", 0.0
    churn_features = pd.DataFrame([customer_info])
    churn_features = pd.get_dummies(churn_features)
    model_columns = list(strategy_models.values())[0].get_booster().feature_names
    churn_features = churn_features.reindex(columns=model_columns, fill_value=0)
    best_strategy = None
    best_success_probability = 0.0
    for strategy, model in strategy_models.items():
        success_proba = model.predict_proba(churn_features)[0][1]
        if success_proba > best_success_probability:
            best_success_probability = success_proba
            best_strategy = strategy
    if best_strategy is None:
        best_strategy = list(strategy_models.keys())[0]
    return best_strategy, best_success_probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_plans', methods=['POST'])
def ask_plans():
    try:
        data = request.get_json()
        question = data.get('question', '')
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        short_plan_answer = ask_ai_about_plans(question)
        return jsonify({
            "error": "",
            "ai_plan_answer": short_plan_answer
        })
    except Exception as e:
        app.logger.error("Error in /ask_plans: %s", str(e))
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/get_customer', methods=['POST'])
def get_customer():
    try:
        data = request.get_json()
        
        if not data or 'first_name' not in data or 'last_name' not in data:
            return jsonify({"error": "No valid customer name provided"}), 400
        
        first_name = data['first_name']
        last_name  = data['last_name']
        customer_info = get_customer_info(first_name, last_name)
        if not customer_info:
            return jsonify({"error": "Customer not found."}), 404

        
        ct = customer_info.get('customer_type', '')
        if pd.isna(ct) or (isinstance(ct, str) and ct.strip() == ""):
            if 'customer_type' not in data:
                
                response_data = {
                    "first_name": customer_info.get('first_name', ''),
                    "last_name": customer_info.get('last_name', ''),
                    "age": customer_info.get('age', ''),
                    "residence": customer_info.get('residence', ''),
                    "phone_number": customer_info.get('phone_number', ''),
                    "email": customer_info.get('email', ''),
                    "tariff_plan": customer_info.get('tariff_plan', ''),
                    "contract_type": customer_info.get('contract_type', ''),
                    "customer_type": "Missing data",
                    "income_level": customer_info.get('income_level', ''),
                    "contract_satisfaction": customer_info.get('contract_satisfaction', ''),
                    "monthly_price": customer_info.get('monthly_price', ''),
                    "years_in_company": customer_info.get('years_in_company', ''),
                    "monthly_usage_hours": customer_info.get('monthly_usage_hours', ''),
                    "num_family_lines": customer_info.get('num_family_lines', ''),
                    "churn": "",
                    "retention_strategy": "",
                    "ai_response": "Customer type is missing. Please provide it.",
                    "missing": True
                }
                app.logger.debug("Partial response (missing customer_type): %s", response_data)
                return jsonify(response_data)
            else:
                customer_info['customer_type'] = data.get('customer_type').capitalize()
        
        
        churn_label, churn_proba = predict_churn(customer_info)
        if churn_label == "High":
            retention_strategy, _ = recommend_best_strategy(customer_info)
        else:
            retention_strategy = "Not needed"

        summary = f"""
Customer: {customer_info['first_name']} {customer_info['last_name']}
Age: {customer_info['age']}
Residence: {customer_info['residence']}
Tariff Plan: {customer_info['tariff_plan']}
Churn Prediction: {churn_label} ({churn_proba:.2f})
Retention Strategy: {retention_strategy}
"""
        try:
            openai_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant providing customer insights."},
                    {"role": "user", "content": summary}
                ]
            )
            ai_response = openai_response["choices"][0]["message"]["content"]
        except Exception as e:
            ai_response = f"Error in AI response: {str(e)}"

        return jsonify({
            "first_name": customer_info.get('first_name', ''),
            "last_name": customer_info.get('last_name', ''),
            "age": customer_info.get('age', ''),
            "residence": customer_info.get('residence', ''),
            "phone_number": customer_info.get('phone_number', ''),
            "email": customer_info.get('email', ''),
            "tariff_plan": customer_info.get('tariff_plan', ''),
            "contract_type": customer_info.get('contract_type', ''),
            "customer_type": customer_info.get('customer_type', ''),
            "income_level": customer_info.get('income_level', ''),
            "contract_satisfaction": customer_info.get('contract_satisfaction', ''),
            "monthly_price": customer_info.get('monthly_price', ''),
            "years_in_company": customer_info.get('years_in_company', ''),
            "monthly_usage_hours": customer_info.get('monthly_usage_hours', ''),
            "num_family_lines": customer_info.get('num_family_lines', ''),
            "churn": churn_label,
            "retention_strategy": retention_strategy,
            "ai_response": ai_response,
            "missing": False
        })
    except Exception as e:
        app.logger.error("Error in /get_customer: %s", str(e))
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True)
