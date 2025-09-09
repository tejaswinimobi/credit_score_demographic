import requests

url = "http://127.0.0.1:8080/predict"

sample_entry = {
    "ph_number": "260971230000",
    "latitude": -40.4167,
    "longitude": 28.2833,
    "total_credit_amount": 100,
    "monthly_income": 200,
    "monthly_installments": 0,
    "monthly_saving": 200,
    "currency": "ZMW",
    "date_of_birth": "1990-05-19",
    "employed_since": "2015-08-01",
    "occupation_type": "Teacher",
    "highest_education": "Bachelor",
    "marital_status": "Married",
    "employee_type": "Full-time",
    "gender": "female",
    "city": "Lusaka",
    "cc_code": "ZM",
    "device": "Samsung Galaxy A12"
}

response = requests.post(url, json=sample_entry)

print("Status Code:", response.status_code)
print("Raw Response:", response.text)   # <-- Debug
