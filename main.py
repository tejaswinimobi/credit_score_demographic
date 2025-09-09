from fastapi import FastAPI
from pydantic import BaseModel
from app.pipeline import run_combined_pipeline_single

app = FastAPI(title="Credit Decision Engine", version="1.0.0",debug=True)

# ✅ Input schema for API
class CustomerData(BaseModel):
    ph_number: str
    latitude: float
    longitude: float
    total_credit_amount: float
    monthly_income: float
    monthly_installments: float
    monthly_saving: float
    currency: str
    date_of_birth: str
    employed_since: str
    occupation_type: str
    highest_education: str
    marital_status: str
    employee_type: str
    gender: str
    city: str
    cc_code: str
    device: str


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/predict")
def predict(data: CustomerData):
    """
    Run the prediction pipeline with customer data
    """
    # Convert Pydantic model → dict
    entry = data.dict()

    # Run pipeline (must return DataFrame)
    result_df = run_combined_pipeline_single(entry)

    # Select only required fields for response
    result = result_df[[
        "ph_number",
        "credit_score",
        "combined_prob",
        "combined_pred",
        "customer_group",
        "predicted_income_usd",
        "predicted_loan_usd",
        "alpha_predicted_income_usd",
        "alpha_predicted_loan_usd"
    ]].iloc[0].to_dict()

    return {"prediction": result}
