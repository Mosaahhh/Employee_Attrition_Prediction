app.title = "Employee Attrition Prediction"
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
import numpy as np
import dash_bootstrap_components as dbc

# Load your trained model and scaler
model = joblib.load("logic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for deployment on Render
app.title = "Employee Attrition Prediction"

# Layout
app.layout = dbc.Container([
    html.H1("Employee Attrition Predictor", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Age"),
            dcc.Input(id="input-age", type="number", placeholder="e.g. 35", className="form-control")
        ]),
        dbc.Col([
            dbc.Label("Monthly Income"),
            dcc.Input(id="input-income", type="number", placeholder="e.g. 5000", className="form-control")
        ]),
        dbc.Col([
            dbc.Label("Job Role"),
            dcc.Dropdown(
                id="input-role",
                options=[
                    {"label": "Sales Executive", "value": "Sales Executive"},
                    {"label": "Research Scientist", "value": "Research Scientist"},
                    {"label": "Laboratory Technician", "value": "Laboratory Technician"},
                    # Add all job roles seen during training
                ],
                placeholder="Select Job Role"
            )
        ])
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Years at Company"),
            dcc.Input(id="input-years", type="number", placeholder="e.g. 5", className="form-control")
        ]),
        dbc.Col([
            dbc.Label("OverTime"),
            dcc.Dropdown(
                id="input-overtime",
                options=[
                    {"label": "Yes", "value": "Yes"},
                    {"label": "No", "value": "No"}
                ],
                placeholder="Select OverTime Status"
            )
        ])
    ], className="mb-3"),

    dbc.Button("Predict", id="predict-btn", color="primary", className="mb-4"),

    html.Div(id="prediction-output", className="fs-4 fw-bold")
], fluid=True)

# Callback
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("input-age", "value"),
    State("input-income", "value"),
    State("input-role", "value"),
    State("input-years", "value"),
    State("input-overtime", "value")
)
def predict_attrition(n_clicks, age, income, role, years, overtime):
    if not n_clicks:
        return ""
    try:
        # Manual encoding for categorical features (make sure it matches training preprocessing)
        role_map = {
            "Sales Executive": 0,
            "Research Scientist": 1,
            "Laboratory Technician": 2,
            # Continue based on training
        }
        overtime_map = {"No": 0, "Yes": 1}

        data = pd.DataFrame([[
            age,
            income,
            role_map.get(role, 0),
            years,
            overtime_map.get(overtime, 0)
        ]], columns=["Age", "MonthlyIncome", "JobRole", "YearsAtCompany", "OverTime"])

        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0][1]

        if prediction == 1:
            return f"⚠️ Attrition Likely (Confidence: {prob:.2%})"
        else:
            return f"✅ Employee Likely to Stay (Confidence: {1 - prob:.2%})"

    except Exception as e:
        return f"Error during prediction: {str(e)}"
        
@app.callback(
    Output("prediction-output", "children"),
    Output("prediction-graph", "figure"),
    Input("predict-btn", "n_clicks"),
    State("input-age", "value"),
    State("input-income", "value"),
    State("input-role", "value"),
    State("input-years", "value"),
    State("input-overtime", "value")
)
def predict_attrition(n_clicks, age, income, role, years, overtime):
    if not n_clicks or ctx.triggered_id != "predict-btn":
        return "", go.Figure()

    try:
        # Manual encoding for categorical features
        role_map = {
            "Sales Executive": 0,
            "Research Scientist": 1,
            "Laboratory Technician": 2,
            # Add others as needed
        }
        overtime_map = {"No": 0, "Yes": 1}

        data = pd.DataFrame([[
            age,
            income,
            role_map.get(role, 0),
            years,
            overtime_map.get(overtime, 0)
        ]], columns=["Age", "MonthlyIncome", "JobRole", "YearsAtCompany", "OverTime"])

        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0]

        message = (
            f"⚠️ Attrition Likely (Confidence: {prob[1]:.2%})"
            if prediction == 1
            else f"✅ Employee Likely to Stay (Confidence: {prob[0]:.2%})"
        )

        # Visualization
        fig = go.Figure(
            data=[
                go.Bar(x=["No Attrition", "Attrition"], y=[prob[0], prob[1]], marker_color=["green", "red"])
            ]
        )
        fig.update_layout(
            title="Prediction Confidence",
            yaxis=dict(title="Probability"),
            xaxis=dict(title="Outcome"),
            height=400
        )

        return message, fig

    except Exception as e:
        return f"Error during prediction: {str(e)}", go.Figure()
        
# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
