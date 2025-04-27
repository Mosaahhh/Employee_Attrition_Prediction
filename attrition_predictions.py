import dash
from dash import html, dcc, Input, Output, State, ctx
import pandas as pd
import joblib
import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for deployment on Render
app.title = "Employee Attrition Prediction"

# Load your trained model and scaler
model = joblib.load("artifacts/logistic_regression_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

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

    html.Div(id="prediction-output", className="fs-4 fw-bold"),
    dcc.Graph(id="prediction-graph")
], fluid=True)

# Single callback that handles prediction + graph
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

        # Create a dictionary of all feature names, including those that were used in training
        feature_names = [
            "Age", "MonthlyIncome", "JobRole", "YearsAtCompany", "OverTime",
            "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely",
            "Department_Research & Development", "Department_Sales"
        ]

        # Initialize input data with default (0) values for all features
        input_data = {feature: 0 for feature in feature_names}

        # Fill in known inputs
        input_data["Age"] = age
        input_data["MonthlyIncome"] = income
        input_data["JobRole"] = role_map.get(role, 0)
        input_data["YearsAtCompany"] = years
        input_data["OverTime"] = overtime_map.get(overtime, 0)

        # Assuming BusinessTravel and Department are other categorical columns,
        # we'll need to set their one-hot encoded columns to 0 (default assumption).
        # Adjust the columns below based on actual encoding for your model:
        input_data["BusinessTravel_Travel_Frequently"] = 0
        input_data["BusinessTravel_Travel_Rarely"] = 0
        input_data["Department_Research & Development"] = 0
        input_data["Department_Sales"] = 0
        input_data["Education"] = 0

        # Create a DataFrame
        data = pd.DataFrame([input_data])

        # Scale the data
        data_scaled = scaler.transform(data)

        # Make prediction
        prediction = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0]

        # Prediction message
        message = (
            f"⚠️ Attrition Likely (Confidence: {prob[1]:.2%})"
            if prediction == 1
            else f"✅ Employee Likely to Stay (Confidence: {prob[0]:.2%})"
        )

        # Bar chart for probability
        fig = go.Figure(
            data=[go.Bar(
                x=["No Attrition", "Attrition"],
                y=[prob[0], prob[1]],
                marker_color=["green", "red"]
            )]
        )
        fig.update_layout(
            title="Prediction Confidence",
            yaxis_title="Probability",
            xaxis_title="Outcome",
            height=400
        )

        return message, fig

    except Exception as e:
        return f"Error during prediction: {str(e)}", go.Figure()

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

