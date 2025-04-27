import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import plotly.graph_objs as go

# Load the dataset to get unique job roles
data = pd.read_csv("data/HR-Employee-Attrition.csv")
job_roles = sorted(data["JobRole"].unique())

# Load the trained model and scaler
model = joblib.load("artifacts/logistic_regression_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment
app.title = "Employee Attrition Prediction"

# Layout
app.layout = dbc.Container([
    html.H1("Employee Attrition Predictor", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Age"),
            dcc.Input(id="input-age", type="number", placeholder="e.g. 35", className="form-control", min=18, max=100)
        ], width=4),
        dbc.Col([
            dbc.Label("Monthly Income"),
            dcc.Input(id="input-income", type="number", placeholder="e.g. 5000", className="form-control", min=0)
        ], width=4),
        dbc.Col([
            dbc.Label("Job Role"),
            dcc.Dropdown(
                id="input-role",
                options=[{"label": role, "value": role} for role in job_roles],
                placeholder="Select Job Role",
                className="form-control"
            )
        ], width=4)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Years at Company"),
            dcc.Input(id="input-years", type="number", placeholder="e.g. 5", className="form-control", min=0)
        ], width=4),
        dbc.Col([
            dbc.Label("OverTime"),
            dcc.Dropdown(
                id="input-overtime",
                options=[
                    {"label": "Yes", "value": "Yes"},
                    {"label": "No", "value": "No"}
                ],
                placeholder="Select OverTime Status",
                className="form-control"
            )
        ], width=4)
    ], className="mb-3"),
    dbc.Button("Predict", id="predict-btn", color="primary", className="mb-4"),
    html.Div(id="prediction-output", className="fs-4 fw-bold text-center mb-4"),
    dcc.Graph(id="prediction-graph")
], fluid=True)

# Callback to handle prediction and visualization
@app.callback(
    [Output("prediction-output", "children"),
     Output("prediction-graph", "figure")],
    Input("predict-btn", "n_clicks"),
    [State("input-age", "value"),
     State("input-income", "value"),
     State("input-role", "value"),
     State("input-years", "value"),
     State("input-overtime", "value")]
)
def predict_attrition(n_clicks, age, income, role, years, overtime):
    # Initialize empty outputs if button not clicked
    if not n_clicks:
        return "", go.Figure()

    # Validate inputs
    if any(x is None for x in [age, income, role, years, overtime]):
        return "⚠️ Please fill in all fields.", go.Figure()

    try:
        # Define encoding mappings (must match training)
        role_map = {role: idx for idx, role in enumerate(job_roles)}
        overtime_map = {"No": 0, "Yes": 1}

        # Create input DataFrame
        data = pd.DataFrame([[
            age,
            income,
            role_map.get(role),
            years,
            overtime_map.get(overtime)
        ]], columns=["Age", "MonthlyIncome", "JobRole", "YearsAtCompany", "OverTime"])

        # Scale numerical features
        data_scaled = scaler.transform(data)

        # Make prediction
        prediction = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0]

        # Format prediction message
        message = (
            f"⚠️ Attrition Likely (Confidence: {prob[1]:.2%})"
            if prediction == 1
            else f"✅ Employee Likely to Stay (Confidence: {prob[0]:.2%})"
        )

        # Create visualization
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["No Attrition", "Attrition"],
                    y=[prob[0], prob[1]],
                    marker_color=["green", "red"]
                )
            ]
        )
        fig.update_layout(
            title="Prediction Confidence",
            yaxis=dict(title="Probability", range=[0, 1], tickformat=".0%"),
            xaxis=dict(title="Outcome"),
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )

        return message, fig

    except Exception as e:
        return f"Error during prediction: {str(e)}", go.Figure()

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
    
