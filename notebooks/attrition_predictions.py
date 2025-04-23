import dash
from dash import dcc, html, Input, Output, State, dash_table
import base64
import io
import pandas as pd
import joblib
import plotly.express as px
import dash_bootstrap_components as dbc

# Load trained model and scaler
model = joblib.load("artifacts/logistic_regression_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
feature_importance_df = pd.read_csv("artifacts/feature_importance.csv")
trained_features = feature_importance_df["Feature"].tolist()

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Employee Attrition Prediction"

app.layout = dbc.Container([
    html.H2("Employee Attrition Predictor", className="text-center my-4"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select CSV File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '10px',
            'textAlign': 'center',
            'marginBottom': '20px'
        },
        multiple=False
    ),
    html.Div(id='output-table'),
    dcc.Graph(id='probability-plot', style={"marginTop": "40px"})
], fluid=True)

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    else:
        return None
    return df

def make_predictions(df):
    df = df[trained_features]
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    probs = model.predict_proba(df_scaled)[:, 1]
    df['Predicted_Attrition'] = preds
    df['Probability_Yes'] = probs
    return df

@app.callback(
    Output('output-table', 'children'),
    Output('probability-plot', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        df = parse_data(contents, filename)
        if df is not None:
            df_pred = make_predictions(df)
            table = dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df_pred.columns],
                data=df_pred.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'fontWeight': 'bold'}
            )
            fig = px.histogram(df_pred, x="Probability_Yes", nbins=20,
                               title="Predicted Probability of Attrition",
                               labels={"Probability_Yes": "Attrition Probability"})
            fig.add_vline(x=0.5, line_dash="dash", line_color="red")
            return table, fig
    return html.Div(), {}

if __name__ == '__main__':
    app.run_server(debug=True)
