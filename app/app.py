import os
import sys
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd

# 1. THIS MUST COME FIRST!
# It forces Python to look in the parent folder for the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. THEN YOU IMPORT FROM SRC!
from src.ml_engine import ChurnPredictor
from src.data_preprocessing import load_and_preprocess_data
import src.visualizations as viz
# --- Load Data Globally ---
DATA_FILEPATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'CUSTOMER_ANALYTICS_Telecom_churn.csv')
df_global = load_and_preprocess_data(DATA_FILEPATH)
# Initialize and Train ML Model
predictor = ChurnPredictor()
if not df_global.empty:
    predictor.train(df_global)
# --- Page Layout Functions ---
def create_layout_intro_explorer(df):
    page_id = "intro"
    if df.empty: return dbc.Container([dbc.Alert("Error loading data.", color="danger")], fluid=True)
    intro_text = html.Div([ html.H3("Telecom Customer Churn: Exploratory Data Analysis"), html.P(["This project performs an in-depth exploratory data analysis (EDA) on a telecom customer dataset to identify key factors contributing to customer churn. ", "The goal is to understand ", html.Strong("why"), " customers leave the service and build a profile of high-risk customers using data visualization techniques. ", "Interactive visualizations are created using Python, primarily leveraging Pandas for data manipulation and Plotly for plotting."]), html.H5("Dataset"), html.P("The analysis utilizes a telecom customer dataset containing information such as:"), html.Ul([ html.Li([html.Strong("Customer Demographics:"), " Gender, Senior Citizen status, Partner, Dependents."]), html.Li([html.Strong("Account Information:"), " Tenure (months), Contract type, Payment Method, Paperless Billing."]), html.Li([html.Strong("Service Subscriptions:"), " Phone Service, Multiple Lines, Internet Service (DSL, Fiber Optic, No), Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies."]), html.Li([html.Strong("Charges:"), " Monthly Charges, Total Charges."]), html.Li([html.Strong("Churn Status:"), " Whether the customer churned ('Yes' or 'No')."]), ]), html.H5("Methodology"), html.Ul([ html.Li([html.Strong("Data Cleaning & Preprocessing:"), " Handled missing values (e.g., TotalCharges for new customers), corrected data types, and mapped categorical features for clarity (e.g., SeniorCitizen 0/1 to 'No'/'Yes')."]), html.Li([html.Strong("Feature Engineering:"), " Created tenure groups (TenureGroup), a composite RiskScore based on known churn drivers, calculated the number of protective/optional services, and binned monthly charges (MonthlyChargeGroup)."]), html.Li([html.Strong("Exploratory Data Analysis (EDA) with Plotly:"), " Utilized various visualizations (Histograms, Box Plots, Scatter Plots, Bar Charts, Violin Plots, Heatmaps, Faceted Plots, Treemaps, Density Plots) to explore relationships between features and churn."]), ]), html.H5("Using This Page"), html.P(["This 'Introduction & Explorer' page provides this overview and allows you to interactively explore the underlying customer data. ", "Use the ", html.Strong("Search Bar"), " to find specific customers or values within the table. The table will update based on your selections."]) ], className="mb-4 p-3 bg-light rounded-3")
    search_bar = dbc.Input(id=f'{page_id}-search-input', type='search', placeholder='Search table data...', className="mb-3", debounce=True)
    data_table_card = dbc.Card([ dbc.CardHeader("Customer Data Explorer"), dbc.CardBody([ dbc.Alert("Scroll horizontally to see all columns. Click headers to sort.", color="secondary", className="d-none d-md-flex align-items-center"), dash_table.DataTable( id=f'{page_id}-data-table', columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns], data=df.to_dict('records'), editable=False, filter_action="none", sort_action="native", sort_mode="multi", page_action="native", page_current=0, page_size=15, style_table={'overflowX': 'auto', 'minWidth': '100%'}, style_cell={'textAlign': 'left', 'padding': '8px', 'fontFamily': 'sans-serif', 'fontSize': '0.9em', 'border': '1px solid #eee'}, style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'borderBottom': '2px solid black'}, style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}], export_format='csv', export_headers='display', ) ]) ])
    return dbc.Container([ dbc.Row(dbc.Col(intro_text, width=12)), dbc.Row(dbc.Col(search_bar, width=12, md=6)), dbc.Row(dbc.Col(data_table_card, width=12, className="mb-4")) ], fluid=True)

def create_layout_overview(df):
    page_id = "overview"
    if df.empty: return dbc.Container([dbc.Alert("Error loading data.", color="danger")], fluid=True)
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("Customer Overview & Key Churn Factors"), width=12, className="mb-4 text-center")),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Churn Rate by Demographics"), dbc.CardBody(dcc.Graph(id=f'{page_id}-demo', figure=viz.create_subplots_demo_churn(df)))]), dbc.Alert("Gender shows little difference in churn. Senior citizens exhibit a higher churn rate compared to younger customers. Customers without dependents are more likely to churn than those with dependents.", color="light", className="mt-2")], width=12, className="mb-4"), ]),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Churn Rate by Contract Type"), dbc.CardBody(dcc.Graph(id=f'{page_id}-contract', figure=viz.create_bar_contract_churn(df)))]), dbc.Alert("Month-to-month contracts have a significantly higher churn rate (over 40%) compared to one-year (around 11%) and two-year contracts (under 3%), indicating longer contracts provide more stability.", color="light", className="mt-2")], width=12, md=6, className="mb-4"), dbc.Col([dbc.Card([dbc.CardHeader("Churn Rate by Tenure Group"), dbc.CardBody(dcc.Graph(id=f'{page_id}-tenure', figure=viz.create_bar_tenure_group_churn(df)))]), dbc.Alert("Churn is highest for newest customers (0-1 Year tenure, ~47%) and steadily decreases as tenure increases, with customers over 5 years showing the lowest churn rate (~6.6%).", color="light", className="mt-2")], width=12, md=6, className="mb-4"), ]),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Churn Rate by Payment Method"), dbc.CardBody(dcc.Graph(id=f'{page_id}-payment', figure=viz.create_bar_payment_churn(df)))]), dbc.Alert("Electronic check payment method has the highest churn rate (over 45%), while automatic payment methods (bank transfer, credit card) have lower churn.", color="light", className="mt-2")], width=12, md=6, className="mb-4"), dbc.Col([dbc.Card([dbc.CardHeader("Churn Rate by Paperless Billing"), dbc.CardBody(dcc.Graph(id=f'{page_id}-paperless', figure=viz.create_bar_paperless_churn(df)))]), dbc.Alert("Customers using paperless billing have a higher churn rate (around 33%) than those who do not (around 16%).", color="light", className="mt-2")], width=12, md=6, className="mb-4"), ]),
    ], fluid=True)

def create_layout_financial(df):
    page_id = "financial"
    if df.empty: return dbc.Container([dbc.Alert("Error loading data.", color="danger")], fluid=True)
    contract_options = []; payment_options = []
    if 'Contract' in df.columns: contract_options = [{'label': 'All Contracts', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted(df['Contract'].unique())]
    if 'PaymentMethod' in df.columns: payment_options = [{'label': 'All Payment Methods', 'value': 'All'}] + [{'label': i, 'value': i} for i in sorted(df['PaymentMethod'].unique())]
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("Financial Factors and Churn"), width=12, className="mb-4 text-center")),
        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([ dbc.Row([ dbc.Col(dbc.Label("Filter by Contract Type:", html_for=f'{page_id}-contract-dropdown'), width="auto"), dbc.Col(dcc.Dropdown(id=f'{page_id}-contract-dropdown', options=contract_options, value='All', clearable=False), width=True), ], className="mb-2"), dbc.Row([ dbc.Col(dbc.Label("Filter by Payment Method:", html_for=f'{page_id}-payment-dropdown'), width="auto"), dbc.Col(dcc.Dropdown(id=f'{page_id}-payment-dropdown', options=payment_options, value='All', clearable=False), width=True), ]) ])), width=12, className="mb-4")),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Monthly Charges Distribution"), dbc.CardBody(dcc.Graph(id=f'{page_id}-hist-monthly', figure=viz.create_hist_monthly(df)))]), dbc.Alert("Customers who churn (red) tend to have a distribution of monthly charges skewed towards higher values compared to non-churned customers (green). There are notable peaks for churners at higher charge brackets.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), dbc.Col([dbc.Card([dbc.CardHeader("Tenure vs Total Charges"), dbc.CardBody(dcc.Graph(id=f'{page_id}-scatter-tenure-total', figure=viz.create_scatter_tenure_total(df)))]), dbc.Alert("Churned customers (red) generally cluster at lower tenure and consequently lower total charges. Non-churned customers (green) form a clearer, positive correlation: as tenure increases, total charges accumulate. The trendlines indicate this general relationship.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), ]),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Distribution of Tenure by Churn Status"), dbc.CardBody(dcc.Graph(id=f'{page_id}-hist-tenure', figure=viz.create_hist_tenure_churn(df)))]), dbc.Alert("The histogram reveals that a significant number of customers churn within the first few months of their service. For non-churned customers, the tenure distribution is more spread out, indicating customer loyalty increases over time.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), dbc.Col([dbc.Card([dbc.CardHeader("Monthly Charges by Contract (Violin)"), dbc.CardBody(dcc.Graph(id=f'{page_id}-violin-monthly-contract', figure=viz.create_violin_monthly_contract_churn(df)))]), dbc.Alert("Month-to-month contracts show a wider spread of monthly charges for churned customers, often at higher rates. Longer contracts (One/Two year) have lower monthly charges for churned customers, and churn is less frequent within these contract types.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), ]),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Monthly Charges Density by Contract (Ridgeline)"), dbc.CardBody(dcc.Graph(id=f'{page_id}-ridge-monthly-contract', figure=viz.create_ridge_monthly_contract_churn(df)))]), dbc.Alert("This plot displays the density of monthly charges for each contract type, separated by churn status. It helps visualize where charges are concentrated for churners vs. non-churners within each contract agreement. Month-to-month contracts often show churners concentrated at higher monthly charges.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), dbc.Col([dbc.Card([dbc.CardHeader("Average Monthly Charge (Total Charges / Tenure) by Churn Status"), dbc.CardBody(dcc.Graph(id=f'{page_id}-box-avg-charge', figure=viz.create_box_avg_charge_churn(df)))]), dbc.Alert("The average monthly charge (calculated as Total Charges / Tenure) tends to be slightly higher for customers who churn, though there's considerable overlap, especially when considering outliers for non-churners.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), ]),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Churn Rate (%) by Monthly Charge Group"), dbc.CardBody(dcc.Graph(id=f'{page_id}-charge-group', figure=viz.create_bar_charge_group_churn(df)))]), dbc.Alert("Churn rate clearly increases with higher monthly charge groups. Customers in 'Very High' charge groups have the highest propensity to churn, suggesting price sensitivity or value perception issues at higher price points.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), dbc.Col([dbc.Card([dbc.CardHeader("Monthly Charges vs Total Charges (Colored by Tenure Group)"), dbc.CardBody(dcc.Graph(id=f'{page_id}-scatter-monthly-total-tenure', figure=viz.create_scatter_monthly_total_tenure(df)))]), dbc.Alert("This scatter plot illustrates how total charges naturally increase with monthly charges. The color coding by tenure group shows distinct bands, where customers with longer tenure (e.g., '5+ Years') are positioned higher on the Total Charges axis for any given Monthly Charge, reflecting accumulated payments over time.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), ]),
    ], fluid=True)

def create_layout_services_overview(df): 
    page_id = "services_overview"
    if df.empty: return dbc.Container([dbc.Alert("Error loading data.", color="danger")], fluid=True)
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("Services Overview & Churn Impact"), width=12, className="mb-4 text-center")),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Customer Count by Internet Service & Churn"), dbc.CardBody(dcc.Graph(id=f'{page_id}-internet-stack', figure=viz.create_stack_internet_churn(df)))]), dbc.Alert("Fiber optic internet service has a large number of customers, but also a proportionally high number of churners compared to DSL. Customers with no internet service have very low churn.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), dbc.Col([dbc.Card([dbc.CardHeader("Tenure Distribution by Internet Service & Churn"), dbc.CardBody(dcc.Graph(id=f'{page_id}-violin-tenure-internet', figure=viz.create_violin_tenure_internet_churn(df)))]), dbc.Alert("Customers with Fiber optic who churn tend to do so at earlier tenures compared to DSL customers. Those with no internet service who churn are very few and often at very low tenures.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), ]),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Monthly Charges Distribution by Key Service Subscriptions and Churn"), dbc.CardBody(dcc.Graph(id=f'{page_id}-box-services-subplots', figure=viz.create_box_services_subplots_monthly_churn(df)))]), dbc.Alert("This shows how monthly charges distribute for customers with/without key protective services, segmented by churn. It can reveal if churners with certain services are paying more or less than non-churners.", color="light", className="mt-2")], width=12, className="mb-4"), ]),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Churn Rate (%) by Number of Optional Services Subscribed To"), dbc.CardBody(dcc.Graph(id=f'{page_id}-num-optional', figure=viz.create_bar_num_services_churn(df)))]), dbc.Alert("Churn rate tends to be highest for customers with zero optional services (45%) and decreases as more optional services are subscribed to, indicating that more engaged customers are less likely to churn.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), dbc.Col([dbc.Card([dbc.CardHeader("Churn Rate (%) by Number of Protective Services (Internet Users)"), dbc.CardBody(dcc.Graph(id=f'{page_id}-num-protective', figure=viz.create_bar_prot_services_churn(df)))]), dbc.Alert("Among internet users, those with zero protective services (Online Security, Backup, Device Protection, Tech Support) have the highest churn rate (around 56%). The rate significantly decreases as more protective services are adopted.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), ]),
        dbc.Row(dbc.Col([dbc.Card([dbc.CardHeader("Customer Segments Treemap (Size=Count, Color=Avg Monthly $)"), dbc.CardBody(dcc.Graph(id=f'{page_id}-treemap', figure=viz.create_treemap_segments(df)))]), dbc.Alert("The treemap visually breaks down customer segments. For instance, Month-to-month Fiber Optic customers who churn represent a significant portion of the churned population and also have high average monthly charges.", color="light", className="mt-2")], width=12, className="mb-4")),
    ], fluid=True)

def create_layout_service_details(df):
    page_id = "service_details"
    if df.empty: return dbc.Container([dbc.Alert("Error loading data for Service Details page.", color="danger")], fluid=True)
    service_cols_for_radio = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    valid_service_cols_for_radio = [s for s in service_cols_for_radio if s in df.columns]
    default_service = valid_service_cols_for_radio[0] if valid_service_cols_for_radio else None
    service_selection_control = dbc.RadioItems(id=f'{page_id}-selector-radio', options=[{'label': s, 'value': s} for s in valid_service_cols_for_radio], value=default_service, inline=True, className="mb-3 mt-2", labelClassName="me-3", inputClassName="me-1")
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("Detailed Churn Analysis by Specific Service"), width=12, className="mb-4 text-center")),
        dbc.Row(dbc.Col(html.P("Select a service below to view its detailed churn rate breakdown."), width=12)),
        dbc.Row(dbc.Col(service_selection_control, width=12, className="mb-3")),
        dbc.Row(dbc.Col(dbc.Card([ dbc.CardHeader(id=f'{page_id}-plot-header'), dbc.CardBody(dcc.Graph(id=f'{page_id}-plot', figure=viz.create_subplots_services_churn(df, selected_service=default_service))) ]), width=12, lg=8, className="mx-auto")), 
        dbc.Row(dbc.Col(html.Div(id=f'{page_id}-interpretation'), width=12, lg=8, className="mx-auto mt-2")) 
    ], fluid=True)

def create_layout_advanced(df):
    page_id = "advanced"
    if df.empty: return dbc.Container([dbc.Alert("Error loading data.", color="danger")], fluid=True)
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("Advanced Visualizations & Segment Deep Dive"), width=12, className="mb-4 text-center")),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Correlation Heatmap (Numeric Features)"), dbc.CardBody(dcc.Graph(id=f'{page_id}-heatmap', figure=viz.create_heatmap_corr(df)))]), dbc.Alert("The heatmap shows correlations between numeric variables. 'Tenure' and 'TotalCharges' are strongly positively correlated (0.83). 'Churn_numeric' has a notable negative correlation with 'Tenure' (-0.35) and 'TotalCharges' (-0.20), meaning higher tenure/total charges are associated with lower churn. 'MonthlyCharges' has a positive correlation with 'Churn_numeric' (0.19).", color="light", className="mt-2")], width=12, lg=7, className="mb-4"), dbc.Col([dbc.Card([dbc.CardHeader("Distribution of Calculated Risk Score by Churn Status"), dbc.CardBody(dcc.Graph(id=f'{page_id}-risk-score', figure=viz.create_hist_risk_score_churn(df)))]), dbc.Alert("Customers who churn (Yes) tend to have a higher calculated Risk Score, with the distribution shifted to the right compared to non-churners. This suggests the risk factors used are indicative of churn.", color="light", className="mt-2")], width=12, lg=5, className="mb-4"), ]),
        dbc.Row([ dbc.Col([dbc.Card([dbc.CardHeader("Density Contour of Tenure vs Monthly Charges by Churn"), dbc.CardBody(dcc.Graph(id=f'{page_id}-density-contour', figure=viz.create_density_contour_churn(df)))]), dbc.Alert("This plot shows concentration areas. Churned customers often concentrate in regions of low tenure and varying monthly charges, particularly higher charges. Non-churned customers show density across a wider range of tenures, often with moderate monthly charges.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), dbc.Col([dbc.Card([dbc.CardHeader("3D Scatter Plot: Tenure, Monthly Charges, Total Charges by Churn"), dbc.CardBody(dcc.Graph(id=f'{page_id}-3d-scatter', figure=viz.create_scatter_3d(df)))]), dbc.Alert("The 3D scatter plot helps visualize the interplay of tenure, monthly charges, and total charges. Churned customers (red) often occupy a space characterized by lower tenure and lower total charges, but can have high monthly charges.", color="light", className="mt-2")], width=12, lg=6, className="mb-4"), ]),
        dbc.Row(dbc.Col([dbc.Card([dbc.CardHeader("Month-to-Month Fiber Optic: Monthly Charges by Tech Support & Churn"), dbc.CardBody(dcc.Graph(id=f'{page_id}-facet-segment', figure=viz.create_facet_segment_tech_support(df)))]), dbc.Alert("Within the high-risk segment of Month-to-Month Fiber Optic users, those without Tech Support show a higher churn count across various monthly charge levels compared to those with Tech Support.", color="light", className="mt-2")], width=12, className="mb-4")),
        dbc.Row(dbc.Col([dbc.Card([dbc.CardHeader("Tenure vs Monthly Charges - Faceted by Internet Service & Contract"), dbc.CardBody(dcc.Graph(id=f'{page_id}-facet-scatter', figure=viz.create_facet_scatter_tenure_monthly(df)))]), dbc.Alert("This faceted plot breaks down the tenure vs. monthly charges relationship. It clearly shows that Month-to-Month Fiber Optic customers have a high concentration of churn across all tenures, especially at higher monthly charges. Other contract types and internet services show different patterns, generally with lower churn.", color="light", className="mt-2")], width=12, className="mb-4")),
        dbc.Row(dbc.Col([dbc.Card([dbc.CardHeader("Churn Rate (%) by Payment Method (Faceted by Contract Type)"), dbc.CardBody(dcc.Graph(id=f'{page_id}-facet-payment', figure=viz.create_facet_payment_contract_churn(df)))]), dbc.Alert("Electronic check consistently shows higher churn rates across all contract types, but it's most pronounced for Month-to-Month contracts (around 55% churn). Longer contract types (One year, Two year) significantly mitigate the high churn associated with electronic checks, though it remains the highest churn payment method within those contracts as well.", color="light", className="mt-2")], width=12, className="mb-4")),
    ], fluid=True)

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server
def create_layout_predictor(df):
    page_id = "predict"
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("AI Churn Predictor: 'What-If' Scenario Simulator"), width=12, className="mb-4 text-center")),
        dbc.Row([
            # Input Controls Column
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Customer Profile Inputs", className="mb-4"),
                
                dbc.Label("Tenure (Months)"),
                dcc.Slider(id=f'{page_id}-tenure', min=0, max=72, step=1, value=12, tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                
                dbc.Label("Monthly Charges ($)"),
                dcc.Slider(id=f'{page_id}-monthly', min=18, max=120, step=1, value=70, tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                
                dbc.Label("Contract Type"),
                dcc.Dropdown(id=f'{page_id}-contract', options=[{'label': i, 'value': i} for i in ['Month-to-month', 'One year', 'Two year']], value='Month-to-month', clearable=False, className="mb-3"),
                
                dbc.Label("Internet Service"),
                dcc.Dropdown(id=f'{page_id}-internet', options=[{'label': i, 'value': i} for i in ['DSL', 'Fiber optic', 'No']], value='Fiber optic', clearable=False, className="mb-3"),
                
                dbc.Label("Tech Support"),
                dcc.Dropdown(id=f'{page_id}-tech', options=[{'label': i, 'value': i} for i in ['No', 'Yes', 'No internet service']], value='No', clearable=False, className="mb-3"),
                
                dbc.Label("Payment Method"),
                dcc.Dropdown(id=f'{page_id}-payment', options=[{'label': i, 'value': i} for i in ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']], value='Electronic check', clearable=False, className="mb-3"),
                
            ])), width=12, lg=5, className="mb-4"),
            
            # Gauge Output Column
            dbc.Col([
                dbc.Card(dbc.CardBody(dcc.Graph(id=f'{page_id}-gauge-chart'))),
                dbc.Alert("Adjust the parameters on the left to see how business decisions (like upgrading a customer to a 1-year contract or adding tech support) dynamically affect their probability of leaving the company.", color="info", className="mt-3")
            ], width=12, lg=7)
        ])
    ], fluid=True)
# --- Navigation Bar ---
PAGES = [
    {"module": "intro", "name": "Explorer", "layout_func": create_layout_intro_explorer, "path": "/"},
    {"module": "overview", "name": "Overview", "layout_func": create_layout_overview, "path": "/overview"},
    {"module": "financial", "name": "Financial", "layout_func": create_layout_financial, "path": "/financial"},
    {"module": "services_overview", "name": "Services", "layout_func": create_layout_services_overview, "path": "/services-overview"}, 
    {"module": "service_details", "name": "Details", "layout_func": create_layout_service_details, "path": "/service-details"}, 
    {"module": "advanced", "name": "Advanced", "layout_func": create_layout_advanced, "path": "/advanced"},
    {"module": "predict", "name": "AI Predictor", "layout_func": create_layout_predictor, "path": "/predictor"}, # NEW PAGE ADDED HERE
]
navbar = dbc.NavbarSimple(children=[dbc.NavItem(dbc.NavLink(page["name"], href=page["path"])) for page in PAGES], brand="Telecom Customer Churn Analysis", brand_href="/", color="primary", dark=True, className="mb-4", sticky="top")

# --- App Layout ---
app.layout = dbc.Container([dcc.Location(id='url', refresh=False), navbar, html.Div(id='page-content', className="mt-4")], fluid=True)

# --- Callbacks ---
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if df_global.empty: return html.Div(dbc.Alert("Fatal Error: Data loading failed.", color="danger", className="m-4"))
    for page in PAGES:
        if pathname == page["path"]: return page["layout_func"](df_global)
    return html.Div([html.H1("404: Not found", className="text-danger"), html.Hr(), html.P(f"The pathname {pathname} was not recognised...")], className="p-3 bg-light rounded-3 mt-4")

@app.callback(
    Output('intro-data-table', 'data'), 
    [Input('intro-search-input', 'value')], 
    prevent_initial_call=False) 
def update_intro_explorer_table(search_term): 
    if df_global.empty: return [] 
    filtered_df = df_global.copy()
    if search_term and not filtered_df.empty:
        search_term_lower = str(search_term).lower()
        searchable_cols = filtered_df.select_dtypes(include=['object', 'category']).columns
        if not searchable_cols.empty:
            mask = filtered_df[searchable_cols].apply(lambda row: row.astype(str).str.lower().str.contains(search_term_lower, na=False, regex=False).any(), axis=1)
            filtered_df = filtered_df[mask]
    return filtered_df.to_dict('records')

@app.callback(
    [Output('service_details-plot', 'figure'),
     Output('service_details-plot-header', 'children'),
     Output('service_details-interpretation', 'children')],
    [Input('service_details-selector-radio', 'value')],
    prevent_initial_call=False) 
def update_service_details_plot(selected_service):
    if df_global.empty:
        return viz.create_empty_figure("Data not loaded"), "Error", dbc.Alert("Data not loaded.", color="danger")
    if not selected_service: 
        service_cols_for_radio = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        valid_service_cols_for_radio = [s for s in service_cols_for_radio if s in df_global.columns]
        selected_service = valid_service_cols_for_radio[0] if valid_service_cols_for_radio else "All" 
    fig = viz.create_subplots_services_churn(df_global, selected_service=selected_service)
    header_text = f"Churn Rate Breakdown for: {selected_service}"
    interpretation_text = [ html.Strong(f"Insights for {selected_service}: "), html.P(f"This plot shows the percentage of customers who churned ('Yes') versus those who did not ('No'), based on whether they subscribe to {selected_service}. ") ]
    if selected_service in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']: interpretation_text.append(html.P(f"Typically, not having {selected_service} is associated with higher churn. Compare the red (Churn=Yes) portions of the 'No' and 'Yes' bars."))
    elif selected_service in ['StreamingTV', 'StreamingMovies']: interpretation_text.append(html.P(f"Subscription to entertainment services like {selected_service} can indicate customer engagement. Differences in churn rates between subscribers and non-subscribers might be observed."))
    else: interpretation_text.append(html.P(f"Churn rates for {selected_service} can vary. For 'MultipleLines', having the service might indicate a more invested customer, potentially with lower churn."))
    return fig, header_text, dbc.Alert(interpretation_text, color="light", className="mt-2")

@app.callback(
    [Output('financial-hist-monthly', 'figure'), Output('financial-scatter-tenure-total', 'figure')],
    [Input('financial-contract-dropdown', 'value'), Input('financial-payment-dropdown', 'value')],
    prevent_initial_call=False) 
def update_financial_graphs(selected_contract, selected_payment):
    if df_global.empty: no_data_fig = viz.create_empty_figure("Data not loaded"); return no_data_fig, no_data_fig
    filtered_df = df_global.copy(); title_suffix = ""
    if selected_contract != 'All' and 'Contract' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Contract'] == selected_contract]; title_suffix += f" (Contract: {selected_contract})"
    if selected_payment != 'All' and 'PaymentMethod' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['PaymentMethod'] == selected_payment]; title_suffix += f" (Payment: {selected_payment})"
    if filtered_df.empty:
         no_data_fig = viz.create_empty_figure(f'No Data for Selection{title_suffix}'); fig_hist = no_data_fig; fig_scatter = no_data_fig
    else:
        fig_hist = viz.create_hist_monthly(filtered_df, title_suffix=title_suffix); fig_scatter = viz.create_scatter_tenure_total(filtered_df, title_suffix=title_suffix)
    return fig_hist, fig_scatter
    
@app.callback(
    Output('predict-gauge-chart', 'figure'),
    [Input('predict-tenure', 'value'),
     Input('predict-monthly', 'value'),
     Input('predict-contract', 'value'),
     Input('predict-internet', 'value'),
     Input('predict-tech', 'value'),
     Input('predict-payment', 'value')]
)
def run_prediction(tenure, monthly, contract, internet, tech, payment):
    customer_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'Contract': contract,
        'InternetService': internet,
        'TechSupport': tech,
        'PaymentMethod': payment
    }
    
    # Run data through the Random Forest
    probability = predictor.predict(customer_dict)
    
    # Draw the gauge
    return viz.create_probability_gauge(probability)

if __name__ == "__main__":
    if df_global.empty:
        print("\n--- Cannot start server: Data loading failed. Check file path and format. ---")
    else:
        print("\n--- Starting Dash Server ---")
        app.run(debug=True)