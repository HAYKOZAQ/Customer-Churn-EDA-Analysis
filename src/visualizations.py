import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import math
import re

# --- Constants ---
COLOR_DISCRETE_MAP_CHURN = {'Yes': '#E57373', 'No': '#81C784'}
THEME_COLORS_QUALITATIVE = px.colors.qualitative.Pastel
PLOT_TEMPLATE = 'plotly_white'

def create_empty_figure(title="No data available for the current selection"):
    fig = go.Figure()
    fig.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'}, xaxis={'visible': False}, yaxis={'visible': False}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', template=PLOT_TEMPLATE)
    return fig

def create_hist_monthly(df, title_suffix=""):
    if df is None or df.empty or 'MonthlyCharges' not in df.columns or 'Churn' not in df.columns: return create_empty_figure(f'No data for Monthly Charges Distribution{title_suffix}')
    fig = px.histogram(df, x='MonthlyCharges', color='Churn', barmode='overlay', marginal='box', title=f'Distribution of Monthly Charges by Churn{title_suffix}', opacity=0.75, color_discrete_map=COLOR_DISCRETE_MAP_CHURN, template=PLOT_TEMPLATE)
    fig.update_layout(xaxis_title='Monthly Charges ($)', yaxis_title='Number of Customers', legend_title_text='Churn'); return fig

def create_scatter_tenure_total(df, title_suffix=""):
    required_cols = ['tenure', 'TotalCharges', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure(f'No data for Tenure vs Total Charges{title_suffix}')
    hover_cols = ['customerID', 'MonthlyCharges', 'Contract']; valid_hover_cols = [col for col in hover_cols if col in df.columns]
    fig = px.scatter(df, x='tenure', y='TotalCharges', color='Churn', title=f'Tenure vs Total Charges by Churn{title_suffix}', opacity=0.6, hover_data=valid_hover_cols, color_discrete_map=COLOR_DISCRETE_MAP_CHURN, template=PLOT_TEMPLATE)
    fig.update_layout(xaxis_title='Tenure (Months)', yaxis_title='Total Charges ($)', legend_title_text='Churn'); fig.update_traces(marker=dict(size=5)); return fig

def create_bar_contract_churn(df):
    required_cols = ['Contract', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data for Churn Rate by Contract')
    if not isinstance(df['Contract'].dtype, pd.CategoricalDtype): df['Contract'] = df['Contract'].astype('category')
    churn_by_contract = df.groupby('Contract', observed=False)['Churn'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    churn_yes = churn_by_contract[churn_by_contract['Churn'] == 'Yes']
    if churn_yes.empty: return create_empty_figure('No "Yes" churn data for Contracts')
    fig = px.bar(churn_yes, x='Contract', y='percentage', title='Churn Rate (%) by Contract Type', color='Contract', text='percentage', color_discrete_sequence=THEME_COLORS_QUALITATIVE, template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside'); max_perc = churn_yes['percentage'].max()
    fig.update_layout(yaxis_title='Churn Percentage (%)', xaxis_title='Contract Type', uniformtext_minsize=8, uniformtext_mode='hide', yaxis_range=[0, max(10, max_perc * 1.15)]); return fig

def create_stack_internet_churn(df):
    required_cols = ['InternetService', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data for Internet Service vs Churn')
    if not isinstance(df['InternetService'].dtype, pd.CategoricalDtype): df['InternetService'] = df['InternetService'].astype('category')
    internet_churn = df.groupby(['InternetService', 'Churn'], observed=False).size().reset_index(name='count')
    if internet_churn.empty: return create_empty_figure('No grouped data for Internet Service vs Churn')
    fig = px.bar(internet_churn, x='InternetService', y='count', color='Churn', title='Customer Count by Internet Service and Churn Status', barmode='stack', color_discrete_map=COLOR_DISCRETE_MAP_CHURN, template=PLOT_TEMPLATE)
    fig.update_layout(xaxis_title='Internet Service Type', yaxis_title='Number of Customers', legend_title_text='Churn'); return fig

def create_subplots_demo_churn(df):
    required_cols = ['gender', 'SeniorCitizen', 'Dependents', 'Churn'];
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('Missing demographic or churn data')
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Gender', 'Senior Citizen', 'Has Dependents'), shared_yaxes=True)
    added_legend_for_demo = {'No': False, 'Yes': False} 
    def add_demo_trace_with_text(fig_obj, data_grouped, col_idx):
        nonlocal added_legend_for_demo
        for i, trace_status in enumerate(['No', 'Yes']): 
            if trace_status in data_grouped.columns:
                show_legend_flag = (col_idx == 1) and (not added_legend_for_demo[trace_status])
                texts = [f"{y:.1f}%" if y > 0 else "" for y in data_grouped[trace_status]]
                fig_obj.add_trace(go.Bar(name=f'Churn={trace_status}', x=data_grouped.index, y=data_grouped[trace_status], text=texts, textposition='inside', textfont=dict(color='white' if (trace_status == 'Yes' and data_grouped[trace_status].mean() > 25) or (trace_status == 'No' and data_grouped[trace_status].mean() < 25 and data_grouped[trace_status].mean() > 10) else 'black', size=10), marker_color=COLOR_DISCRETE_MAP_CHURN[trace_status], showlegend=show_legend_flag ), row=1, col=col_idx)
                if show_legend_flag: added_legend_for_demo[trace_status] = True
    if 'gender' in df.columns: gender_churn = df.groupby('gender', observed=False)['Churn'].value_counts(normalize=True).mul(100).unstack().fillna(0); add_demo_trace_with_text(fig, gender_churn, 1)
    if 'SeniorCitizen' in df.columns: senior_churn = df.groupby('SeniorCitizen', observed=False)['Churn'].value_counts(normalize=True).mul(100).unstack().fillna(0); add_demo_trace_with_text(fig, senior_churn, 2)
    if 'Dependents' in df.columns: dep_churn = df.groupby('Dependents', observed=False)['Churn'].value_counts(normalize=True).mul(100).unstack().fillna(0); add_demo_trace_with_text(fig, dep_churn, 3)   
    fig.update_layout(title_text='Churn Rate (%) by Demographics', barmode='stack', yaxis_title='Percentage (%)', height=450, template=PLOT_TEMPLATE, legend_title_text='Churn Status', uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_yaxes(range=[0, 100]); return fig

def create_heatmap_corr(df):
    if df is None or df.empty: return create_empty_figure('No data for correlation heatmap')
    numeric_cols_initial = df.select_dtypes(include=np.number).columns.tolist()
    if 'Churn_numeric' in df.columns and 'Churn_numeric' not in numeric_cols_initial: numeric_cols_initial.append('Churn_numeric')
    numeric_cols = [col for col in numeric_cols_initial if 'ID' not in col.lower()]
    if not numeric_cols: return create_empty_figure('No suitable numeric columns found for correlation')
    numeric_df = df[numeric_cols]
    numeric_df_std = numeric_df.std()
    numeric_df = numeric_df.loc[:, numeric_df_std[numeric_df_std > 1e-6].index]
    if numeric_df.empty or numeric_df.shape[1] < 2: return create_empty_figure('Not enough numeric data with variance for correlation')
    corr = numeric_df.corr(); fig = px.imshow(corr, text_auto=".2f", aspect="auto", title='Correlation Heatmap of Numeric Features', color_continuous_scale='RdBu_r', template=PLOT_TEMPLATE)
    num_vars = len(corr.columns); fig_height = max(450, num_vars * 40); fig_width = max(550, num_vars * 50)
    fig.update_layout(height=fig_height, width=fig_width, xaxis_tickangle=45, yaxis_automargin=True, xaxis_automargin=True, margin=dict(l=100, r=50, t=80, b=150)); return fig

def create_bar_payment_churn(df):
    required_cols = ['PaymentMethod', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data for Payment Method vs Churn')
    if not isinstance(df['PaymentMethod'].dtype, pd.CategoricalDtype): df['PaymentMethod'] = df['PaymentMethod'].astype('category')
    churn_by_payment_full = df.groupby('PaymentMethod', observed=False)['Churn'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    churn_yes = churn_by_payment_full[churn_by_payment_full['Churn'] == 'Yes'] 
    if churn_yes.empty: return create_empty_figure('No "Yes" churn data for Payment Methods')
    fig = px.bar(churn_yes, x='PaymentMethod', y='percentage', title='Churn Rate (%) by Payment Method', color='PaymentMethod', text='percentage', color_discrete_sequence=THEME_COLORS_QUALITATIVE, template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside') 
    max_perc = churn_yes['percentage'].max() if not churn_yes.empty else 10 
    fig.update_layout(yaxis_title='Churn Percentage (%)', xaxis_title='Payment Method', uniformtext_minsize=8, uniformtext_mode='hide', yaxis_range=[0, max(10, max_perc * 1.15)], xaxis_tickangle=-30, showlegend=True); return fig

def create_bar_paperless_churn(df):
    required_cols = ['PaperlessBilling', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data for Paperless Billing vs Churn')
    if not isinstance(df['PaperlessBilling'].dtype, pd.CategoricalDtype): df['PaperlessBilling'] = df['PaperlessBilling'].astype('category')
    churn_by_paperless = df.groupby('PaperlessBilling', observed=False)['Churn'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    churn_yes = churn_by_paperless[churn_by_paperless['Churn'] == 'Yes']
    if churn_yes.empty: return create_empty_figure('No "Yes" churn data for Paperless Billing')
    fig = px.bar(churn_yes, x='PaperlessBilling', y='percentage', title='Churn Rate (%) by Paperless Billing Status', color='PaperlessBilling', color_discrete_map={'Yes': '#FFB74D', 'No': '#64B5F6'}, text='percentage', template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside'); max_perc = churn_yes['percentage'].max()
    fig.update_layout(yaxis_title='Churn Percentage (%)', xaxis_title='Paperless Billing', uniformtext_minsize=8, uniformtext_mode='hide', yaxis_range=[0, max(10, max_perc * 1.15)]); return fig

def create_hist_tenure_churn(df):
    required_cols = ['tenure', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data for Tenure Distribution')
    fig = px.histogram(df, x='tenure', color='Churn', barmode='overlay', marginal='box', title='Distribution of Tenure by Churn Status', opacity=0.75, color_discrete_map=COLOR_DISCRETE_MAP_CHURN, template=PLOT_TEMPLATE)
    fig.update_layout(xaxis_title='Tenure (Months)', yaxis_title='Number of Customers', legend_title_text='Churn'); return fig

def create_subplots_services_churn(df, selected_service="All"):
    if df is None or df.empty: return create_empty_figure('No data for Service Subscription Analysis')
    service_cols_all = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    plot_service_cols = []
    if selected_service != "All" and selected_service in service_cols_all and selected_service in df.columns:
        plot_service_cols = [selected_service]; n_cols = 1; n_rows = 1; plot_height = 450 
    else: 
        selected_service = "All" 
        plot_service_cols = [s for s in service_cols_all if s in df.columns]
        if not plot_service_cols: return create_empty_figure('No valid service columns found')
        n_cols = 4; n_rows = math.ceil(len(plot_service_cols) / n_cols); plot_height = 280 * n_rows + 70
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=plot_service_cols, shared_yaxes=True, vertical_spacing=0.3 if n_rows > 1 else 0.15, horizontal_spacing=0.08)
    legend_added_map = {'No': False, 'Yes': False}
    for r_idx, service_chunk in enumerate([plot_service_cols[i:i + n_cols] for i in range(0, len(plot_service_cols), n_cols)]):
        for c_idx, service in enumerate(service_chunk):
            current_row = r_idx + 1; current_col = c_idx + 1
            categories_to_remove = ['No phone service', 'No internet service']
            temp_df_service = df[~df[service].isin(categories_to_remove)].copy()
            if not temp_df_service.empty:
                if isinstance(temp_df_service[service].dtype, pd.CategoricalDtype): temp_df_service[service] = temp_df_service[service].cat.remove_unused_categories()
                else:
                    valid_cats = [cat for cat in ['No', 'Yes'] if cat in temp_df_service[service].unique()]
                    if valid_cats: temp_df_service[service] = pd.Categorical(temp_df_service[service], categories=valid_cats, ordered=False); temp_df_service.dropna(subset=[service], inplace=True)
                    else: fig.add_annotation(text=f"No 'Yes'/'No' Data for {service}", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, row=current_row, col=current_col); continue
                if temp_df_service.empty: fig.add_annotation(text=f"No Data for {service}", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, row=current_row, col=current_col); continue
                churn_by_service = temp_df_service.groupby(service, observed=False)['Churn'].value_counts(normalize=True).mul(100).unstack().fillna(0)
                final_x_categories = [cat for cat in ['No', 'Yes'] if cat in churn_by_service.index]
                if not final_x_categories: fig.add_annotation(text=f"No 'Yes'/'No' Data for {service}", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, row=current_row, col=current_col); continue
                churn_by_service = churn_by_service.reindex(final_x_categories).fillna(0)
                for churn_status in ['No', 'Yes']:
                    if churn_status in churn_by_service.columns:
                        show_legend_flag = not legend_added_map[churn_status]
                        texts = [f"{y:.1f}%" if y > 1 else "" for y in churn_by_service[churn_status]]
                        fig.add_trace(go.Bar(name=f'Churn={churn_status}', x=churn_by_service.index, y=churn_by_service[churn_status], text=texts, textposition='inside', textfont=dict(color='white' if (churn_status == 'Yes' and churn_by_service[churn_status].mean() > 20) or (churn_status == 'No' and churn_by_service[churn_status].mean() < 20 and churn_by_service[churn_status].mean() > 10) else 'black', size=10), marker_color=COLOR_DISCRETE_MAP_CHURN[churn_status], showlegend=show_legend_flag), row=current_row, col=current_col)
                        if show_legend_flag: legend_added_map[churn_status] = True
            else: fig.add_annotation(text=f"No Data for {service}", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, row=current_row, col=current_col)
    title_text = f'Churn Rate (%) by {selected_service}' if selected_service != "All" else 'Churn Rate (%) by Service Subscription'
    fig.update_layout(title_text=title_text, height=plot_height, barmode='stack', uniformtext_minsize=8, uniformtext_mode='hide', template=PLOT_TEMPLATE, legend_title_text='Churn Status', margin=dict(t=80, b=50, l=70, r=40))
    fig.update_yaxes(title_text='Percentage (%)', range=[0, 100], row=1, col=1)
    if n_cols > 1 or n_rows > 1:
        for r_val in range(1, n_rows + 1):
            for c_val in range(1, n_cols + 1):
                 fig.update_yaxes(range=[0, 100], row=r_val, col=c_val)
    return fig

def create_violin_monthly_contract_churn(df):
    required_cols = ['MonthlyCharges', 'Contract', 'Churn'];
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data for Monthly Charges by Contract')
    hover_cols = ['tenure', 'TotalCharges']; valid_hover_cols = [col for col in hover_cols if col in df.columns]
    fig = px.violin(df, y="MonthlyCharges", x="Contract", color="Churn", box=True, points="all", title="Monthly Charges Distribution by Contract Type and Churn", hover_data=valid_hover_cols, color_discrete_map=COLOR_DISCRETE_MAP_CHURN, template=PLOT_TEMPLATE, violinmode='group')
    fig.update_traces(points="all", jitter=0.05, pointpos=0, marker=dict(size=2, opacity=0.7)) 
    fig.update_layout(yaxis_title="Monthly Charges ($)", xaxis_title="Contract Type", legend_title_text='Churn'); return fig

def create_violin_tenure_internet_churn(df):
    required_cols = ['tenure', 'InternetService', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data for Tenure by Internet Service')
    hover_cols = ['MonthlyCharges', 'TotalCharges']; valid_hover_cols = [col for col in hover_cols if col in df.columns]
    fig = px.violin(df, y="tenure", x="InternetService", color="Churn", box=True, points=False, title="Tenure Distribution by Internet Service and Churn", hover_data=valid_hover_cols, color_discrete_map=COLOR_DISCRETE_MAP_CHURN, template=PLOT_TEMPLATE, violinmode='group')
    fig.update_layout(yaxis_title="Tenure (Months)", xaxis_title="Internet Service Type", legend_title_text='Churn'); return fig

def create_bar_tenure_group_churn(df):
    required_cols = ['TenureGroup', 'Churn'];
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data or TenureGroup column available')
    if not isinstance(df['TenureGroup'].dtype, pd.CategoricalDtype) or not df['TenureGroup'].cat.ordered:
        labels_tenure = ['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years'];
        try: df['TenureGroup'] = pd.Categorical(df['TenureGroup'], categories=labels_tenure, ordered=True)
        except Exception as e: print(f"Error casting TenureGroup: {e}"); return create_empty_figure('Error processing TenureGroup data')
    df_plot = df.dropna(subset=['TenureGroup', 'Churn'])
    if df_plot.empty: return create_empty_figure('No valid data for Tenure Groups')
    churn_by_group = df_plot.groupby('TenureGroup', observed=False)['Churn'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    churn_yes = churn_by_group[churn_by_group['Churn'] == 'Yes']
    if churn_yes.empty: return create_empty_figure('No "Yes" churn data for Tenure Groups')
    fig = px.bar(churn_yes, x='TenureGroup', y='percentage', title='Churn Rate (%) by Customer Tenure Group', color='TenureGroup', text='percentage', color_discrete_sequence=px.colors.sequential.Viridis, template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside'); max_perc = churn_yes['percentage'].max()
    fig.update_layout(yaxis_title='Churn Percentage (%)', xaxis_title='Tenure Group', uniformtext_minsize=8, uniformtext_mode='hide', yaxis_range=[0, max(10, max_perc * 1.15)]); return fig

def create_box_services_subplots_monthly_churn(df):
    required_cols = ['MonthlyCharges', 'Churn', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('Missing data for Service Box Plots')
    services_for_box = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    plot_df_base = df[df['InternetService'] != 'No'].copy()
    if plot_df_base.empty: return create_empty_figure('No internet users to analyze for these services.')
    n_cols = 2; n_rows = math.ceil(len(services_for_box) / n_cols)
    valid_subplot_titles = [s for s in services_for_box if s in plot_df_base.columns]
    if not valid_subplot_titles: return create_empty_figure('None of the specified services for box plots are available.')
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=valid_subplot_titles, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.2, horizontal_spacing=0.1)
    current_row = 1; current_col = 1; legend_added = False
    for service in valid_subplot_titles: 
        plot_df_service = plot_df_base[ (plot_df_base[service] != 'No internet service') & (plot_df_base[service].isin(['Yes', 'No'])) ].copy()
        if not plot_df_service.empty:
            plot_df_service[service] = pd.Categorical(plot_df_service[service], categories=['No', 'Yes'], ordered=True)
            df_no_churn = plot_df_service[plot_df_service['Churn'] == 'No']
            df_yes_churn = plot_df_service[plot_df_service['Churn'] == 'Yes']
            if not df_no_churn.empty: fig.add_trace(go.Box(x=df_no_churn['MonthlyCharges'], y=df_no_churn[service], name='No Churn', marker_color=COLOR_DISCRETE_MAP_CHURN['No'], orientation='h', boxpoints='outliers', showlegend=not legend_added), row=current_row, col=current_col)
            if not df_yes_churn.empty: fig.add_trace(go.Box(x=df_yes_churn['MonthlyCharges'], y=df_yes_churn[service], name='Churn', marker_color=COLOR_DISCRETE_MAP_CHURN['Yes'], orientation='h', boxpoints='outliers', showlegend=(not legend_added and current_row == 1 and current_col == 1)), row=current_row, col=current_col)
            if not legend_added and (not df_no_churn.empty or not df_yes_churn.empty): legend_added = True 
        else: fig.add_annotation(text="No Data for this service category", xref="paper", yref="paper", x= (current_col - 0.5) / n_cols, y= 1 - (current_row - 0.5) / n_rows, showarrow=False, row=current_row, col=current_col)
        current_col += 1
        if current_col > n_cols: current_col = 1; current_row += 1
    fig.update_layout(title_text='Monthly Charges Distribution by Key Service Subscription and Churn', height=300 * n_rows + 100, template=PLOT_TEMPLATE, boxmode='group', showlegend=True, legend_title_text="Churn Status", margin=dict(l=100, r=30, t=80, b=50))
    fig.update_yaxes(title_text="Service Subscribed", categoryorder='array', categoryarray=['No', 'Yes'])
    fig.update_xaxes(title_text="Monthly Charges ($)") 
    if not fig.data: return create_empty_figure("No data to display for Service Box Plots")
    return fig

def create_ridge_monthly_contract_churn(df):
    if df is None or df.empty or 'Contract' not in df.columns or 'Churn' not in df.columns or 'MonthlyCharges' not in df.columns: return create_empty_figure('Missing data for Ridgeline Plot')
    fig_ridge = go.Figure(); contracts = sorted(df['Contract'].unique()); colors = px.colors.qualitative.Vivid 
    y_categories = [];
    for contract in contracts: y_categories.append(f'{contract} - Churn'); y_categories.append(f'{contract} - No Churn')
    y_categories.reverse() 
    for i, contract in enumerate(contracts):
        df_no = df[(df['Contract'] == contract) & (df['Churn'] == 'No')]; df_yes = df[(df['Contract'] == contract) & (df['Churn'] == 'Yes')]
        y_val_no_churn = f'{contract} - No Churn'; y_val_churn = f'{contract} - Churn'
        if not df_no.empty: fig_ridge.add_trace(go.Violin(x=df_no['MonthlyCharges'], y0=y_val_no_churn, name=y_val_no_churn, side='positive', orientation='h', points=False, scalemode='width', line_color=colors[i*2 % len(colors)], fillcolor=colors[i*2 % len(colors)], opacity=0.7, showlegend=True ))
        if not df_yes.empty: fig_ridge.add_trace(go.Violin(x=df_yes['MonthlyCharges'], y0=y_val_churn, name=y_val_churn, side='positive', orientation='h', points=False, scalemode='width', line_color=colors[(i*2+1) % len(colors)], fillcolor=colors[(i*2+1) % len(colors)], opacity=0.7, showlegend=True ))
    fig_ridge.update_traces(meanline_visible=True, width=0.8)
    fig_ridge.update_layout(title_text="Monthly Charges Distribution Density by Contract Type & Churn (Ridgeline)", xaxis_zeroline=False, xaxis_title="Monthly Charges ($)", yaxis_title="Contract Type & Churn Status", height=max(400, len(y_categories) * 50 + 100), template=PLOT_TEMPLATE, violingap=0, violingroupgap=0.2, yaxis=dict(categoryorder='array', categoryarray=y_categories))
    if not fig_ridge.data: return create_empty_figure("No data to display for Ridgeline Plot")
    return fig_ridge

def create_facet_scatter_tenure_monthly(df):
    required_cols = ['tenure', 'MonthlyCharges', 'Churn', 'InternetService', 'Contract']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('Missing data for Faceted Scatter Plot')
    hover_cols = ['PaymentMethod']; valid_hover_cols = [col for col in hover_cols if col in df.columns]
    fig = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn", facet_col="InternetService", facet_row="Contract", title="Tenure vs Monthly Charges - Faceted by Internet Service & Contract", height=700, color_discrete_map=COLOR_DISCRETE_MAP_CHURN, category_orders={"Contract": ["Month-to-month", "One year", "Two year"], "InternetService": ["No", "DSL", "Fiber optic"]}, hover_data=valid_hover_cols, template=PLOT_TEMPLATE)
    fig.update_layout(xaxis_title="Tenure (Months)", yaxis_title="Monthly Charges ($)", legend_title_text='Churn', margin=dict(l=60, r=30, t=80, b=60))
    fig.update_traces(marker=dict(size=4, opacity=0.6)); fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])); return fig

def create_bar_num_services_churn(df):
    required_cols = ['NumOptionalServices', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data for Number of Optional Services')
    churn_by_num_services = df.groupby('NumOptionalServices')['Churn'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    churn_yes = churn_by_num_services[churn_by_num_services['Churn'] == 'Yes']
    if churn_yes.empty: return create_empty_figure('No "Yes" churn data for Number of Optional Services')
    fig = px.bar(churn_yes, x='NumOptionalServices', y='percentage', title='Churn Rate (%) by Number of Optional Services Subscribed To', text='percentage', color='NumOptionalServices', color_continuous_scale=px.colors.sequential.YlGn, labels={'NumOptionalServices': 'Number of Optional Services'}, template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside'); max_perc = churn_yes['percentage'].max()
    fig.update_layout(yaxis_title='Churn Percentage (%)', xaxis_title='Number of Optional Services', yaxis_range=[0, max(10, max_perc * 1.15)]); return fig

def create_treemap_segments(df):
    required_cols = ['Contract', 'InternetService', 'Churn', 'MonthlyCharges'];
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('Missing data for Treemap Segments')
    treemap_df = df.copy(); treemap_df['customer_count'] = 1
    segment_agg = treemap_df.groupby(['Contract', 'InternetService', 'Churn'], observed=False).agg(customer_count=('customer_count', 'sum'), avg_monthly_charge=('MonthlyCharges', 'mean')).reset_index()
    if segment_agg.empty: return create_empty_figure('No segments found for Treemap')
    fig = px.treemap(segment_agg, path=[px.Constant("All Customers"), 'Contract', 'InternetService', 'Churn'], values='customer_count', color='avg_monthly_charge', hover_data={'avg_monthly_charge':':.2f'}, color_continuous_scale='YlGnBu', title='Treemap of Customer Segments by Contract & Internet Service (Size=Count, Color=Avg Monthly Charge)', template=PLOT_TEMPLATE)
    fig.update_traces(textinfo="label+percent parent"); fig.update_layout(height=600, margin=dict(t=50, l=25, r=25, b=25)); return fig

def create_scatter_3d(df):
     required_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'];
     if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('Missing data for 3D Scatter Plot')
     fig = px.scatter_3d(df, x='tenure', y='MonthlyCharges', z='TotalCharges', color='Churn', title='3D Scatter Plot: Tenure, Monthly Charges, Total Charges by Churn', labels={'tenure':'Tenure', 'MonthlyCharges':'Monthly $', 'TotalCharges':'Total $'}, color_discrete_map=COLOR_DISCRETE_MAP_CHURN, opacity=0.7, template=PLOT_TEMPLATE, height=600)
     fig.update_layout(margin=dict(l=10, r=10, b=10, t=50), legend_title_text='Churn'); return fig

def create_box_avg_charge_churn(df):
    required_cols = ['tenure', 'TotalCharges', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('Missing data for Average Monthly Charge')
    df_tenure_gt_0 = df[df['tenure'] > 0].copy()
    if df_tenure_gt_0.empty: return create_empty_figure('No customers with tenure > 0')
    df_tenure_gt_0['AvgMonthlyChargePerTenure'] = df_tenure_gt_0['TotalCharges'] / df_tenure_gt_0['tenure']
    fig = px.box(df_tenure_gt_0, x='Churn', y='AvgMonthlyChargePerTenure', color='Churn', title='Average Monthly Charge (Total Charges / Tenure) by Churn Status', labels={'AvgMonthlyChargePerTenure': 'Average Monthly Charge over Tenure ($)'}, points="outliers", color_discrete_map=COLOR_DISCRETE_MAP_CHURN, template=PLOT_TEMPLATE)
    fig.update_layout(xaxis_title='Churn Status', yaxis_title='Avg. Monthly Charge ($)', legend_title_text='Churn'); return fig

def create_facet_segment_tech_support(df):
    required_cols = ['Contract', 'InternetService', 'MonthlyCharges', 'TechSupport', 'Churn'];
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('Missing data for High-Churn Segment Analysis')
    high_churn_segment_df = df[(df['Contract'] == 'Month-to-month') & (df['InternetService'] == 'Fiber optic')].copy()
    if high_churn_segment_df.empty: return create_empty_figure('No data for Month-to-Month Fiber Optic segment')
    fig = px.histogram(high_churn_segment_df, x="MonthlyCharges", color="Churn", facet_col="TechSupport", barmode='overlay', marginal="box", title='Month-to-Month Fiber Optic: Monthly Charges by Tech Support & Churn', labels={'TechSupport': 'Has Tech Support'}, color_discrete_map=COLOR_DISCRETE_MAP_CHURN, category_orders={"TechSupport": ["No", "Yes"]}, template=PLOT_TEMPLATE)
    fig.update_layout(xaxis_title='Monthly Charges ($)', yaxis_title='Count', legend_title_text='Churn'); fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])); return fig

def create_density_contour_churn(df):
    required_cols = ['tenure', 'MonthlyCharges', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('Missing data for Density Contour Plot')
    fig = px.density_contour(df, x="tenure", y="MonthlyCharges", color="Churn", marginal_x="histogram", marginal_y="histogram", title="Density Contour of Tenure vs Monthly Charges by Churn", labels={'tenure':'Tenure (Months)', 'MonthlyCharges':'Monthly Charges ($)'}, color_discrete_map=COLOR_DISCRETE_MAP_CHURN, template=PLOT_TEMPLATE)
    fig.update_layout(legend_title_text='Churn'); return fig

def create_hist_risk_score_churn(df):
    required_cols = ['RiskScore', 'Churn']
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data or RiskScore available')
    fig = px.histogram(df, x='RiskScore', color='Churn', barmode='overlay', marginal='box', title='Distribution of Calculated Risk Score by Churn Status', labels={'RiskScore': 'Customer Risk Score (Higher = More Risk Factors)'}, opacity=0.75, color_discrete_map=COLOR_DISCRETE_MAP_CHURN, template=PLOT_TEMPLATE)
    fig.update_layout(xaxis_title='Risk Score', yaxis_title='Number of Customers', legend_title_text='Churn'); return fig

def create_bar_prot_services_churn(df):
    required_cols = ['NumProtectiveServices', 'Churn', 'InternetService'];
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data or NumProtectiveServices available')
    df_internet_users = df[df['InternetService'] != 'No'].copy()
    if df_internet_users.empty: return create_empty_figure('No Internet Users Found for Protective Service Analysis')
    churn_by_prot_services = df_internet_users.groupby('NumProtectiveServices')['Churn'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    churn_yes = churn_by_prot_services[churn_by_prot_services['Churn'] == 'Yes']
    if churn_yes.empty: return create_empty_figure('No "Yes" churn data for Protective Services')
    fig = px.bar(churn_yes, x='NumProtectiveServices', y='percentage', title='Churn Rate (%) by Number of Protective Services (Internet Users)', text='percentage', labels={'NumProtectiveServices': 'Number of Protective Services'}, color='NumProtectiveServices', color_continuous_scale=px.colors.sequential.Blues_r, template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside'); max_perc = churn_yes['percentage'].max()
    fig.update_layout(yaxis_title='Churn Percentage (%)', xaxis_title='Number of Protective Services', yaxis_range=[0, max(10, max_perc * 1.15)]); return fig

def create_bar_charge_group_churn(df):
    required_cols = ['MonthlyChargeGroup', 'Churn'];
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('No data or MonthlyChargeGroup available')
    if not isinstance(df['MonthlyChargeGroup'].dtype, pd.CategoricalDtype) or not df['MonthlyChargeGroup'].cat.ordered:
        try:
             def get_sort_key(label): label_str = str(label); numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', label_str)]; return min(numbers) if numbers else 0
             ordered_labels = sorted(df['MonthlyChargeGroup'].dropna().unique(), key=get_sort_key); df['MonthlyChargeGroup'] = df['MonthlyChargeGroup'].cat.set_categories(ordered_labels, ordered=True)
        except: print("Warning: Could not order MonthlyChargeGroup labels."); df['MonthlyChargeGroup'] = df['MonthlyChargeGroup'].cat.as_ordered()
    df_plot = df.dropna(subset=['MonthlyChargeGroup', 'Churn']);
    if df_plot.empty: return create_empty_figure('No data for charge groups')
    churn_by_charge_group = df_plot.groupby('MonthlyChargeGroup', observed=False)['Churn'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    churn_yes = churn_by_charge_group[churn_by_charge_group['Churn'] == 'Yes']
    if churn_yes.empty: return create_empty_figure('No "Yes" churn data for Monthly Charge Groups')
    fig = px.bar(churn_yes, x='MonthlyChargeGroup', y='percentage', title='Churn Rate (%) by Monthly Charge Group', text='percentage', color='MonthlyChargeGroup', color_discrete_sequence=px.colors.sequential.OrRd, template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside'); max_perc = churn_yes['percentage'].max()
    fig.update_layout(yaxis_title='Churn Percentage (%)', xaxis_title='Monthly Charge Group', uniformtext_minsize=8, uniformtext_mode='hide', yaxis_range=[0, max(10, max_perc * 1.15)]); return fig

def create_scatter_monthly_total_tenure(df):
    required_cols = ['MonthlyCharges', 'TotalCharges', 'TenureGroup', 'Churn', 'tenure'];
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('Missing data for Monthly vs Total Charges by Tenure')
    if not isinstance(df['TenureGroup'].dtype, pd.CategoricalDtype) or not df['TenureGroup'].cat.ordered:
         labels_tenure = ['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years'];
         try: df['TenureGroup'] = pd.Categorical(df['TenureGroup'], categories=labels_tenure, ordered=True)
         except Exception as e: print(f"Error casting TenureGroup: {e}"); return create_empty_figure('Error processing TenureGroup')
    df_plot = df[(df['tenure'] > 0) & (df['TenureGroup'].notna())].copy();
    if df_plot.empty: return create_empty_figure('No customers with tenure > 0 and valid TenureGroup')
    hover_cols = ['customerID', 'tenure']; valid_hover_cols = [col for col in hover_cols if col in df.columns]; valid_hover_cols.append('Churn')
    fig = px.scatter(df_plot, x='MonthlyCharges', y='TotalCharges', color='TenureGroup', title='Monthly Charges vs Total Charges (Colored by Tenure Group)', labels={'MonthlyCharges': 'Monthly Charges ($)', 'TotalCharges': 'Total Charges ($)'}, color_discrete_sequence=px.colors.sequential.Viridis, hover_data=valid_hover_cols, template=PLOT_TEMPLATE)
    fig.update_traces(marker=dict(size=5, opacity=0.7)); fig.update_layout(legend_title_text='Tenure Group'); return fig

def create_facet_payment_contract_churn(df):
    required_cols = ['Contract', 'PaymentMethod', 'Churn'];
    if df is None or df.empty or not all(col in df.columns for col in required_cols): return create_empty_figure('Missing data for Payment Method by Contract')
    churn_pay_contract = df.groupby(['Contract', 'PaymentMethod'], observed=False)['Churn'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    churn_yes = churn_pay_contract[churn_pay_contract['Churn'] == 'Yes']
    if churn_yes.empty: return create_empty_figure('No "Yes" churn data for Payment Method by Contract')
    fig = px.bar(churn_yes, x='PaymentMethod', y='percentage', color='PaymentMethod', facet_col='Contract', title='Churn Rate (%) by Payment Method (Faceted by Contract Type)', labels={'percentage': 'Churn Rate (%)'}, text='percentage', color_discrete_sequence=THEME_COLORS_QUALITATIVE, category_orders={"Contract": ["Month-to-month", "One year", "Two year"]}, template=PLOT_TEMPLATE)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside'); max_perc = churn_yes['percentage'].max()
    fig.update_layout(yaxis_title='Churn Percentage (%)', uniformtext_minsize=8, uniformtext_mode='hide', yaxis_range=[0, max(10, max_perc * 1.15)])
    fig.for_each_xaxis(lambda axis: axis.title.update(text="Payment Method")); fig.update_xaxes(tickangle=-30); fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])); return fig

def create_probability_gauge(probability):
    """Generates a speedometer gauge for the ML prediction."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Churn Probability", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Hide standard bar
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#81C784'},   # Green (Safe)
                {'range': [30, 70], 'color': '#FFB74D'},  # Orange (Warning)
                {'range': [70, 100], 'color': '#E57373'}  # Red (Danger)
            ],
            'threshold': {
                'line': {'color': "black", 'width': 6},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20), template=PLOT_TEMPLATE)
    return fig