import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template
import dash_dangerously_set_inner_html
import plotly.express as px
import plotly.graph_objects as go

world_data = pd.read_excel('./data/WorldBank.xlsx')

# Add Population (M) Column - Using Millions to make it easier to read
world_data['Population (M)'] = (world_data['GDP (USD)'] / world_data['GDP per capita (USD)']) / 1000000

hdi_data = pd.read_csv('./data/HDI.csv')

data_2014 = pd.merge(
    world_data.query('Year  in ([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014])'), 
    hdi_data[['iso3',
              'hdi_2000',
              'hdi_2001',
              'hdi_2002',
              'hdi_2003',
              'hdi_2004',
              'hdi_2005',
              'hdi_2006',
              'hdi_2007',
              'hdi_2008',
              'hdi_2009',
              'hdi_2010',
              'hdi_2011',
              'hdi_2012',
              'hdi_2013',                         
              'hdi_2014']], 
    left_on='Country Code', 
    right_on='iso3', 
    how='left')
# Remove unneeded column
data_2014.drop('iso3',axis=1,inplace=True)

gdp_pivot = world_data.pivot_table(
    index = 'Year',
    columns = 'Region',
    values = 'GDP (USD)',   
    aggfunc = 'sum',
)
gdp_pivot.reset_index(inplace=True)

pop_pivot = world_data.pivot_table(
    index = 'Year',
    columns = 'Region',
    values = 'Population (M)',       
    aggfunc = 'sum',
)
pop_pivot.reset_index(inplace=True)

data_region = data_2014.groupby('Region').agg(
    # I added other years as a test for further development
    # Typically we would be using just 'hdi_2014'
    {
        'hdi_2000': 'mean',
        'hdi_2001': 'mean',
        'hdi_2002': 'mean',
        'hdi_2003': 'mean',
        'hdi_2004': 'mean',
        'hdi_2005': 'mean',
        'hdi_2006': 'mean',
        'hdi_2007': 'mean',
        'hdi_2008': 'mean',
        'hdi_2009': 'mean',
        'hdi_2010': 'mean',
        'hdi_2011': 'mean',
        'hdi_2012': 'mean',
        'hdi_2013': 'mean',
        'hdi_2014': 'mean'},
)

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

region_dict = {
    'East Asia & Pacific':'#F94144',
    'Europe & Central Asia':'#F3722C',
    'Latin America & Caribbean':'#F8961E',
    'Middle East & North Africa':'#F9C74F',
    'North America':'#90BE6D',
    'South Asia':'#43AA8B',
    'Sub-Saharan Africa':'#577590',
}
population_dict = {
    '250':0.2,
    '500':0.4,
    '750':0.6,
    '1000':0.8,
    '1250':1.0
}
# set colours

coolwarm_custom = [
    "#b40426",
    "#ff8f7c",
    "#ffe1a3", 
    "#a8d0ff", 
    "#6090ff",     
    "#3b4cc0", 
]  # Colors range from red to blue


# Read in descriptive text from files.

with open("gdp_text.txt", "r", encoding="utf-8") as file:
    gdp_text = file.read()

with open("life_expectancy_text.txt", "r", encoding="utf-8") as file:
    life_expenctancy_text = file.read()

with open("hdi_text.txt", "r", encoding="utf-8") as file:
    hdi_text = file.read()


app = Dash(
    __name__, external_stylesheets=[dbc.themes.PULSE, dbc_css]
)  # Temporarily commented out so tabs show somw shading if not selected

server = app.server 
# We set the minsize and maxsize of the poplutation for the legend of the life expectany bubble chart
minsize = min(data_2014["Population (M)"])  # / 10
if minsize < 15:
    minsize = 15
maxsize = max(data_2014["Population (M)"]) / 20

load_figure_template("PULSE")

# Function to create the population legend
def create_population_legend():
    # Create a list of columns based on the population_dict
    cols = []

    for pop, factor in population_dict.items():
        # Circle for each population factor
        circle = html.Div(
            style={
                "width": f"{maxsize * factor}px",
                "height": f"{maxsize * factor}px",
                "background-color": "#080808",
                "border-radius": "50%",
                "display": "inline-block",
                "margin": "0 auto",  # Ensures the circle is centered within the column
            }
        )
        
        # Add the circle and the population value to the columns list
        cols.append(
            dbc.Col(
                children=[
                    # Create a flexbox container to align the circle and text horizontally
                    html.Div(
                        children=[
                            circle,
                            html.H6(str(pop), style={'display': 'inline-block', 'text-align': 'center', 'margin-top':'10px','margin-left': '10px', 'line-height': '1.5'})
                        ],
                        style={
                            "display": "flex",  # Align elements horizontally in a row
                            "align-items": "center",  # Vertically centre both circle and text
                            "justify-content": "center",  # Centre them horizontally within the row
                        }
                    )
                ],
                width="auto",
                style={"display": "inline-block", "text-align": "center"}
            )
        )
    
    # Return a row containing the generated columns, ensuring the entire legend is centered
    return dbc.Row(
        dbc.Card(
            dbc.Row(cols, align="center", justify="start"),
            style={"border": "none"},
        ),
        justify="start",  # Ensures everything is aligned at the start of the row
    )




# Function to create the region legend
def create_region_legend():
    return [
        dbc.Row(
            dbc.Col(
                [
                    html.Div(
                        style={
                            'margin-left': '10px',
                            "width": '20px',
                            "height": '10px',
                            "background-color": color,
                            "display": "inline-block",
                        }
                    ),
                    html.Div(
                        region,
                        style={"display": "inline-block", 'margin-left': '20px'},
                    )
                ],
                style={'margin-bottom': '0px', 'margin-top': '0px'}
            )            
        )
        for region, color in region_dict.items()
    ]


# Main layout structure
app.layout = dbc.Container(
    children=[
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            id="report_title",
                            className="bg-primary text-white p-2 mb-2 text-center",
                        )
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dcc.Markdown("Select A Region:"),
                                dcc.Dropdown(
                                    id="region_dropdown",
                                    options=[
                                        {"label": region, "value": region}
                                        for region in data_2014["Region"].unique()
                                    ],
                                    value="all_values",
                                    multi=True,
                                    className="dbc",
                                ),
                            ],
                            style={"border": "none"},
                        )
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dcc.Markdown("Select A Country"),
                                dcc.Dropdown(
                                    id="country_dropdown",
                                    options=[
                                        {"label": country, "value": country}
                                        for country in data_2014["Country Name"].unique()
                                    ],
                                    value="all_values",
                                    multi=True,
                                    className="dbc",
                                ),
                            ],
                            style={"border": "none"},
                        )
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dcc.Markdown("Select A Year"),
                                dcc.Dropdown(
                                    id="year_dropdown",
                                    options=[
                                        {"label": str(year), "value": year}
                                        for year in data_2014["Year"].unique()
                                    ],
                                    value=2014,
                                    multi=True,
                                    className="dbc",
                                ),
                            ],
                            style={"border": "none"},
                        )
                    ]
                ),
            ]
        ),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H6('GDP Has Grown Exponentially Over Time')
                ],
                style={
                        'border': 'none',
                        'margin-top':'10px'
                      },
                )
            ],
                width=4
            ),
            dbc.Col([
                dbc.Card([
                    html.H6(id='pop_title',children='Population has Surged from 2 Billion to 7.5 Billion',className='card-title')
                ],
                style={
                        'border': 'none',
                        'margin-top':'10px'
                      },
                )
            ],
                width=4
            ),
        ]),
        dbc.Row([            
            dbc.Col([
                dbc.Card([
                    dcc.Graph(id="gdp_chart")
                ],
                style={"border": "none",'margin-left':'0px'},
                )
            ],
            width = 4),            
            dbc.Col([
                dbc.Card([                     
                     dcc.Graph(id="pop_chart")
                 ],
                 style={"border": "none", 'margin-left':'0px'},
                ),
            ],
            width = 4),
            dbc.Col([
                dbc.Card(
                    children = create_region_legend(),  # Insert generated legend rows here
                    style={'margin-top': '0px', 'margin-left':'10px','margin-bottom': '10px'}
                     
                ),
                dbc.Row([
                    dbc.Card([
                        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(gdp_text)
                        ],
                        style={
                        'margin-bottom': '10px', 
                        'margin-top': '10px',   
                        'margin-left': '10px',
                        'border': 'none'}
                    ),
                ],
                style={'border': 'none'}
                ),
            ],
            width = 4
            ),
        ],
        style={'margin-top': '0px'}
        ),
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    html.H6(
                        "Life Expectancy Increases as Countries get richer",
                        className="card-title",
                    ),
                    style={'border': 'none'}
                )
            ],
            width = 9    
            ),
        ]),
        dbc.Col([],
            width = 3
        ),
        dbc.Row([            
            dbc.Col(
                html.H6("Population (M)", className="card-title"),
                #width = 'auto',
                width = 3,
                style={"display": "inline-block",'margin-bottom': '10px'},
            ),
            dbc.Col([    
                dbc.Card(
                    children=create_population_legend(),  # Call the function to generate the legend
                    style={
                        'margin-top': '0px', 
                        'margin-left': '10px', 
                        'margin-bottom': '10px',      
                        'border': 'none'}  
                ),                
            ],                 
            #width = 'auto'
            width = 6, 
            style={"display": "inline-block","border": "none"}),
            dbc.Col([],
                width = 3
            ),
        ], align="center"),  # Vertically center content in the row
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="life_expectancy_chart")
            ],
            width = 9
            ),
            dbc.Col([
                dbc.Card([
                        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(life_expenctancy_text)
                        ],
                        style={
                        'margin-bottom': '10px', 
                        'margin-top': '0px',                        
                        'border': 'none'}
                        )
                ])
            ]),   
        dbc.Row([            
            dbc.Col([
                dbc.Card([
                    html.H6('HDI by Region')
                ],
                style={'border': 'none'}
                ),
            ],
                width = 4
            ),
            dbc.Col([
                dbc.Card([
                    html.H6('Electricity Drives Development')
                ],
                style={'border': 'none'}
                ),
            ],
                width = 4
            ),
            dbc.Col([
                dbc.Card([                    
                ],
                style={'border': 'none'}         
                ),
            ],
                width = 4
            ),
        ]),
        dbc.Row([            
            dbc.Col([
                dbc.Card([
                    dcc.Graph(
                        id='hdi_region_chart',
                        config={"responsive": True}
                    )
                    ],
                    style={"border": "none"},
                    ),                            
                
            ],
                width = 4
            ),
            dbc.Col([
                dbc.Card([
                    dcc.Graph(id='electricity_chart')
                ],
                    style={'border': 'none'}
                ),                                
            ],
                width = 4
            ),
            dbc.Col([
                dbc.Card([  
                    dash_dangerously_set_inner_html.DangerouslySetInnerHTML(hdi_text)
                ],
                    style={'border': 'none'}
                 ),                                    
            ],
                width = 4       
            ),                              
            ]                
        )        
    ]
)


@app.callback(Output("country_dropdown", "options"), Input("region_dropdown", "value"))
def update_country_dropdown(selected_regions):
    # If no region is selected or 'all_values' is selected, return all countries
    if not selected_regions or "all_values" in selected_regions:
        return [
            {"label": country, "value": country}
            for country in data_2014["Country Name"].unique()
        ]

    # Filter the countries based on the selected region(s)
    filtered_countries = data_2014[data_2014["Region"].isin(selected_regions)][
        "Country Name"
    ].unique()
    return [{"label": country, "value": country} for country in filtered_countries]


@app.callback(
    Output('report_title', 'children'),
    Output('gdp_chart', 'figure'),
    Output('pop_chart', 'figure'),
    Output('life_expectancy_chart', 'figure'),
    Output('hdi_region_chart','figure'),
    Output('electricity_chart','figure'),
    Output('pop_title', 'children'),
    Input('region_dropdown', 'value'),
    Input('country_dropdown', 'value'),
    Input('year_dropdown', 'value'),
)
def display_infomation(region_dropdown_value, country_dropdown_value, year_dropdown_value):

    hdi_column = f"hdi_{year_dropdown_value}"
    title = "Tracing Global Growth and Development, 1960 – 2018"

    # If the region list is empty then assign all_values to the value for further checks.
    if region_dropdown_value == []:
        region_dropdown_value = "all_values"

    # Filter the data based on dropdown values
    if region_dropdown_value == "all_values" or region_dropdown_value is None:
        dataframe = data_2014.query(f"Year == {year_dropdown_value}")
    else:
        dataframe = data_2014[
            data_2014["Region"].str.contains("|".join(region_dropdown_value))
        ].query(f"Year == {year_dropdown_value}")

    if country_dropdown_value != "all_values" and country_dropdown_value is not None:
        dataframe = dataframe[
            dataframe["Country Name"].str.contains("|".join(country_dropdown_value))
        ]

    columns_to_keep = dataframe.columns[:16].tolist()
    if hdi_column in dataframe.columns:
        columns_to_keep.append(hdi_column)

    dataframe = dataframe[columns_to_keep].reset_index(drop=True)

    # Initialize gdp_pivot_filtered to the default gdp_pivot
    gdp_pivot_filtered = gdp_pivot.copy()

    # Filter the GDP Pivot table to the selected region
    if region_dropdown_value == "all_values" or region_dropdown_value is None:
        gdp_pivot_filtered = gdp_pivot.reset_index(drop=True)  # Reset index
    else:
        columns_to_keep = ["Year"]  # Always include the 'Year' column
        for region in region_dropdown_value:
            if region in gdp_pivot.columns:
                columns_to_keep.append(region)
        gdp_pivot_filtered = gdp_pivot[columns_to_keep].reset_index(
            drop=True
        )  # Reset index

    gdp_pivot_filtered.iloc[:, 1:] = (
        gdp_pivot_filtered.iloc[:, 1:] / 1_000_000_000_000
    )  # Divide all GDP values by 1 trillion


    gdp_long = gdp_pivot_filtered.melt(
        id_vars="Year", 
        var_name="Region", 
        value_name="GDP (USD)")
    gdp_long['GDP (USD)'] = gdp_long['GDP (USD)'] / 1_000_000_000_000

    color_sequence = [region_dict[region] for region in gdp_pivot_filtered.columns[1:]]
    
        
    # Create the GDP chart
    gdp_fig = px.area(
        gdp_long,
        #gdp_pivot_filtered,
        #[gdp_pivot_filtered[region] / 1_000_000_000_000 for region in gdp_pivot_filtered.iloc[-1].sort_values(ascending=False).index],
        x="Year",  # Specify 'Year' as x-axis
        y='GDP (USD)',
        color='Region',            
        labels={"value": "GDP (Trillions)", "variable": "Region"},  # Custom labels
        title=None
    )
    

    for trace in gdp_fig.data:
        region_name = trace.name  # Each trace's name corresponds to a region
        if region_name in region_dict:
            colour = region_dict[region_name]
            trace.line.color = colour         # Set line color to match the region
            trace.marker.color = colour       # Ensure marker colour also matches (acts as line edge)
            trace.fillcolor = colour          # Set fill colour to dictionary colour

    trace.update(line=dict(color=colour), fillcolor=colour) 
    gdp_fig.update_layout(showlegend=False)

    gdp_fig.update_layout(yaxis_title="GDP (Trillions)", xaxis_title="Year", showlegend=False)
    gdp_fig.update_xaxes(showgrid=False)
    gdp_fig.update_yaxes(showgrid=False)
    # Initialize pop_pivot_filtered to the default pop_pivot

    pop_pivot_filtered = pop_pivot.copy()

    # Filter the GDP Pivot table to the selected region
    if region_dropdown_value == "all_values" or region_dropdown_value is None:
        pop_pivot_filtered = pop_pivot.reset_index(drop=True)  # Reset index
    else:
        columns_to_keep = ["Year"]  # Always include the 'Year' column
        for region in region_dropdown_value:
            if region in pop_pivot.columns:
                columns_to_keep.append(region)
        pop_pivot_filtered = pop_pivot[columns_to_keep].reset_index(
            drop=True
        )  # Reset index


    pop_long = pd.melt(
        pop_pivot_filtered,
        id_vars=['Year'],             # The 'Year' column will remain fixed
        var_name='Region',            # New column that will store region names
        value_name='Population'       # New column that will store population values
    )   
    
    first_year = pop_long["Year"].min()
    final_year = pop_long["Year"].max()

    # Get total population for the first and last years    
    total_first_year = (
        pop_long[pop_long["Year"] == first_year]
        ["Population"].sum() / 1_000
    )
    total_final_year = (
        pop_long[pop_long["Year"] == final_year]
        ["Population"].sum() / 1_000
    )

    # Set the wording in the title
    if (total_final_year - total_first_year >= 0) and (
        total_final_year - total_first_year < 0.6
    ):
        title_phrase = "Grown"
    elif total_final_year - total_first_year < 0:
        title_phrase = "Reduced"
    else:
        title_phrase = "Surged"

    # Create the Population chart
    pop_title=f"Population has {title_phrase} from {total_first_year:,.1f} Billion to {total_final_year:,.1f} Billion"
    pop_fig = px.area(
        pop_long,
        x="Year",
        y="Population",
        color="Region",   
        title=None
    )    

    for trace in pop_fig.data:
        region_name = trace.name
        if region_name in region_dict:
            colour = region_dict[region_name]
            trace.line.color = colour         # Set line color to match the region
            trace.marker.color = colour       # Ensure marker colour also matches
            trace.fillcolor = colour          # Set fill color to match region_dict


    pop_fig.update_xaxes(showgrid=False)
    pop_fig.update_yaxes(showgrid=False)
    pop_fig.update_layout(yaxis_title="Population (Billions)", xaxis_title="Year",showlegend=False)
    
  
    # Create the bubble chart
    minsize = min(data_2014["Population (M)"])
    maxsize = max(data_2014["Population (M)"])
    
    # Drop NaN values from the dataframe before using them in the scatter plot
    dataframe = dataframe.dropna(subset=["Population (M)"])
    dataframe = dataframe.rename(columns={hdi_column: "HDI"})
    dataframe = dataframe.query(f"Year == {year_dropdown_value}")

   
    life_fig = px.scatter(
        dataframe,
        x="Life expectancy at birth (years)",
        y="GDP per capita (USD)",
        size="Population (M)",
        size_max=2.0 * maxsize / 50,
        color="Region",
        #color_discrete_map=colour_map,
        color_discrete_map=region_dict,
    )
    life_fig.update_yaxes(
        title_text="GDP per capita (USD)", tickmode="linear", showgrid=False
    )
    life_fig.update_xaxes(showgrid=False)
    life_fig.update_layout(
        yaxis_type="log",
        showlegend=False                
    )
    
    data_region_filtered = data_region.copy()     
    data_region_filtered = data_region_filtered[hdi_column].reset_index()    
    data_region_filtered = data_region_filtered.rename(columns={hdi_column: 'HDI'})
    data_region_filtered = data_region_filtered.sort_values('HDI',ascending=False)
    
    hdi_fig = px.bar(
        data_region_filtered,
        y = data_region_filtered['Region'],
        x = 'HDI',
        color='Region',
        color_discrete_map=region_dict,
        title=None,
    )
    hdi_fig.update_layout(        
        xaxis_title='HDI',
        xaxis_tickformat='.0%',
        margin=dict(l=20, r=20, t=40, b=40),
        showlegend=False,
        width=400,
        bargap=0.3,
        yaxis={'visible': True, 'showticklabels': True},
        xaxis={'visible': False, 'showticklabels': False}
    )
    hdi_fig.update_xaxes(
        showgrid=False,    
        range=[0,1.1],
        visible=False)
    hdi_fig.update_yaxes(
        title=None,    # Remove the y-axis title
        showline=True,
        showticklabels=True  # Keep the tick labels visible
    )
    hdi_fig.update_yaxes(
        showgrid=False
    )
    hdi_fig.update_traces(
        texttemplate='%{x:.0%}',  # Format labels as whole percentages
        textposition='outside'  # Position labels outside the bars
    )       
    
    dataframe = dataframe.query(f'`Country Name` != "Iceland" and Year == {year_dropdown_value}')
    max_gdp = max(dataframe['GDP per capita (USD)']) * 0.7 # To limit the yaxis as the default value is too high at 200K.        
    electric_fig = px.scatter(
    dataframe,
    x='Electric power consumption (kWh per capita)',
    y='GDP per capita (USD)',
    color='HDI',
    title=None,
    color_continuous_scale=coolwarm_custom  # Use custom coolwarm-like scale
)

    # Adjust y-axis range & turn of gridlines
    electric_fig.update_yaxes(
        range=[0, max_gdp],        
        showgrid=False
    )
    electric_fig.update_xaxes(
        showgrid=False
    )
    # Disable the default color bar legend
    electric_fig.update_layout(        
        coloraxis_showscale=False,
        legend_title="HDI Levels"
    )
    
    # Custom circular legend creation in 0.4 to 0.9 range with 0.1 steps
    legend_steps = np.arange(0.4, 1.0, 0.1)  # Values from 0.4 to 0.9 in 0.1 steps
    legend_steps = sorted(legend_steps, reverse=True)  # Reverse order for descending display
    legend_y_position = 1.15  # Adjust as needed
    
    # Add each legend marker manually using go.Scatter
    for i, value in enumerate(legend_steps):
        value = round(value * 10,0) / 10
        electric_fig.add_trace(
            go.Scatter(
                x=[None],  # Dummy data for legend circles
                y=[None],
                mode='markers',
                marker=dict(                    
                    size=15,
                    color=coolwarm_custom[i],  # Color based on HDI, matching custom scale
                    opacity=0.7
                ),                
                name=f'{value * 100:.1f}%',  # Show value as a percentage                
            )
        )
    
    # Adjust layout for the legend
    legend=dict(
        title='HDI Levels',
        orientation='v',  # Set the orientation to vertical
        yanchor='top',
        y=0.5,  # Center the legend vertically
        xanchor='left',
        x=1.1,  # Position the legend to the right of the plot
        traceorder='normal'
    )   

    return title, gdp_fig, pop_fig, life_fig, hdi_fig, electric_fig,pop_title


if __name__ == "__main__":    
    #app.run_server(debug=True, mode="inline", port=8658) # Used for deployment in Jupyter Notebook
    app.run_server(debug=True) # Used for deployment to Docker

