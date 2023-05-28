import streamlit as st
import numpy as np
import time

# cadCAD standard dependencies
from cadCAD.configuration.utils import config_sim
from cadCAD.configuration import Experiment
from cadCAD.engine import ExecutionMode, ExecutionContext
from cadCAD.engine import Executor
from cadCAD import configs

# Additional dependencies
import json
import requests as req
import math
import pandas as pd
import plotly.express as px

# Setup / Preparatory Steps
API_URI = 'https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2'

GRAPH_QUERY = '''
{
  markets {
    borrowRate
    supplyRate
    totalBorrows
    totalSupply
    exchangeRate
  }
}
'''

# Retrieve data from query
JSON = {'query': GRAPH_QUERY}
r = req.post(API_URI, json=JSON)
graph_data = json.loads(r.content)['data']

# Data Wrangle the Data
raw_df = pd.DataFrame(graph_data['markets'])
df = (raw_df.assign(totalBorrows=lambda df: pd.to_numeric(df.totalBorrows))
            .assign(totalSupply=lambda df: pd.to_numeric(df.totalSupply))
            .assign(borrowRate=lambda df: pd.to_numeric(df.borrowRate))
            .assign(supplyRate=lambda df: pd.to_numeric(df.supplyRate))
            .assign(exchangeRate=lambda df: pd.to_numeric(df.exchangeRate))
            .reset_index()
      )

# Modelling
# State Variables
initial_state = {
    'lender_APY': 0.0,
    'borrower_rate': 0.0,
    'utilization_rate': 0.0,
    'exchange_rate': 0.0
}

# System Parameters
df_dict = df.to_dict(orient='index')

system_params = {
    'new_df': [df_dict],
    'exchange_rate': ['exchangeRate']
}

# Policy Functions
def p_rates(params, substep, state_history, previous_state):
    t = previous_state['timestep']
    ts_data = params['new_df'][t]
    
    lender_APY = ts_data['supplyRate']
    borrower_rate = ts_data['borrowRate']
    exchange_rate = params['exchange_rate']
    total_borrowed = ts_data['totalBorrows']
    TVL = ts_data['totalSupply']

    try:
        utilization_rate = pd.to_numeric(total_borrowed) / pd.to_numeric(TVL) * 100
    except ZeroDivisionError:
        utilization_rate = 0

    return {
        'lender_APY': lender_APY,
        'borrower_rate': borrower_rate,
        'exchange_rate': exchange_rate,
        'utilization_rate': utilization_rate
    }

# State Update Functions
def s_lender_APY(params, substep, state_history, previous_state, policy_input):
    value = policy_input['lender_APY']
    return ('lender_APY', value)

def s_borrower_APY(params, substep, state_history, previous_state, policy_input):
    value = policy_input['borrower_rate']
    return ('borrower_rate', value)

def s_utilization_rate(params, substep, state_history, previous_state, policy_input):
    value = policy_input['utilization_rate']
    return ('utilization_rate', value)

def s_exchange_rate(params, substep, state_history, previous_state, policy_input):
    value = policy_input['exchange_rate']
    return ('exchange_rate', value)

# Partial State Update Blocks
partial_state_update_blocks = [
    {
        'policies': {
            'policy_rates': p_rates
        },
        'variables': {
            's_lender_APY': s_lender_APY,
            's_borrower_APY': s_borrower_APY,
            's_utilization_rate': s_utilization_rate,
            's_exchange_rate': s_exchange_rate
        }
    }
]

# Configuration Dictionary
config = {
    'T': range(len(df_dict)),  # Set number of iterations
    'N': 1,  # Set number of monte carlo runs
    'M': system_params,
    'initial_state': initial_state,
    'partial_state_update_blocks': partial_state_update_blocks
}

exp = Experiment()
exp.append_configs(config)

exec_mode = ExecutionMode()
exec_context = ExecutionContext()

simulation = Executor(exec_context=exec_context, configs=configs)
raw_result, tensor = simulation.execute()
result = pd.DataFrame(raw_result)

# Plotting
fig = px.line(result, x='timestep', y=['lender_APY', 'borrower_rate', 'utilization_rate', 'exchange_rate'],
              labels={'timestep': 'Time Step', 'value': 'Value', 'variable': 'Variable'})
fig.show()

# Streamlit code
st.title("cadCAD Simulation with Streamlit")

# Sidebar widgets
run_simulation = st.sidebar.button("Run Simulation")
re_run = st.sidebar.button("Re-run")

if run_simulation:
    # Display progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Execute cadCAD simulation
    exec_mode = ExecutionMode()
    exec_context = ExecutionContext()

    simulation = Executor(exec_context=exec_context, configs=configs)
    raw_result, tensor = simulation.execute()
    result = pd.DataFrame(raw_result)

    # Update progress bar and status text
    for i in range(len(result)):
        status_text.text("%i%% Complete" % ((i + 1) / len(result) * 100))
        progress_bar.progress((i + 1) / len(result) * 100)
        time.sleep(0.05)

    progress_bar.empty()

    # Display output table
    st.subheader("Simulation Results")
    st.write(result)

# Chart section (unchanged)
st.subheader("Simulation Chart")
chart = st.line_chart()

if run_simulation:
    for i in range(len(result)):
        new_rows = result.iloc[i, :]
        chart.add_rows(new_rows)
        time.sleep(0.05)

# Re-run button (unchanged)
if re_run:
    st.experimental_rerun()