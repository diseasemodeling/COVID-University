import pandas as pd
from datetime import datetime
import numpy as np
import pathlib
import json
import os

def setup_global_variables(state, inv_dt1, num_inf1, decision_making_date, 
                           travel_num_inf1, sim_week1, pop_size1, trans_prob1,
                           num_to_init_trace1, path, heroku = False):
                       
    ##### user defined input #######
    global init_num_inf
    init_num_inf = num_inf1

    global enter_state
    enter_state = state

    global inv_dt
    inv_dt = inv_dt1

    global dt
    dt = 1/inv_dt

    global trans_prob
    trans_prob = trans_prob1

    global T_max

    global test_cost

    global total_pop 
    total_pop = pop_size1

    global pop_dist_v
    pop_dist_v = read_pop_dist(state, pop_size1, path = path, heroku = heroku)

    global day_decison_making 
    day_decison_making = decision_making_date

    global travel_num_inf
    travel_num_inf = travel_num_inf1

    global num_to_init_trace
    num_to_init_trace = num_to_init_trace1

    #### calculate second attack rate
    global SAR
    SAR = -8.5806 * np.power(trans_prob,2) + 3.4568 * trans_prob + 0.0312

    global test_sensitivity
    test_sensitivity = 0.9

    global sim_week
    sim_week = sim_week1
    ##### user defined input #######

    global tot_risk
    tot_risk = 2

    global tot_age
    tot_age = 101

    ##### read simulation input #######
    sim_result = read_sim_inputs(state = enter_state, path = path, heroku = heroku)
    global symp_hospitalization_v
    symp_hospitalization_v = sim_result[0]

    global percent_dead_recover_days_v
    percent_dead_recover_days_v = sim_result[1]

    global input_list_const_v # dataframe
    input_list_const_v = sim_result[3]
    
    global Q
    Q = sim_result[4]

    # read beta value and scale factor
    beta_vals = sim_result[5]
    global beta_before_sd 
    beta_before_sd  = beta_vals[0]

    global beta_after_sd
    beta_after_sd = beta_vals[1]

    global hosp_scale
    hosp_scale = beta_vals[2]

    global dead_scale
    dead_scale  = beta_vals[3]

    # convert to transition rate matrix
    global rates_indices
    rates_indices = reading_indices(Q)

    global diag_indices
    diag_indices = diag_indices_loc(Q)

    ##### read RL input #######
    rl_result = read_RL_inputs(state = enter_state, path = path, heroku = heroku)
    global VSL
    VSL = rl_result#[0]

# Function to read population distribution by State /university
# Re-distribute the population size by age and gender 
# Input parameters: 
# state - State/University you want to model
# pop_size - population size you want to model
def read_pop_dist(state, pop_size, path, heroku = False):
    print(path)
    print(heroku)
    # excel = pathlib.Path('data/age_dist_univ.xlsx')
    if heroku == False:
        excel =  os.path.join(path,'app\\COVID19master\\data\\age_dist_univ.xlsx')
    else:
        excel = os.path.join(path,'app/COVID19master/data/age_dist_univ.xlsx')
    age_dist = pd.read_excel(excel, sheet_name = state, index_col = 0)
    pop_dist_mod = pop_size * age_dist
    pop_dist_mod['age'] = pop_dist_mod.index
    pop_dist_mod = pop_dist_mod[['age', 'female', 'male']]
    pop_dist_mod_v = pop_dist_mod.values
    return pop_dist_mod_v

# Function to read simulation related parameters
def read_sim_inputs(state, path, heroku=False):
    # the Excel files need to read
    # excel1= pathlib.Path('data/COVID_input_parameters.xlsx')
    # excel2 = pathlib.Path('data/pop_dist.xlsx')
    # excel3 = pathlib.Path('data/states_beta.xlsx')
    if heroku == False:
        excel1= os.path.join(path,'app\\COVID19master\\data\\COVID_input_parameters.xlsx')
        excel3 = os.path.join(path,'app\\COVID19master\\data\\states_beta.xlsx')
    else:
        excel1= os.path.join(path,'app/COVID19master/data/COVID_input_parameters.xlsx')
        excel3 = os.path.join(path,'app/COVID19master/data/states_beta.xlsx')

    # read blank Q-matrix
    q_mat_blank = pd.read_excel(excel1, sheet_name = 'q-mat_blank')
    q_mat_blank_v = q_mat_blank.values

    # read input paramteres for simulation
    input_list_const = pd.read_excel(excel1, sheet_name = 'input_list_const', index_col = 0)

    # read age related hospitalization probabilities
    symp_hospitalization = pd.read_excel(excel1, sheet_name='symp_hospitalization')
    symp_hospitalization_v = symp_hospitalization.values

    # read age and gender related death and recovery probabilities and time from day of hospitalization
    percent_dead_recover_days = pd.read_excel(excel1, sheet_name = 'percent_dead_recover_days')
    percent_dead_recover_days_v = percent_dead_recover_days.values

    # read population distribution of the State
    # pop_dist = pd.read_excel(excel2, sheet_name = state)
    # pop_dist_v = pop_dist.values

    pop_dist_v = 0  # dummy value

    # beta for the State
    states_betas = pd.read_excel(excel3, sheet_name = 'Sheet1', index_col = 0)
    beta_v = states_betas.loc[state]


    return (symp_hospitalization_v, percent_dead_recover_days_v,
            pop_dist_v, input_list_const, q_mat_blank_v, beta_v)

# Returns 6 values
# [0] = symp_hospitalization_v - the hospitalization data (type: NumPy array)
# [1] = percent_dead_recover_days_v - recovery and death data (type: NumPy array)
# [2] = pop_dist_v - inital population distribution of susceptible by age and risk group (type: NumPy array)
# [3] = input_list_const_v - list of input parameters (type: NumPy array)
# [4] = q_mat_blank_v - Blank transition rate matrix: there is a state transition where the rate is > 0 (type: NumPy array of size 10x10)
# [5] = beta_v - beta value for the State (max and min), hospitalization scale and death scale (type: DataFrame)


# Function to read and extract indices of the q mat where value is = 1
# Input parameters for this function
# Blank Q-matrix
def reading_indices(Q):
    rate_indices = np.where(Q == 1)
    list_rate = list(zip(rate_indices[0], rate_indices[1]))
    return list_rate
# Returns 1 value
# A list of length 16, represents the compartment flow; e.g. (0,1) represents 0 -> 1 (type: list)


# Function to extract indices of the diagonal of the q mat where value
# Input parameters for this function
# Blank Q-matrix
def diag_indices_loc(Q):
    mat_size = Q.shape
    diag_index = np.diag_indices(mat_size[0], mat_size[1])
    diag_index_fin = list(zip(diag_index[0], diag_index[1]))

    return diag_index_fin
# Returns 1 value
# An np array of size 10x1. Here we have a 10x10 q mat so we have 10 diagonal values.
# e.g. (0,0) represents 0 -> 0, (1,1) represents 1 -> 1


# Function to read RL input
# Input paramters:
# state - State/University you want to model
def read_RL_inputs(state, path, heroku=False):
    # read data
    # excel = pathlib.Path('data/RL_input.xlsx')
    if heroku == False:
        excel = os.path.join(path,'app\\COVID19master\\data\\RL_input.xlsx')
    else:
        excel = os.path.join(path, 'app/COVID19master/data/RL_input.xlsx')

    df = pd.ExcelFile(excel)
    # read VSL
    VSL1 = df.parse(sheet_name='VSL_mod')
    VSL2 = VSL1.to_numpy()
    VSL3 = np.transpose(VSL2)
    VSL = VSL3[:][1]

    return  VSL
# Returns 1 value
# [0] =  value of statistical life by age 0 - 100
    