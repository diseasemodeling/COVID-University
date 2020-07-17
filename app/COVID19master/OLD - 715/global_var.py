import pandas as pd
from datetime import datetime
import numpy as np
import pathlib
import json
import os



def setup_global_variables(state, inv_dt1, num_inf1, decision_making_date,
                           travel_num_inf1, test_sensitivity1, sim_week1, #unif1,
                           final_simul_start_date1, path, heroku = False):
    ##### do not change ######
    global tot_risk
    tot_risk = 2

    global tot_age
    tot_age = 101
    ##### do not change ######

    ##### user defined input #######
    global init_num_inf
    init_num_inf = num_inf1

    global enter_state
    enter_state = state

    global inv_dt
    inv_dt = inv_dt1

    global dt
    dt = 1/inv_dt
    ##### user defined input #######

    ##### read simulation input #######
    sim_result = read_sim_inputs(state = enter_state, path = path, heroku = heroku)
    global symp_hospitalization_v
    symp_hospitalization_v = sim_result[0]

    global percent_dead_recover_days_v
    percent_dead_recover_days_v = sim_result[1]

    # global pop_dist_v
    # pop_dist_v = sim_result[2]

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

    ###### read date related result ##########
    """date_result = read_date(state = enter_state)
    final_simul_start_date = date_result[0]

    global begin_decision_date
    begin_decision_date = date_result[2]

    global days_of_simul_pre_sd
    days_of_simul_pre_sd = date_result[3]

    global days_of_simul_post_sd
    days_of_simul_post_sd = date_result[4]

    global dry_run_end_diag
    dry_run_end_diag = date_result[5]

    global actual_data
    actual_data = date_result[6]
    """
    ##### read RL input #######
    # rl_result = read_RL_inputs(state = enter_state, start_sim_date = final_simul_start_date)
    rl_result = read_RL_inputs(state = enter_state, path=path, heroku=heroku)
    global VSL
    VSL = rl_result#[0]


    """global lab_for
    lab_for = rl_result[1]

    global K_val
    K_val = rl_result[2]

    global A_val
    A_val = rl_result[3]

    global duration_unemployment
    duration_unemployment = rl_result[4]

    global init_unemploy
    init_unemploy = rl_result[5]

    global acutal_unemp
    acutal_unemp = rl_result[6]

    global pre_results_dict
    global pre_results_df
    pre_results_dict, pre_results_df = read_pre_results(state)"""

    global T_max

    # global md_salary

    global test_cost

    global pop_dist_v
    global total_pop
    total_pop, pop_dist_v = read_pop_dist(state, prop=1, path = path, heroku = heroku)

    global day_decison_making
    day_decison_making = decision_making_date

    global travel_num_inf
    travel_num_inf = travel_num_inf1

    global SAR

    global test_sensitivity
    test_sensitivity = test_sensitivity1

    global sim_week
    sim_week = sim_week1

    #global unif
    #unif = unif1

    global final_simul_start_date
    final_simul_start_date = final_simul_start_date1

    # global pre_results_dict
    # global pre_results_df
    # pre_results_dict, pre_results_df = read_pre_results(state, path = path, heroku=heroku)

# Function to read pre-run results for simulation and plotting
# def read_pre_results(state):
#     if heroku == False:
#         f1 = os.path.join(path,'app\\COVID19master\\data\\results\\{0}_sim_results.json'.format(state))
#         f2 = os.path.join(path,'app\\COVID19master\\data\\results\\{0}_sim_results.xlsx'.format(state))
#     else:
#         f1 = os.path.join(path,'app/COVID19master/data/results/{0}_sim_results.json'.format(state))
#         f2 = os.path.join(path,'app/COVID19master/data/results/{0}_sim_results.xlsx'.format(state))

#     #f1 = pathlib.Path('data/results/{0}_sim_results.json'.format(state))
#     json_file = open(f1,)
#     data = json.load(json_file)
#     new_dict = {}
#     for key, value in data.items():
#         if isinstance(value, list):
#             new_dict[key] = np.array([[x] for x in value])
#         elif isinstance(value, str):
#             new_dict[key] = pd.Timestamp(value).date()
#         else:
#             new_dict[key] = value
#     json_file.close()

#     #f2 =  pathlib.Path('data/results/{0}_sim_results.xlsx'.format(state))
#     df = pd.ExcelFile(f2)
#     return new_dict, df

# Function to read population distribution by State /university
# Distribute age to a certain propotion of the total population
def read_pop_dist(state, prop, path, heroku = False):
    print(path)
    print(heroku)
    # excel = pathlib.Path('data/age_dist_univ.xlsx')
    if heroku == False:
        excel =  os.path.join(path,'app\\COVID19master\\data\\age_dist_univ.xlsx')
    else:
        excel = os.path.join(path,'app/COVID19master/data/age_dist_univ.xlsx')

    pop_dist = pd.read_excel(excel, sheet_name = state, index_col = 0)
    total_pop = pop_dist.sum().sum()
    total_pop_mod = total_pop * prop
    age_dist = pop_dist/total_pop
    pop_dist_mod = total_pop_mod * age_dist
    # travel_dist = 0.1 * pop_dist_mod   # 10 percent of total population

    pop_dist_mod['age'] = pop_dist_mod.index
    pop_dist_mod = pop_dist_mod[['age', 'female', 'male']]

    # travel_dist['age'] = travel_dist.index
    # travel_dist = travel_dist[['age', 'female', 'male']]

    pop_dist_mod_v = pop_dist_mod.values
    # travel_dist_v = travel_dist.values

    # return total_pop_mod, pop_dist_mod_v, travel_dist_v
    return total_pop_mod, pop_dist_mod_v


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


# Function to read date related data
def read_date(state):
    # the Excel files need to read
    excel1 = pathlib.Path('data/COVID_input_parameters.xlsx')
    excel2 = pathlib.Path('data/actual_valid_data.xlsx')

    # read social distancing start date
    sd_date = pd.read_excel(excel1, sheet_name='sd_date')

    # read actual epidemic data and filter by State, category
    raw_valid_data = pd.read_excel(excel2, sheet_name = 'Sheet1', index_col = 0)
    is_state = raw_valid_data['state'] == state
    valid_data = raw_valid_data[is_state]
    valid_data = valid_data.loc[valid_data['positive'] > 0]
    valid_data = valid_data.sort_values(by = ['positive'])

    # transform actual epidemic data
    valid_data_v = valid_data.loc[:, ('positive','death', 'hospitalized')]
    valid_data_v.index = pd.to_datetime(valid_data_v.index,  format='%Y%m%d', errors='coerce')
    valid_data_v.rename(columns= {'positive':'actual cumulative diagnosis', 'death':'actual cumulative deaths',\
                                  'hospitalized': 'actual cumulative hospitalized'}, inplace=True)

    # The beginning date where the number of infections exceed 0
    final_simul_start_date = pd.Timestamp(str(valid_data.first_valid_index())).date()

    # The date when decision making begins
    # begin_decision_date = pd.to_datetime('today')
    begin_decision_date = pd.Timestamp.today().date()

    # Min number of infections that need to be diagnosed for dry run to end
    dry_run_end_diag = int(valid_data.iloc[0]['positive'])

    # The date where social distancing begins in the state
    sd_start_date_state = sd_date[sd_date['state'] == state]['effective date'].values[0]
    sd_start_date_state = pd.Timestamp(sd_start_date_state).date()

    # The number of days before social distancing was introduced
    days_of_simul_pre_sd = abs((sd_start_date_state - final_simul_start_date).days)

    # The number of days after social distancing was introduced (before decision making)
    days_of_simul_post_sd = abs((sd_start_date_state - begin_decision_date).days) - 1

    return (final_simul_start_date, sd_start_date_state, begin_decision_date, \
            days_of_simul_pre_sd, days_of_simul_post_sd, dry_run_end_diag, valid_data_v)

# Returns 7 values:
# [0] = final_simul_start_date - the date when simulation begins (type: pandas timestamp)
# [1] = sd_start_date_state - the date when social distancing begins (type: pandas timestamp)
# [2] = begin_decision_date - the date when decision making begins (type: pandas timestamp)
# [3] = days_of_simul_pre_sd - difference between [0] and [1] (type: integer)
# [4] = days_of_simul_post_sd - difference between [1] and [2] (type: integer)
# [5] = dry_run_end_diag - number of diagnoses at end of dry run (type: integer)
# [6] = valid_data_v - actual epidemic data (type: DataFrame)


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
# state - State you want to model
# start_sim_date -  start date of simulation

# def read_RL_inputs(state, start_sim_date):
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
    # # read labor force participation rate
    # lab_for_v = df.parse(sheet_name='labor_for', index_col = 0)
    # val = lab_for_v.loc[state, 'Labor force participate rate']/100

    # # read maximum and minimum unemployment rate
    # cof_unemploy = df.parse(sheet_name='unemploy_cof_mod', index_col = 0)
    # K_val = cof_unemploy.loc[state, 'max']/100
    # A_val = cof_unemploy.loc[state, 'min']/100

    # # read actual unemployment rate
    # acutal_unemp = df.parse(sheet_name='actual_unemploy_mod')
    # acutal_unemp['Date'] = pd.to_datetime(acutal_unemp['Date'], format='%Y%m%d', errors='coerce')
    # acutal_unemp = acutal_unemp.loc[:, ('Date', state)]
    # acutal_unemp = acutal_unemp[(acutal_unemp['Date'] > pd.Timestamp(start_sim_date))]
    # acutal_unemp = acutal_unemp.reset_index(drop = True)
    # acutal_unemp[state] = acutal_unemp[state].astype(float)

    # # read initial unemployment rate
    # init_unemploy = acutal_unemp.loc[0, state]/100

    # acutal_unemp.rename(columns = {state: 'Actual unemployment rate'}, inplace = True)
    # acutal_unemp['Actual unemployment rate'] /= 100

    # # read duration from start of social distancing to the maximum of unemployment rate
    # dur = df.parse(sheet_name = 'duration_unemploy', index_col = 0)
    # dur = dur.loc[state]
    # max_date = pd.to_datetime(dur.loc['max_date'], format='%Y%m%d', errors='coerce')
    # sd_date = pd.to_datetime(dur.loc['sd_date'], format='%Y%m%d', errors='coerce')
    # duration_unemployment = abs(max_date - sd_date).days

    # read median wage and cost of testing by type
    # others = df.parse(sheet_name='others')
    # others_list = others.values.tolist()
    # md_salary = others_list[0][0]/8 *(40/7)
    # test_cost = others_list[0][1:]

    # return  VSL, val, K_val, A_val, duration_unemployment, init_unemploy, acutal_unemp


# Returns 8 values
# [0] = VSL - the Value of statistical life by age (type: NumPy array of size 101x1)
# [1] = val - Labor force participation rate, assumed constant (type: float)
# [2] = K_val - Maximum unemployment rate  (type: float)
# [3] = A_val - Minimum unemployment rate  (type: float)
# [4] = duration_unemployment - The duration between the start of social distancing to the maximum (type: integer)
# [5] = init_unemployment - The initial unemployment rate once the simulation starts (type: integer)
# [6] = acutal_unemp - actual unemployment rate (type: DataFrame)

# Function to read cost related data
# def read_cost(data_path):
#     df = pd.ExcelFile(data_path)
#     others = df.parse(sheet_name='Sheet1')
#     others_list = others.values.tolist()
#     md_salary = others_list[0][0]/8 *(40/7)
#     test_cost = others_list[0][1:]
#     return md_salary, test_cost
# Returns 2 values
# [0] = md_salary - coverted daily median wage (type: float)
# [1] = test_cost - A list of size 1x3 which is the cost of symptom-based testing,
#       contact tracing and universal testing (type: list)
