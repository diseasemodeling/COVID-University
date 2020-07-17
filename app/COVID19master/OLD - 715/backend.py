import numpy as np
import pandas as pd
import time, os, json, copy

from app.COVID19master import global_varNew as gv
from app.COVID19master import COVID_model_colab as cov

def read_ABC_OLD(from_java):
    data = {'Date':{}, 'A':{}, 'B':{}, 'C':{}}
    policies = from_java['policy']
    for name, policy in policies.items():
        for plan, array in policy.items():
            data[plan][name] = array
    plans = ['A', 'B', 'C']
    policies = ['CR', 'CT', 'UT']
    dates = data['Date']['CT'] # hard coded
    days_per_tp = 7
    index = [0] + [i for i in dates for j in range(days_per_tp)]
    rl_input = {}
    for plan in plans:
        rl_input[plan] = pd.DataFrame(index=index, columns=policies)
        for policy, changes in data[plan].items():
            to_rl = [0]
            for change in changes:
                to_rl += [change] * days_per_tp
            rl_input[plan][policy] = to_rl
        rl_input[plan].iloc[0] = rl_input[plan].iloc[1].values
        rl_input[plan] = rl_input[plan].values
        if np.sum(rl_input[plan]==0) == len(rl_input[plan])*3:
            rl_input.pop(plan)
    return rl_input

def read_ABC(from_java):
    data = {'A':{}, 'B':{}, 'C':{}}
    policies = from_java['policy']
    num_days = int(from_java['lenSim'])
    for name, policy in policies.items():
        for plan, array in policy.items():
            data[plan][name] = array
    plans = ['A', 'B', 'C']
    policies = ['CR', 'MT', 'TT']
    index = [i for i in range(num_days)]
    rl_input = {}
    for plan in plans:
        rl_input[plan] = pd.DataFrame(index=index, columns=policies)
        for policy, value in data[plan].items():
            if policy in policies:
                rl_input[plan][policy] = value
            rl_input[plan] = rl_input[plan].values
    return rl_input



# Funtion for one scenario analysis
def main_run(state, decision, T_max, popSize = 38037, costs=[50,50,50,50],
             inital_inf = 0, outside_inf = 0, transRisk, startSim = '2020-08-24', 
             endSim = '2020-11-20', data=None, pre_data = None,
             heroku=False, max_time=25):
    path = os.getcwd()
    inv_dt = 10                 # insert time steps within each day
    init_num_inf = 4            # initial number of infected people
    # prop = 1                    # proportion of the total population
    travel_num_inf = 0.5        # number of infected among travelers
    #test_sensitivity = 0.9      # testing sensitivity
    unif = 'N'
    decision_making_date = pd.Timestamp(2020, 8,24)      # date of starting decision making
    final_simul_start_date = pd.Timestamp(2020, 11,20)   # date of last simulation date
    sim_week = final_simul_start_date.week - decision_making_date.week + 1
    gv.setup_global_variables(state, inv_dt, init_num_inf, decision_making_date.date(),
                              travel_num_inf, sim_week, #unif,
                              final_simul_start_date.date(), path, heroku=heroku)
    # gv.md_salary = uw
    gv.test_cost = costs
    # distribution the simulation population by age and gender
    gv.total_pop, gv.pop_dist_v = gv.read_pop_dist(state, uw, path = path, heroku = heroku)
    # gv.T_max = T_max
    gv.T_max = abs((decision_making_date.date() - final_simul_start_date.date()).days)
    # ^ set gloable variables
    timer, time_start = 0, time.time() # set timer and current time
    model = cov.CovidModel(data=data, heroku=heroku) # establish model
    i = 0 # set loop counter
    d_m = decision[i] # set current policy at time=now
    # while there time now < time end AND
    # while timer < max_time AND time_step is at the end of a day (aka no partial days)
    while model.t < model.T_total and (timer < max_time or i % model.inv_dt != 0):
        model.t += 1
        if model.t % 25 == 0: print('t', model.t, np.round(timer, 2))
        # ^ print progress
        if i % model.inv_dt == 0 and i//model.inv_dt < len(decision): # if next day, set policy for the new day
            d_m = decision[i//model.inv_dt]

        model.step(action_t = d_m) # run step
        i += 1  # move time
        timer = time.time() - time_start # update timer
    mod = model.t - model.d * model.inv_dt
    mod = 0
    date_range = pd.date_range(start= model.sim_start_day, periods= model.d, freq = 'D')
    # ^^ FROM XINMENG ... idk what this means
    # save results... this is used to restart a simulation on next loop
    dic = {'self.pop_dist_sim': model.pop_dist_sim[model.t- mod].tolist(),
           'self.num_diag': model.num_diag[model.t- mod].tolist(),
           'self.num_hosp': model.num_hosp[model.t- mod].tolist(),
           'self.num_dead': model.num_dead[model.t- mod].tolist(),
           'self.num_base_test': model.num_base_test[model.t- mod].tolist(),
           'self.num_uni_test': model.num_uni_test[model.t- mod].tolist(),
           'self.num_trac_test': model.num_trac_test[model.t-mod].tolist(),
           'self.tot_num_diag': model.tot_num_diag[model.t-mod],
           'self.tot_num_dead': model.tot_num_dead[model.t-mod],
           'self.tot_num_hosp': model.tot_num_hosp[model.t-mod],
           'self.op_ob.cumulative_cost_plot[0]': model.op_ob.cumulative_cost_plot[model.d],
           #'self.rate_unemploy': model.rate_unemploy[model.t-mod],
           'self.next_start_day': date_range[-1].strftime("%m/%d/%Y"),
           'self.t': model.t}
    # get results for graphics on website
    # output = model.op_ob.write_output(pre_results = gv.pre_results_df,
    #                                   date_range = date_range,
    #                                   pre_data = pre_data)
    output = model.op_ob.write_current_results_mod()
    remaining_decision = decision[i//model.inv_dt :]
    # ^ cut the simulation policy for what still needs to be simulated
    is_complete = 'True' if model.T_total - model.t <= 2 else 'False'
    # ^ check if simulation is done
    # results = {'pre_data':dic, 'to_java':output, 'remaining_decision':remaining_decision,
    #            'is_complete':is_complete, 'cost':costs, 'unemp':uw}
    results = {'pre_data':dic, 'to_java':output, 'remaining_decision':remaining_decision,
               'is_complete':is_complete, 'cost':costs, 'prop': uw}
    results = prep_results_for_java(results, pre_data)
    # ^^ prep results for java
    return results


def prep_results_for_java(results, prior_results=None):
    results = copy.deepcopy(results)
    results['is_complete'] = str(results['is_complete'])
    if type(results['to_java']) == type(None):
        results['to_java'] = json.dumps(results['to_java'])
    else:
        temp = results['to_java']
        temp = temp.loc[temp['cumulative diagnosis']!=0]
        if type(prior_results) != type(None):
            temp = prior_results.append(temp, ignore_index=True)
        results['to_java'] = json.dumps(temp.astype(str).to_dict('index'))
    results['remaining_decision'] = json.dumps(results['remaining_decision'].tolist())
    results['cost'] = json.dumps(results['cost'])
    results['prop'] = json.dumps(results['prop'])
    results['pre_data'] = json.dumps(results['pre_data'])
    return results


def prep_input_for_python(results):
    results = copy.deepcopy(results)
    for plan, instructions in results.items():
        if plan in ['A', 'B', 'C']:
            if instructions['to_java'] != 'null':
                instructions['to_java'] = pd.read_json(instructions['to_java']).T
            else:
                instructions['to_java'] = None
            instructions['remaining_decision'] = np.array(json.loads(instructions['remaining_decision']))
            instructions['pre_data'] = json.loads(instructions['pre_data'])
            instructions['cost'] = json.loads(instructions['cost'])
            instructions['prop'] =  json.loads(instructions['prop'])
            results[plan] = instructions
    return results
