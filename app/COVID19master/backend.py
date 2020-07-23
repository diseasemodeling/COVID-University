import numpy as np
import pandas as pd
import time, os, json, copy
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from app.COVID19master import global_var as gv
from app.COVID19master import COVID_model_colab as cov

def read_ABC(from_java):
    data = {'A':{}, 'B':{}, 'C':{}}
    policies = from_java['policy']
    num_days = int(from_java['lenSim'])
    for name, policy in policies.items():
        for plan, array in policy.items():
            data[plan][name] = array
    plans = ['A', 'B', 'C']
    policies = ['CR', 'TT', 'MT']
    index = [i for i in range(num_days+1)]
    rl_input = {}
    for plan in plans:
        rl_input[plan] = pd.DataFrame(index=index, columns=policies)
        for policy, value in data[plan].items():
            if policy in policies:
                if policy in ['MT', 'TT']:
                    value = float(value) / 100
                rl_input[plan][policy] = [float(value) for i in index]
        rl_input[plan] = rl_input[plan].values
    return rl_input

def main_run(decision, T_max, data=None, state='UMASS', pop_size = 38037,
             costs=[50,50,50,50], init_num_inf = 0, travel_num_inf = 0.5,
             startSim = '2020-08-24', endSim = '2020-11-20', trans_prob=0.249,
             num_to_init_trace = 20, filename = 'model.pkl', heroku=False,
             max_time=25): # e.g. filename = 'model.pkl'
    path = os.getcwd()         
    inv_dt = 10                 # insert time steps within each day

    # ^ set gloable variables
    timer, time_start = 0, time.time() # set timer and current time
    if data['load_pickle'] == 'True':
        with open(filename , 'rb') as input:
            model = pickle.load(input) # load model
            print('loading')
    else:
        if data != None:
            pop_size = data['pop_size']
            costs = data['costs']
            startSim = data['startSim']
            endSim = data['endSim']
            init_num_inf = data['init_num_inf']
            travel_num_inf = data['travel_num_inf']
            trans_prob = data['trans_prob']
            num_to_init_trace = data['num_to_init_trace']
            state = data['state']
        decision_making_date = pd.Timestamp(startSim)      # date of starting decision making
        final_simul_end_date = pd.Timestamp(endSim)   # date of last simulation date
        sim_week = final_simul_end_date.week - decision_making_date.week + 1
        gv.setup_global_variables(state, inv_dt, init_num_inf, decision_making_date.date(),
                                  travel_num_inf,sim_week, pop_size, trans_prob, num_to_init_trace,
                                  path, heroku = heroku)
        gv.test_cost = costs
        # distribute the simulation population by age and gender
        gv.pop_dist_v = gv.read_pop_dist(state, pop_size, path = path, heroku = heroku)
        gv.T_max = abs((decision_making_date.date() - final_simul_end_date.date()).days) + 1
        model = cov.CovidModel(heroku=heroku) # establish model
        print('initializing')
    i = 0 # set loop counter
    d_m = decision[i] # set current policy at time=now
    while model.t < model.T_total and (timer < max_time or i % model.inv_dt != 0):
        # while there time now < time end AND
        # while timer < max_time AND
        # while if time_step is at the end of a day (aka no partial days)
        model.t += 1
        if model.t % 25 == 0: print('t', model.t, np.round(timer, 2)) # print progress
        if i % model.inv_dt == 0 and i//model.inv_dt < len(decision): # if next day, set policy for the new day
            d_m = decision[i//model.inv_dt]
        model.step(action_t = d_m) # run step
        i += 1  # move time
        timer = time.time() - time_start # update timer
    output = model.op_ob.write_scenario_needed_results()
    # get results for graphics on website
    remaining_decision = decision[i//model.inv_dt :] # cut the simulation policy for what still needs to be simulated
    is_complete = 'True' if len(remaining_decision) == 0 else 'False'  # ^ check if simulation is done
    is_complete = 'True' if model.T_total - model.t <= 2 else 'False'

    data['load_pickle'] = 'True'
    data['is_complete'] = is_complete
    data['to_java'] = output
    data = prep_results_for_java(data)
    
    while timer < 3:
        timer = time.time() - time_start
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    
    return data    
    



def prep_results_for_java(results, prior_results=None):
    print(results.keys())
    results = copy.deepcopy(results)
    results['is_complete'] = str(results['is_complete'])
    results['load_pickle'] = str(results['load_pickle'])
    if type(results['to_java']) == type(None):
        results['to_java'] = json.dumps(results['to_java'])
    else:
        temp = results['to_java']
        temp = temp.loc[temp['Cumulative diagnosis']!=0]
        #if type(prior_results) != type(None):
        #    temp = prior_results.append(temp, ignore_index=True)
        results['to_java'] = json.dumps(temp.astype(str).to_dict('index'))
    results['remaining_decision'] = json.dumps(results['remaining_decision'].tolist())
    results['costs'] = json.dumps(results['costs'])
    results['pop_size'] = json.dumps(results['pop_size'])
    results['trans_prob'] = json.dumps(results['trans_prob'])
    results['init_num_inf'] = json.dumps(results['init_num_inf'])
    results['travel_num_inf'] = json.dumps(results['travel_num_inf'])
    results['state'] = json.dumps(results['state'])
    results['startSim'] = json.dumps(results['startSim'])
    results['endSim'] = json.dumps(results['endSim'])
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
            instructions['costs'] = json.loads(instructions['costs'])
            instructions['pop_size'] = json.loads(instructions['pop_size'])
            instructions['trans_prob'] = json.loads(instructions['trans_prob'])
            instructions['init_num_inf'] = json.loads(instructions['init_num_inf'])
            instructions['travel_num_inf'] = json.loads(instructions['travel_num_inf'])
            instructions['state'] = json.loads(instructions['state'])
            instructions['endSim'] = json.loads(instructions['endSim'])
            instructions['startSim'] = json.loads(instructions['startSim'])
            #instructions[plan] = instructions
    return results

def prep_input_excel(results):
    dont_include = ['to_java', 'remaining_decision', 'is_complete', 'pre_data']
    cost_name = {0: 'Cost of Sympton-Based Test (Per Person)', 1:'Cost of Trace and Test (Per Person)',
                 2: 'Cost of Mass Test (Per Person)', 3:'Cost of Quarentine (Per Day)',
                 4:'Cost of Quarentine (Per Day)'}
    to_excel = {}
    for key, plan in results.items():
        if key in ['A', 'B', 'C']:
            to_excel[key] = {}
            for point, value in plan.items():
                if point not in dont_include:
                    if point == 'costs':
                        value = json.loads(value)
                        for i, cost in enumerate(value):
                            print(i, cost)
                            to_excel[key][cost_name[i]] = cost
                    else:
                        to_excel[key][point] = value
    to_excel = pd.DataFrame.from_dict(to_excel)
    return to_excel


# def main_run(state, decision, T_max, pop_size = 38037, costs=[50,50,50,50],
#              init_num_inf = 0, travel_num_inf = 0.5, startSim = '2020-08-24', 
#              endSim = '2020-11-20', trans_prob=0.249,  num_to_init_trace = 20, 
#              filename = None, heroku=False, max_time=25): # e.g. filename = 'model.pkl'
#     path = os.getcwd()         
#     inv_dt = 10                 # insert time steps within each day

#     decision_making_date = pd.Timestamp(startSim)      # date of starting decision making
#     final_simul_end_date = pd.Timestamp(endSim)   # date of last simulation date
#     sim_week = final_simul_end_date.week - decision_making_date.week + 1
#     gv.setup_global_variables(state, inv_dt, init_num_inf, decision_making_date.date(),
#                               travel_num_inf,sim_week, pop_size, trans_prob, num_to_init_trace,
#                               path, heroku = heroku)
#     gv.test_cost = costs
#     # distribute the simulation population by age and gender
#     gv.pop_dist_v = gv.read_pop_dist(state, pop_size, path = path, heroku = heroku)
#     gv.T_max = abs((decision_making_date.date() - final_simul_end_date.date()).days) + 1
#     # ^ set gloable variables
#     timer, time_start = 0, time.time() # set timer and current time
#     try :
#         with open(filename , 'rb') as input:
#             model = pickle.load(input) # load model
#             print('loading')
#     except:
#         model = cov.CovidModel(heroku=heroku) # establish model
#         print('initializing')
#     i = 0 # set loop counter
#     d_m = decision[i] # set current policy at time=now
#     while model.t < model.T_total and (timer < max_time or i % model.inv_dt != 0):
#         # while there time now < time end AND
#         # while timer < max_time AND
#         # while if time_step is at the end of a day (aka no partial days)
#         model.t += 1
#         if model.t % 25 == 0: print('t', model.t, np.round(timer, 2)) # print progress
#         if i % model.inv_dt == 0 and i//model.inv_dt < len(decision): # if next day, set policy for the new day
#             d_m = decision[i//model.inv_dt]
#         model.step(action_t = d_m) # run step
#         i += 1  # move time
#         timer = time.time() - time_start # update timer
    
#     with open(filename, 'wb') as output:  # Overwrites any existing file.
#         pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    
#     if model.t == model.T_total:
#         output = model.op_ob.write_scenario_needed_results()
#         return output

"""# Funtion for one scenario analysis
def main_run(state, decision, T_max, pop_size = 38037, costs=[50,50,50,50],
             init_num_inf = 0, travel_num_inf = 0.5, startSim = '2020-08-24', 
             endSim = '2020-11-20', trans_prob=0.249, data=None, pre_data = None,
             heroku=False, max_time=25):
    path = os.getcwd()
    inv_dt = 10                 # insert time steps within each day

    decision_making_date = pd.Timestamp(startSim)      # date of starting decision making
    final_simul_end_date = pd.Timestamp(endSim)   # date of last simulation date
    sim_week = final_simul_end_date.week - decision_making_date.week + 1
    gv.setup_global_variables(state, inv_dt, init_num_inf, decision_making_date.date(),
                              travel_num_inf,sim_week, final_simul_end_date.date(),
                              pop_size, trans_prob, path, heroku=heroku)
    gv.test_cost = costs
    # distribution the simulation population by age and gender
    gv.pop_dist_v = gv.read_pop_dist(state, pop_size, path = path, heroku = heroku)
    gv.T_max = abs((decision_making_date.date() - final_simul_end_date.date()).days) + 1
    # ^ set gloable variables
    timer, time_start = 0, time.time() # set timer and current time
    model = cov.CovidModel(data=data, heroku=heroku) # establish model
    i = 0 # set loop counter
    d_m = decision[i] # set current policy at time=now
    while model.t < model.T_total and (timer < max_time or i % model.inv_dt != 0):
        # while there time now < time end AND
        # while timer < max_time AND
        # while if time_step is at the end of a day (aka no partial days)
        model.t += 1
        if model.t % 25 == 0: print('t', model.t, np.round(timer, 2)) # print progress
        if i % model.inv_dt == 0 and i//model.inv_dt < len(decision): # if next day, set policy for the new day
            d_m = decision[i//model.inv_dt]
        model.step(action_t = d_m) # run step
        i += 1  # move time
        timer = time.time() - time_start # update timer
    mod = model.t - model.d * model.inv_dt
    print('mod', mod)
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
           'self.tot_num_new_inf': model.tot_num_new_inf[model.t-mod],
           'self.op_ob.cumulative_cost_plot': model.op_ob.cumulative_cost_plot[model.d-1],
           'self.next_start_day': date_range[-1].strftime("%m/%d/%Y"),
           'self.num_diag_hist': model.num_diag[(model.d-13): (model.d+1)].tolist(),
           'self.t': model.t}

    # get results for graphics on website
    output = model.op_ob.write_current_results_mod()
    remaining_decision = decision[i//model.inv_dt :]
    # ^ cut the simulation policy for what still needs to be simulated
    is_complete = 'True' if model.T_total - model.t <= 2 else 'False'
    is_complete = 'True' if len(remaining_decision) == 0 else 'False'
    # ^ check if simulation is done
    results = {'pre_data':dic, 'to_java':output, 'remaining_decision':remaining_decision,
               'is_complete':is_complete, 'cost':costs, 'pop_size': pop_size,
               'trans_prob':trans_prob, 'travel_num_inf': travel_num_inf,
               'init_num_inf':init_num_inf, 'state':state, 'startSim':startSim,
               'endSim':endSim}
    results = prep_results_for_java(results, pre_data)
    # ^^ prep results for java
    return results"""