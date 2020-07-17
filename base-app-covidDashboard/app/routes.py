from flask import (Blueprint, flash, redirect, render_template, request,
                   url_for, jsonify, Response, send_file, make_response)
from app.COVID19master import backend
import os
import pandas as pd
import xlsxwriter

bp = Blueprint('blueprint', __name__)



@bp.route('/', methods=('GET', 'POST'))
def index():
    #return render_template('index.html')
    return render_template('policy_builder.html')


@bp.route('/download_newfile')
def download_newfile():
    path = 'results.xlsx'
    #return send_file(path, attachment_filename='results.xlsx', as_attachment=True)
    return send_file(path, attachment_filename='simulationResults.xlsx',
                     as_attachment=True, cache_timeout=0)


@bp.route('/prep_sim', methods=('GET','POST'))
def prep_sim():
    get = request.get_json() # retrieve input data from ajax request
    # if this is the first time the simulation is being run through, prep the
    # data from the front end... see line 91 where this is set as False.
    if get['new'] == 'True':
        rl_input = backend.read_ABC(get)
        # transform user inputs... will return
        results = {plan:{'is_complete':'False',
                         'remaining_decision':decision,
                         'to_java':None,
                         'pre_data':None,
                         'cost':[int(cost) for cost in get['cost']],
                         'pop_size':int(get['popSize']),
                         'init_num_inf':int(get['initialInfection']),
                         'travel_num_inf':float(get['outsideInfection']),
                         'startSim': get['start'],
                         'endSim':get['end'],
                         'state':get['state'],
                         'trans_prob':float(get['policy']['TR'][plan]) / 100}
                   for plan, decision in rl_input.items()}
    else:
    # else, if this is NOT the first time the prep_sim function is called,
    # take the partially completed simulation data and prep it.
        results = backend.prep_input_for_python(get)
    heroku = False if len(os.getcwd()) > 25 else True # set the paths
    max_time = 20 # passed to simulation as the max time to run for
    stop=False # condition to make sure simulation loop does not start another plan
    for plan, instructions in results.items(): # for plan in [A,B,C]
        if type(instructions) == dict:
            # if simulation isn't done, and stop condition not set yet
            if instructions['is_complete'] == 'False' and stop == False:
                decision = instructions['remaining_decision']
                output = backend.main_run(state=instructions['state'],
                                          decision = decision,
                                          T_max = decision.shape[0],
                                          pop_size = instructions['pop_size'],
                                          costs = instructions['cost'],
                                          init_num_inf =instructions['init_num_inf'],
                                          travel_num_inf =instructions['travel_num_inf'],
                                          startSim = instructions['startSim'],
                                          endSim = instructions['endSim'],
                                          trans_prob = instructions['trans_prob'],
                                          data=instructions['pre_data'],
                                          pre_data=instructions['to_java'],
                                          heroku=heroku,
                                          max_time=max_time)
                # ^ run simulation... will end after 15 seconds, or if simulation
                # of current plan is completed
                stop = True # set stop conditions
                results[plan] = output # save results
            else:
                # if plan has already been simulated, prep back for java
                results[plan] = backend.prep_results_for_java(results[plan])
    # check to see if all plans are simulated
    if all([v['is_complete'] == 'True' for k,v in results.items() if type(v) == dict]):
        status = 'Finished'
        inputs = backend.prep_input_excel(results)
        print(inputs)
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        if heroku == False:
            writer = pd.ExcelWriter('app\\results.xlsx', engine='xlsxwriter')
        else:
            writer = pd.ExcelWriter('app/results.xlsx', engine='xlsxwriter')
        # Write each dataframe to a different worksheet.
        inputs.to_excel(writer, sheet_name='Simulation Inputs')
        pd.read_json(results['A']['to_java']).T.to_excel(writer, sheet_name='Plan A')
        if 'B' in results.keys():
            pd.read_json(results['B']['to_java']).T.to_excel(writer, sheet_name='Plan B')
        if 'C' in results.keys():
            pd.read_json(results['C']['to_java']).T.to_excel(writer, sheet_name='Plan C')
        writer.save() 
    else:
        status = 'Not Finished'
    results['status'] = status # set status
    results['new'] = 'False' # set status
    return jsonify(status='success', data=results)
    
    