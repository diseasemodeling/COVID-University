# Good source for annotation: http://members.cbio.mines-paristech.fr/~nvaroquaux/tmp/matplotlib/examples/pylab_examples/annotation_demo2.html
import numpy as np
import pandas as pd

class output_var:

    def __init__(self, sizeofrun, state, start_d, decision_d):

        self.time_step = np.zeros(sizeofrun)     
        self.num_inf_plot = np.zeros(sizeofrun)                 # number of diagnosis            
        self.num_hosp_plot = np.zeros(sizeofrun)                # number of hospitalized 
        self.num_dead_plot = np.zeros(sizeofrun)                # number of deaths
        self.VSL_plot = np.zeros(sizeofrun)                     # VSL loss
        # self.SAL_plot = np.zeros(sizeofrun)                     # wage loss for unemployed
        self.cumulative_inf = np.zeros(sizeofrun)               # cumulative diagnosis
        self.cumulative_hosp = np.zeros(sizeofrun)              # cumulative hospitalized 
        self.cumulative_dead = np.zeros(sizeofrun)              # cumulative deaths
        self.cumulative_new_inf_plot = np.zeros(sizeofrun)      # cumulative new infection
        # self.unemployment = np.zeros(sizeofrun)                 # unemployment rate
        self.univ_test_cost = np.zeros(sizeofrun)               # cost of universal testing
        self.trac_test_cost = np.zeros(sizeofrun)               # cost of contact and tracing
        self.bse_test_cost = np.zeros(sizeofrun)                # cost of symptom-based testing
        self.tot_test_cost_plot = np.zeros(sizeofrun)           # total cost of testing 

        self.num_uni = np.zeros(sizeofrun)                      # number of universal testing
        self.num_trac = np.zeros(sizeofrun)                     # number of contact and tracing
        self.num_base = np.zeros(sizeofrun)                     # number of symptom-based testing
        self.policy_plot = np.zeros((sizeofrun, 3))             # decision choices 
        self.num_diag_inf = np.zeros(sizeofrun)                 # number of infection, diagnosed
        self.num_undiag_inf = np.zeros(sizeofrun)               # number of infection, undiagnosed Q_L + Q_E + Q_I
        self.num_new_inf_plot = np.zeros(sizeofrun)             # number of new infection L + E + I
        self.num_quarantined_plot = np.zeros(sizeofrun)         # number of quarantined each day
        self.T_c_plot = np.zeros(sizeofrun)                     # number of contact and tracing needed
        self.T_u_plot = np.zeros(sizeofrun)                     # number of universal testing needed
        self.travel_num_inf_plot = np.zeros(sizeofrun)          # number of travel related infection 

        
        # number of hospitalization and deaths among each age group
        self.tot_hosp_AgeGroup1_plot = np.zeros(sizeofrun)
        self.tot_hosp_AgeGroup2_plot = np.zeros(sizeofrun)
        self.tot_hosp_AgeGroup3_plot = np.zeros(sizeofrun)
        self.tot_hosp_AgeGroup4_plot = np.zeros(sizeofrun)
        self.tot_hosp_AgeGroup5_plot = np.zeros(sizeofrun)
        self.tot_hosp_AgeGroup6_plot = np.zeros(sizeofrun)
        self.tot_hosp_AgeGroup7_plot = np.zeros(sizeofrun)
        self.tot_hosp_AgeGroup8_plot = np.zeros(sizeofrun)
        self.tot_dead_AgeGroup1_plot = np.zeros(sizeofrun)
        self.tot_dead_AgeGroup2_plot = np.zeros(sizeofrun)
        self.tot_dead_AgeGroup3_plot = np.zeros(sizeofrun)
        self.tot_dead_AgeGroup4_plot = np.zeros(sizeofrun)
        self.tot_dead_AgeGroup5_plot = np.zeros(sizeofrun)
        self.tot_dead_AgeGroup6_plot = np.zeros(sizeofrun)
        self.tot_dead_AgeGroup7_plot = np.zeros(sizeofrun)
        self.tot_dead_AgeGroup8_plot = np.zeros(sizeofrun)


        # define some parameters for plotting
        self.State = state
        self.start_d = start_d         # date of starting simulation 
        self.decision_d = decision_d   # date of starting decsion making 
        self.date_range = pd.date_range(start = self.start_d, periods= sizeofrun, freq = 'D')          
        self.dpi = 300         # figure dpi

        
    # write results from current simulation to a new DataFrame
    def write_current_results_mod(self):
        df = pd.DataFrame({'Date': self.date_range,
                           'Value of statistical life-year (VSL) loss': self.VSL_plot,
                           'cost of universal testing': self.univ_test_cost,
                           'cost of contact tracing':self.trac_test_cost,
                           'cost of symptom-based testing': self.bse_test_cost,
                           'total cost of testing': self.tot_test_cost_plot,
                           'number of new diagnosis through contact tracing': self.num_trac,
                           'number of new diagnosis through symptom-based testing': self.num_base,
                           'number of new diagnosis through universal testing':self.num_uni,
                           'Contacts per day': self.policy_plot[:, 0],
                           'Testing capacity – maximum tests per day through contact tracing': self.policy_plot[:, 1],
                           'Testing capacity – maximum tests per day through universal testing': self.policy_plot[:, 2],
                           'cumulative diagnosis': self.cumulative_inf,
                           'cumulative hospitalized': self.cumulative_hosp,
                           'cumulative deaths': self.cumulative_dead,
                           'daily diagnosis': self.num_inf_plot,
                           'daily hospitalized': self.num_hosp_plot,
                           'daily deaths': self.num_dead_plot,
                           'number of infected, diagnosed': self.num_diag_inf,
                           'number of infected, undiagnosed': self.num_undiag_inf,
                           'number of new infection': self.num_new_inf_plot,
                           'cumulative new infection': self.cumulative_new_inf_plot,
                           'number of quarantined': self.num_quarantined_plot,
                           'number of infected among traveling': self.travel_num_inf_plot})
        return df

