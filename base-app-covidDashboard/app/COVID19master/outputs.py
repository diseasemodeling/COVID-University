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
        self.cumulative_cost_plot = np.zeros(sizeofrun)         # cumulative cost

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
        df = pd.DataFrame({'Date': self.date_range[1:],
                           'Value of statistical life-year (VSL) loss': self.VSL_plot[1:],
                           'Cost of mass tests': self.univ_test_cost[1:],
                           'Cost of trace and tests':self.trac_test_cost[1:],
                           'Cost of symptom-based tests': self.bse_test_cost[1:],
                           'total cost of testing': self.tot_test_cost_plot[1:],
                           'Number of new diagnosis through contact tracing': self.num_trac[1:],
                           'Number of new diagnosis through symptom-based testing': self.num_base[1:],
                           'Number of new diagnosis through universal testing':self.num_uni[1:],
                           'Number of diagnosis per day': self.num_inf_plot[1:],
                           'Number of hospitalized per day': self.num_hosp_plot[1:],
                           'Number of deaths per day': self.num_dead_plot[1:],
                           'Number of infected, diagnosed': self.num_diag_inf[1:],
                           'Number of infected, undiagnosed': self.num_undiag_inf[1:],
                           'Number of new infection': self.num_new_inf_plot[1:],
                           'Number of quarantined (per day)(with 0'+ '%' +' false positives)': self.num_quarantined_plot[1:],
                           'Number of infected among traveling': self.travel_num_inf_plot[1:],
                           'Number of trace and tests (per day)': self.T_c_plot[1:],
                           'Number of mass tests (per day)': self.T_u_plot[1:],
                           'Contacts per day': self.policy_plot[1:, 0],
                           'Testing capacity – maximum tests per day through contact tracing': self.policy_plot[1:, 1],
                           'Testing capacity – maximum tests per day through universal testing': self.policy_plot[1:, 2],
                           'Cumulative diagnosis': self.cumulative_inf[1:],
                           'Cumulative hospitalized': self.cumulative_hosp[1:],
                           'Cumulative deaths': self.cumulative_dead[1:],
                           'Cumulative cost': self.tot_test_cost_plot.cumsum()[1:],
                           'Cumulative cases (diagnosed and undiagnosed)': self.cumulative_new_inf_plot[1:]})
        return df
