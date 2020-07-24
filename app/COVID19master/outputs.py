import numpy as np
import pandas as pd


class output_var:

    def __init__(self, sizeofrun, state, start_d, decision_d):

        self.num_inf_plot = np.zeros(sizeofrun)                 # number of diagnosis            
        self.num_hosp_plot = np.zeros(sizeofrun)                # number of hospitalized 
        self.num_dead_plot = np.zeros(sizeofrun)                # number of deaths
        self.VSL_plot = np.zeros(sizeofrun)                     # value of statistical life 
       
        self.cumulative_inf = np.zeros(sizeofrun)               # cumulative diagnosis
        self.cumulative_hosp = np.zeros(sizeofrun)              # cumulative hospitalized 
        self.cumulative_dead = np.zeros(sizeofrun)              # cumulative deaths
        self.cumulative_new_inf_plot = np.zeros(sizeofrun)      # cumulative new infection
        self.cumulative_cost_plot = np.zeros(sizeofrun)         # cumulative cost

        self.univ_test_cost = np.zeros(sizeofrun)               # cost of universal testing
        self.trac_test_cost = np.zeros(sizeofrun)               # cost of contact and tracing
        self.bse_test_cost = np.zeros(sizeofrun)                # cost of symptom-based testing
        self.tot_test_cost_plot = np.zeros(sizeofrun)           # total cost of testing 
        self.quarantine_cost_plot = np.zeros(sizeofrun)         # cost of quarantine

        self.num_uni = np.zeros(sizeofrun)                      # number of universal testing
        self.num_trac = np.zeros(sizeofrun)                     # number of contact and tracing
        self.num_base = np.zeros(sizeofrun)                     # number of symptom-based testing
        self.policy_plot = np.zeros((sizeofrun, 3))             # decision choices 
        self.num_diag_inf = np.zeros(sizeofrun)                 # number of diagnosed infections Q_L + Q_E + Q_I
        self.num_undiag_inf = np.zeros(sizeofrun)               # number of undiagnosed infections L + E + I
        self.num_new_inf_plot = np.zeros(sizeofrun)             # number of new infections 

        self.num_quarantined_plot = np.zeros(sizeofrun)         # number of quarantined 

        self.T_c_plot = np.zeros(sizeofrun)                     # number of contact and tracing needed
        self.T_u_plot = np.zeros(sizeofrun)                     # number of universal testing needed

        self.travel_num_inf_plot = np.zeros(sizeofrun)          # number of travel related infection 
        

        # number of hospitalization and deaths by age group
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
        self.date_range = pd.date_range(start = self.start_d, periods= sizeofrun, freq = 'D')  # date range
       
     
        
    # Function to output scenario analysis needed results 
    def write_scenario_needed_results(self):
        df = pd.DataFrame({'Date': self.date_range[1:],
                           'Value of statistical life-year (VSL) loss': self.VSL_plot[1:],
                           'Cost of mass tests': self.univ_test_cost[1:],
                           'Cost of contact trace and tests':self.trac_test_cost[1:],
                           'Cost of symptom-based tests': self.bse_test_cost[1:],
                           'Cost of quarantine': self.quarantine_cost_plot[1:],
                           'Total cost of tests': self.tot_test_cost_plot[1:],
                           'Number of new diagnosis through contact trace and tests': self.num_trac[1:],
                           'Number of new diagnosis through symptom-based tests': self.num_base[1:],
                           'Number of new diagnosis through mass tests':self.num_uni[1:],
                           'Newly diagnosed': self.num_inf_plot[1:],
                           'Number of hospitalized per day': self.num_hosp_plot[1:],
                           'Number of deaths per day': self.num_dead_plot[1:],
                           'Number of diagnosed infections': self.num_diag_inf[1:],
                           'Number of undiagnosed infections': self.num_undiag_inf[1:],
                           'Number of new infections': self.num_new_inf_plot[1:],
                           'Number of quarantines (only true positives)': self.num_quarantined_plot[1:],
                           'Number of travel-related infection': self.travel_num_inf_plot[1:],
                           'Number of trace and tests': self.T_c_plot[1:],
                           'Number of mass tests': self.T_u_plot[1:],
                           'Contact rate - average contacts per person per day': self.policy_plot[1:, 0],
                        #    'Testing capacity – maximum tests per day through contact trace and tests': self.policy_plot[:, 1],
                        #    'Testing capacity – maximum tests per day through mass tests': self.policy_plot[:, 2],
                           'Percentage through contact trace and tests': self.policy_plot[1:, 1],
                           'Percentage through mass tests': self.policy_plot[1:, 2],
                           'Cumulative diagnosis': self.cumulative_inf[1:],
                           'Cumulative hospitalized': self.cumulative_hosp[1:],
                           'Cumulative deaths': self.cumulative_dead[1:],
                           'Cumulative cases (diagnosed and undiagnosed)': self.cumulative_new_inf_plot[1:],
                           'Cumulative costs': self.cumulative_cost_plot[1:]
                           
                        #    'Number of hospitalized: age < 25': self.tot_hosp_AgeGroup1_plot,
                        #    'Number of hospitalized: age 25-29': self.tot_hosp_AgeGroup2_plot,
                        #    'Number of hospitalized: age 30-39': self.tot_hosp_AgeGroup3_plot,
                        #    'Number of hospitalized: age 40-49': self.tot_hosp_AgeGroup4_plot,
                        #    'Number of hospitalized: age 50-59': self.tot_hosp_AgeGroup5_plot,
                        #    'Number of hospitalized: age 60-69': self.tot_hosp_AgeGroup6_plot,
                        #    'Number of hospitalized: age 70-79': self.tot_hosp_AgeGroup7_plot,
                        #    'Number of hospitalized: age 80+': self.tot_hosp_AgeGroup8_plot,
                        #    'Number of deaths: age < 25': self.tot_dead_AgeGroup1_plot,
                        #    'Number of deaths: age 25-29': self.tot_dead_AgeGroup2_plot,
                        #    'Number of deaths: age 30-39': self.tot_dead_AgeGroup3_plot,
                        #    'Number of deaths: age 40-49': self.tot_dead_AgeGroup4_plot,
                        #    'Number of deaths: age 50-59': self.tot_dead_AgeGroup5_plot,
                        #    'Number of deaths: age 60-69': self.tot_dead_AgeGroup6_plot,
                        #    'Number of deaths: age 70-79': self.tot_dead_AgeGroup7_plot,
                        #    'Number of deaths: age 80+': self.tot_dead_AgeGroup8_plot,
                           
                           })
        return df

    def write_summary_results(self, df, p, c, a_c,  SAR, R0, a_u, unit_cost, num_init_trace, table_num, pop_size):   
        # summary file
        df2 = pd.DataFrame({'Transmission risk': p,
                            'Contact rate - average contacts per person per day': c,
                            'Percentage through contact trace and tests': a_c,
                            'Percentage through mass tests': a_u,
                            'Unit cost of symptom-based tests': unit_cost[0],
                            'Unit cost of contact trace and tests': unit_cost[1],
                            'Unit cost of mass tests': unit_cost[2],
                            'Unit cost of quarantine': unit_cost[3],
                            'Number diagnosed for tracing initiation':num_init_trace,
                            'Population size': pop_size,
                            'R0': R0,
                            'Second attack rate': SAR,
                            'Cumulative value of statistical life-year (VSL) loss': df['Value of statistical life-year (VSL) loss'].sum(),
                            'Cumulative number of contact trace and tests needed': df['Number of trace and tests'].sum(),
                            'Cumulative number of mass tests needed': df['Number of mass tests'].sum(),
                            'Cumulative diagnosis through contact trace and tests': df['Number of new diagnosis through contact trace and tests'].sum(),
                            'Cumulative diagnosis through symptom-based tests': df['Number of new diagnosis through symptom-based tests'].sum(),
                            'Cumulative diagnosis through mass tests': df['Number of new diagnosis through mass tests'].sum(),
                            'Cumulative diagnosis': df['Cumulative diagnosis'][-1:].to_numpy()[0],
                            'Cumulative hospitalized': df['Cumulative hospitalized'][-1:].to_numpy()[0],
                            'Cumulative deaths': df['Cumulative deaths'][-1:].to_numpy()[0], 
                            'Cumulative cases (diagnosed and undiagnosed)': df['Cumulative cases (diagnosed and undiagnosed)'][-1:].to_numpy()[0],
                            'Peak contact trace and tests': df['Number of trace and tests'].max(),
                            'Peak quarantined': df['Number of quarantined (only true positives)'].max(),
                            'Cumulative quarantined': df['Number of quarantined (only true positives)'].sum(),
                            'Cumulative costs': df['Cumulative costs'][-1:].to_numpy()[0],
                            'Cumulative costs of testing': df['Total cost of tests'].sum(),
                            'Table': table_num},
                            index=[0])     
             
        return df2
    

 