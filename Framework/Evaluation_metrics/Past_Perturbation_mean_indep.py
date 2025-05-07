import numpy as np
from evaluation_template import evaluation_template 

class Past_Perturbation_mean_indep(evaluation_template):
    r'''
    The acceleration assumed based on a dynamic unicycle model.
    '''
     
    def evaluate_prediction_method(self):
        Path_true, Path_unperturbed, Pred_agent = self.get_true_past_paths(return_unperturbed = True) # Num_samples x num_agents x n_I x 2 , Num_samples x num_agents

        Perturbation = Path_true - Path_unperturbed # N x n_A x n_I x 2
        Perturbation = Perturbation[Pred_agent] # N x n_I x 2

        Perturbation = np.linalg.norm(Perturbation, axis = -1) # N x n_I
        Perturbation = np.mean(Perturbation, axis = -1) # N x n_I
        Error = np.mean(Perturbation, axis = 0) # n_I
        return [Error]
    
    
    def get_name(self = None):
        names = {'print': 'Mean perturbation',
                 'file': 'Pert_mean_indep',
                 'latex': r'\emph{$D_{\text{mean}}$ [m]}'}
        return names

    def setup_method(self):
        pass
        
    def get_output_type(self = None):
        return 'path_all_wo_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def check_applicability(self):
        return None
    
    def partial_calculation(self = None):
        return 'Pred_agents' 
    
    def is_log_scale(self = None):
        return False
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
    
    def metric_boundaries(self = None):
        return [0.0, None]