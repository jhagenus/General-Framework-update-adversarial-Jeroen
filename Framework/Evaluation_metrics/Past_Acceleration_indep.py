import numpy as np
from evaluation_template import evaluation_template 

class Past_Acceleration_indep(evaluation_template):
    r'''
    The acceleration assumed based on a dynamic unicycle model.
    '''
     
    def evaluate_prediction_method(self):
        Path_true, Pred_agent = self.get_true_past_paths() # Num_samples x num_agents x n_I x 2 , Num_samples x num_agents

        Path_true = Path_true[Pred_agent] # N x n_I x 2

        # Get displacement
        DP = Path_true[:, 1:] - Path_true[:, :-1] # N x n_I-1 x 2
        dt = self.data_set.dt

        # Get velocity
        V = np.linalg.norm(DP, axis=-1) / dt # N x n_I-1
        # headings = np.arctan2(DP[:, :, 1], DP[:, :, 0]) # N x n_I-1

        # Get second order derivatives
        # heading_diff = headings[:, 1:] - headings[:, :-1] # N x n_I-2
        # heading_diff = np.mod(heading_diff + np.pi, 2 * np.pi) - np.pi # N x n_I-2
        # yaw_rate = heading_diff / dt # N x n_I-2

        velocity_diff = V[:, 1:] - V[:, :-1] # N x n_I-2
        acceleration = velocity_diff / dt # N x n_I-2
        # curvature = yaw_rate / (np.abs(V[:, :-1]) + 1e-4) # N x n_I-2


        # Get mean over timesteps
        abs_acceleration = np.abs(acceleration) # N x n_I-2
        Error = abs_acceleration.mean()

        # Adjust for assumption of there being 1 zero acceleration timestep at teh beginning
        Error *= abs_acceleration.shape[1] / (abs_acceleration.shape[1] + 1)

        return [Error]
    
    
    def get_name(self = None):
        names = {'print': 'Acceleration',
                 'file': 'Acc_indep',
                 'latex': r'\emph{$a$ [m]}'}
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
        return True
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
    
    def metric_boundaries(self = None):
        return [0.0, None]