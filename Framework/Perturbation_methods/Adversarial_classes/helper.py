import numpy as np
import torch

class Helper:
    @staticmethod
    def check_conversion_control_action(adv_position, X, Y, future_action):
        # Check if the control action is converted correctly
        if future_action:
            equal_tensors = torch.round(adv_position[:, 0, :, :], decimals=2) == torch.round(torch.cat((X[:, 0, :],Y[:, 0, :]),dim=1), decimals=2)
        else:
            equal_tensors = torch.round(adv_position[:, 0, :], decimals=2) == torch.round(X[:, 0, :], decimals=2)

        if not torch.all(equal_tensors):
            raise ValueError("The dynamical transformation is not correct.")
        
    def create_new_input(X,Y, ADE_loss_adv_future_GT,ADE_loss_adv_future_pred, fake_collision_loss_GT, fake_collision_loss_Pred, hide_collision_loss_GT, hide_collision_loss_Pred):
        if ADE_loss_adv_future_GT or ADE_loss_adv_future_pred or fake_collision_loss_GT or fake_collision_loss_Pred or hide_collision_loss_GT  or hide_collision_loss_Pred:
            future_action = True
            new_input = torch.cat((X,Y),dim=2)
        else:
            future_action = False
            new_input = X

        return new_input, future_action
  
    @staticmethod
    def return_to_old_shape(Y_new_pert, Y_shape):
        nan_array = np.full((Y_shape[0], Y_shape[1], Y_shape[2]-Y_new_pert.shape[2], Y_shape[3]), np.nan)
        return np.concatenate((Y_new_pert, nan_array), axis=2)
    
    @staticmethod
    def validate_plot_settings(First, Second):
        if Second and not First:
            raise ValueError("Second element can only be True if First element is also True.")
    
    @staticmethod
    def remove_nan_values(Y):
        # Calculate the maximum length where all values are non-NaN in the path lenght channel across all samples
        max_length = np.min(np.sum(~np.isnan(Y[:, :, :, 0]), axis=2)[:, 0])
        
        # Trim the array to the maximum length without NaN values in the path lenght channel
        Y_trimmed = Y[:, :, :max_length, :]
    
        return Y_trimmed
    
    @staticmethod
    def assert_only_one_true(*args):
        # Check that exactly one argument is True
        assert sum(args) == 1, "Assertion Error: Exactly one loss function must be activated."

    @staticmethod
    def assert_only_one_true_barrier(*args):
        # Check that exactly one argument is True
        assert sum(args) <= 1, "Assertion Error: Only one barrier loss function can be activated or none."

    @staticmethod
    def masked_data(X, Y):
        # Check the edge case scenario where the vehicle is standing still, by checking if the first and last position are within 2 meters
        mask_values_X = np.abs(X[:,1,0,0]-X[:,1,-1,0]) < 0.2
        mask_values_Y = np.abs(Y[:,1,0,1]-Y[:,1,-1,1]) < 0.2

        return mask_values_X, mask_values_Y
    
    @staticmethod
    def flip_dimensions(X, Y, agent, flip_dimensions):
        # Early exit if no dimension flipping is required
        if flip_dimensions == False:
            return X, Y, None
        
        # Determine the indices for the target and ego agents
        i_agent_perturbed = np.where(agent == 'tar')[0][0]
        i_agent_collision = np.where(agent == 'ego')[0][0]

        # Create an array of indices for other agents, excluding the target and ego agents
        other_agents = np.arange(Y.shape[1])
        other_agents = np.delete(other_agents, [i_agent_perturbed, i_agent_collision])

        # Construct a new order for agents: target, ego, followed by the rest
        agent_order = np.array([i_agent_perturbed, i_agent_collision, *other_agents])

        # Rearrange the X and Y arrays according to the new agent order
        X = X[:, agent_order, :, :]
        Y = Y[:, agent_order, :, :]

        return X, Y, agent_order
    
    @staticmethod
    def find_limits_data(X, Y, index):
        min_value_x = np.inf
        max_value_x = -np.inf
        min_value_y = np.inf
        max_value_y = -np.inf

        # Find plot limits
        for j in range(X.shape[1]):
            min_value_x = min(min_value_x, np.min(X[index,j,:,0]))
            min_value_x = min(min_value_x, np.min(Y[index,j,:,0]))

            max_value_x = max(max_value_x, np.max(X[index,j,:,0]))
            max_value_x = max(max_value_x, np.max(Y[index,j,:,0]))

            min_value_y = min(min_value_y, np.min(X[index,j,:,1]))
            min_value_y = min(min_value_y, np.min(Y[index,j,:,1]))

            max_value_y = max(max_value_y, np.max(X[index,j,:,1]))
            max_value_y = max(max_value_y, np.max(Y[index,j,:,1]))

        return min_value_x, max_value_x, min_value_y, max_value_y
    
    @staticmethod
    def convert_to_tensor(X, Y, spline_data):
        # Convert all inputs to tensors
        X = Helper.to_cuda_tensor(X)
        Y = Helper.to_cuda_tensor(Y)

        # Handle the optional spline data
        if spline_data is not None:
            spline_data = Helper.to_cuda_tensor(spline_data)
        else:
            spline_data = None
        
        return X, Y, spline_data
    
    @staticmethod
    def to_cuda_tensor(np_array):
        # Helper function to convert numpy arrays to CUDA tensors
        return torch.from_numpy(np_array).to(dtype=torch.float32, device='cuda')
    
    @staticmethod
    def detach_tensor(X_new_adv, Y_new_adv, Pred_t,Pred_iter_1):
        # Detach tensors, move them to CPU, and convert them to NumPy arrays
        X_new_pert = X_new_adv.detach().cpu().numpy()
        Y_new_pert = Y_new_adv.detach().cpu().numpy()
        Pred_t = Pred_t.detach().cpu().numpy()
        Pred_iter_1 = Pred_iter_1.detach().cpu().numpy()

        # Calculate the mean of the predicted future positions
        Pred_t = np.mean(Pred_t, axis=1)
        Pred_iter_1 = np.mean(Pred_iter_1, axis=1)
        
        return X_new_pert, Y_new_pert, Pred_t, Pred_iter_1
    
    @staticmethod
    def is_monotonic(data):
        # Check for monotonic increasing
        is_increasing = np.all(data[:-1,0] <= data[1:,0])
        # Check for monotonic decreasing
        is_decreasing = np.all(data[:-1,0] >= data[1:,0])

        return is_increasing or is_decreasing
    
    @staticmethod
    def return_data(adv_position, X, Y, future_action):
        if future_action:
            X_new, Y_new = torch.split(adv_position, [X.shape[2], Y.shape[2]], dim=2)
            X_new_adv = X_new
            Y_new_adv = Y_new
        else: 
            X_new = adv_position
            Y_new = Y
            X_new_adv = adv_position
            Y_new_adv = Y

        return X_new, Y_new, X_new_adv, Y_new_adv
    
    @staticmethod
    def retrieve_name_attack(ADE_loss,ADE_loss_adv_future_GT,ADE_loss_adv_future_pred,collision_loss,fake_collision_loss_GT,fake_collision_loss_Pred,hide_collision_loss_GT,hide_collision_loss_Pred):
         # Dictionary to hold the variables and their names
        variables = {
            "ADE_loss": ADE_loss,
            "ADE_loss_adv_future_GT": ADE_loss_adv_future_GT,
            "ADE_loss_adv_future_pred": ADE_loss_adv_future_pred,
            "collision_loss": collision_loss,
            "fake_collision_loss_GT": fake_collision_loss_GT,
            "fake_collision_loss_Pred": fake_collision_loss_Pred,
            "hide_collision_loss_GT": hide_collision_loss_GT,
            "hide_collision_loss_Pred": hide_collision_loss_Pred,
        }

        # Loop through the dictionary and return the name of the variable that is True
        for name, value in variables.items():
            if value:
                return name