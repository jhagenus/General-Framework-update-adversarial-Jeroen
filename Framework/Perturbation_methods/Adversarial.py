from perturbation_template import perturbation_template
import pandas as pd
import os
import numpy as np
import importlib
from Data_sets.data_interface import data_interface
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline, CubicHermiteSpline, UnivariateSpline, interp1d
from PIL import Image
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.lines as mlines

class Adversarial(perturbation_template):
    def check_and_extract_kwargs(self, kwargs):
        '''
        This function checks if the input dictionary is complete and extracts the required values.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the required keys and values.

        Returns
        -------
        None.

        '''
        assert 'data_set_dict' in kwargs.keys(), "Adverserial model dataset is missing (required key: 'data_set_dict')."
        assert 'data_param' in kwargs.keys(), "Adverserial model data param is missing (required key: 'data_param')."
        assert 'splitter_dict' in kwargs.keys(), "Adverserial model splitter is missing (required key: 'splitter_dict')."
        assert 'model_dict' in kwargs.keys(), "Adverserial model is missing (required key: 'model_dict')."
        assert 'exp_parameters' in kwargs.keys(), "Adverserial model experiment parameters are missing (required key: 'exp_parameters')."

        assert kwargs['exp_parameters'][6] == 'predefined', "Perturbed datasets can only be used if the agents' roles are predefined."

        self.kwargs = kwargs

        # Check that in splitter dict the length of repetition is only 1 (i.e., only one splitting method)
        if isinstance(kwargs['splitter_dict']['repetition'], list):

            if len(kwargs['splitter_dict']['repetition']) > 1:
                raise ValueError("The splitting dictionary neccessary to define the trained model used " + 
                                "for the adversarial attack can only contain one singel repetition " + 
                                "(i.e, the value assigned to the key 'repetition' CANNOT be a list with a lenght larger than one).")
            
            kwargs['splitter_dict']['repetition'] = kwargs['splitter_dict']['repetition'][0]
        
        # Load the perturbation model
        pert_data_set = data_interface(kwargs['data_set_dict'], kwargs['exp_parameters'])
        pert_data_set.reset()

        # Select or load repective datasets
        pert_data_set.get_data(**kwargs['data_param'])

        # Exctract splitting method parameters
        pert_splitter_name = kwargs['splitter_dict']['Type']
        pert_splitter_rep = [kwargs['splitter_dict']['repetition']]
        pert_splitter_tp = kwargs['splitter_dict']['test_part']
            
        pert_splitter_module = importlib.import_module(pert_splitter_name)
        pert_splitter_class = getattr(pert_splitter_module, pert_splitter_name)
        
        # Initialize and apply Splitting method
        pert_splitter = pert_splitter_class(pert_data_set, pert_splitter_tp, pert_splitter_rep)
        pert_splitter.split_data()
        
        # Extract per model dict
        if isinstance(kwargs['model_dict'], str):
            pert_model_name   = kwargs['model_dict']
            pert_model_kwargs = {}
        elif isinstance(kwargs['model_dict'], dict):
            assert 'model' in kwargs['model_dict'].keys(), "No model name is provided."
            assert isinstance(kwargs['model_dict']['model'], str), "A model is set as a string."
            pert_model_name = kwargs['model_dict']['model']
            if not 'kwargs' in kwargs['model_dict'].keys():
                pert_model_kwargs = {}
            else:
                assert isinstance(kwargs['model_dict']['kwargs'], dict), "The kwargs value must be a dictionary."
                pert_model_kwargs = kwargs['model_dict']['kwargs']
        else:
            raise TypeError("The provided model must be string or dictionary")
        
        # Get model class
        pert_model_module = importlib.import_module(pert_model_name)
        pert_model_class = getattr(pert_model_module, pert_model_name)
        
        # Initialize the model
        self.pert_model = pert_model_class(pert_model_kwargs, pert_data_set, pert_splitter, True)
        
        # TODO: Check if self.pert_model can call the function that is needed later in perturb_batch (i.e., self.pert_model.adv_generation())

        # Train the model on the given training set
        self.pert_model.train()

        # Define the name of the perturbation method
        self.name = self.pert_model.model_file.split(os.sep)[-1][:-4]

    def perturb_batch(self, X, Y, T, agent, Domain):
        '''
        This function takes a batch of data and generates perturbations.


        Parameters
        ----------
        X : np.ndarray
            This is the past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or at some timesteps partially not observed, then this can include np.nan values.
        Y : np.ndarray, optional
            This is the future observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or at some timesteps partially not observed, then this can include np.nan values. 
            This value is not returned for **mode** = *'pred'*.
        T : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
            the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
            If an agent is not observed at all, the value will instead be '0'.
        Agent_names : np.ndarray
            This is a :math:`N_{agents}` long numpy array. It includes strings with the names of the agents.

        Returns
        -------
        X_pert : np.ndarray
            This is the past perturbed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or at some timesteps partially not observed, then this can include np.nan values.
        Y_pert : np.ndarray, optional
            This is the future perturbed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or at some timesteps partially not observed, then this can include np.nan values. 


        '''

        # Debug mode
        torch.autograd.set_detect_anomaly(True)

        # settings
        # Plot settings input data and spline 
        plot_input = False
        plot_spline = False

        # Plot the loss over the iterations
        plot_loss = True
        loss_store = []

        # Plot the adversarial scene
        static_adv_scene = True
        animated_adv_scene = True

        # validation settings
        self.validate_plot_settings(plot_input, plot_spline)

        # Change ego and tar vehicle
        flip_dimensions = True

        # Spline settings
        spline = True
        spline_interval = 100

        # Initialize parameters
        iter_num = 1
        epsilon_acc = 7
        epsilon_curv = 0.2

        # Learning rate
        learning_rate_decay = True
        gamma = 1

        # Learning rate
        alpha_acc = 3
        alpha_curv = 0.05

        # Make copy of the original data and remove nan from input
        Y_shape = Y.shape
        X_copy = X.copy()
        Y = self.remove_nan_values(Y)
        Y_copy = Y.copy()

        # Select loss function ()
        #ADE loss
        ADE_loss = False
        ADE_loss_barrier = False
        ADE_loss_adv_future = False
        ADE_loss_adv_future_barrier = False

        # Collision loss
        collision_loss = False
        collision_loss_barrier = True
        fake_collision_loss = False
        hide_collision_loss = False

        # check if only one loss function is activated
        self.assert_only_one_true(
                ADE_loss,
                ADE_loss_barrier,
                ADE_loss_adv_future,
                ADE_loss_adv_future_barrier,
                collision_loss,
                collision_loss_barrier,
                fake_collision_loss,
                hide_collision_loss
            )

        # Barrier function
        log_barrier = True
        ADVDO_barrier = False
        spline_barrier = False

        # check if only one barrier function is activated
        self.assert_only_one_true(
                log_barrier,
                ADVDO_barrier,
                spline_barrier
            )

        # Barrier function parameters
        distance_threshold = 1
        log_value = 1.2

        # Check edge case scenarios where agent is standing still
        mask_values_X, mask_values_Y = self.masked_data(X, Y)
        
        # Create data and plot data if required
        X, Y, agent_order, spline_data = self.create_data_plot(X, Y, agent, mask_values_X, mask_values_Y, flip_dimensions, spline_interval, spline, plot_input,plot_spline)
        
        # Convert to tensor
        X, Y, spline_data = self.convert_to_tensor(X, Y, spline_data)

        # Check if future action is required
        if ADE_loss_adv_future or ADE_loss_adv_future_barrier or fake_collision_loss or hide_collision_loss:
            future_action = True
            new_input = torch.cat((X,Y),dim=2)
        else:
            future_action = False
            new_input = X

        # Calculate initial control actions
        control_action, heading_init, velocity_init = self.get_control_actions(X, mask_values_X, flip_dimensions, new_input)

        # Storage for adversarial position
        X_new_adv = torch.zeros_like(X)
        Y_new_adv = torch.zeros_like(Y)

        # Start the optimization of the adversarial attack
        for i in range(iter_num):
            # Reset gradients
            control_action.grad = None

            # Calculate updated adversarial position
            adv_position = self.dynamical_model(new_input, control_action, velocity_init, heading_init)

            # Split the adversarial position back to X and Y
            if future_action:
                X_new, Y_new = torch.split(adv_position, [X.shape[2], Y.shape[2]], dim=2)
                X_new_adv = X_new
                Y_new_adv = Y_new
            else: 
                X_new = adv_position
                Y_new = Y
                X_new_adv = adv_position
                Y_new_adv = Y

            # Output forward pass
            num_steps = Y.shape[2]

            # check if cotrol action are converted correctly
            if i == 0:
                equal_tensors = torch.round(adv_position[:, 0, :], decimals=5) == torch.round(X[:, 0, :], decimals=5)
                if not torch.all(equal_tensors):
                    raise ValueError("The dynamical transformation is not correct.")
                
            # Forward pass through the model
            Pred_t = self.pert_model.predict_batch_tensor(X_new,T,Domain, num_steps)

            # Calculate the loss
            losses = self.calculate_loss(
                        X,
                        X_new,
                        Y,
                        Y_new,
                        Pred_t,
                        ADE_loss,
                        ADE_loss_barrier,
                        ADE_loss_adv_future,
                        ADE_loss_adv_future_barrier,
                        collision_loss,
                        collision_loss_barrier,
                        fake_collision_loss,
                        hide_collision_loss,
                        log_barrier,
                        ADVDO_barrier,
                        spline_barrier,
                        distance_threshold,
                        log_value,
                        spline_data
                        )

            # Store the loss for plot
            loss_store.append(losses.detach().cpu().numpy())
            print(losses)

            # Calulate gradients
            losses.sum().backward()
            grad = control_action.grad

            # Include learning rate decay
            if learning_rate_decay:
                alpha_acc = alpha_acc * (gamma**i)
                alpha_curv = alpha_curv * (gamma**i)

            # Update Control inputs
            with torch.no_grad():
                control_action[:,0,:,0].add_(grad[:,0,:,0], alpha=alpha_acc)
                control_action[:,0,:,1].add_(grad[:,0,:,1], alpha=alpha_curv)
                control_action[:,0,:,0].clamp_(-epsilon_acc, epsilon_acc)
                control_action[:,0,:,1].clamp_(-epsilon_curv, epsilon_curv)
                control_action[:,1:] = 0.0

        # Detach the tensor and convert to numpy
        X_new_pert, Y_new_pert, Pred_t = self.detach_tensor(X_new_adv, Y_new_adv, Pred_t)

        # Plot the results
        self.plot_results(
                X_copy, 
                X_new_pert, 
                Y_copy, 
                Y_new_pert, 
                Pred_t,
                plot_loss, 
                loss_store, 
                future_action,
                static_adv_scene,
                animated_adv_scene
                )

        # Return Y to old shape
        nan_array = np.full((Y_shape[0], Y_shape[1], Y_shape[2]-Y_new_pert.shape[2], Y_shape[3]), np.nan)
        Y_new_pert = np.concatenate((Y_new_pert, nan_array), axis=2)

        if flip_dimensions:
            agent_order_inverse = np.argsort(agent_order)
            X_new_pert = X_new_pert[:, agent_order_inverse, :, :]
            Y_new_pert = Y_new_pert[:, agent_order_inverse, :, :]
        

        return X_new_pert, Y_new_pert
  

    def validate_plot_settings(self,plot_input, plot_spline):
        if plot_spline and not plot_input:
            raise ValueError("plot_spline can only be True if plot_input is also True.")
    

    def remove_nan_values(self, Y):
        # Calculate the maximum length where all values are non-NaN in the path lenght channel across all samples
        max_length = np.min(np.sum(~np.isnan(Y[:, :, :, 0]), axis=2)[:, 0])
        
        # Trim the array to the maximum length without NaN values in the path lenght channel
        Y_trimmed = Y[:, :, :max_length, :]
    
        return Y_trimmed
    
    def assert_only_one_true(self,*args):
        # Check that exactly one argument is True
        assert sum(args) == 1, "Assertion Error: Exactly one loss function must be activated."

    def masked_data(self, X, Y):
        # Check the edge case scenario where the vehicle is standing still, by checking if the first and last position are within 10 cm
        mask_values_X = np.abs(X[:,1,0,0]-X[:,1,-1,0]) < 0.1
        mask_values_Y = np.abs(Y[:,1,0,0]-Y[:,1,-1,0]) < 0.1

        return mask_values_X, mask_values_Y

    def create_data_plot(self, X, Y, agent, mask_values_X, mask_values_Y, flip_dimensions, spline_interval, spline, plot_input, plot_spline):
        # Flip the dimensions of the data if required
        X, Y, agent_order = self.flip_dimensions(X, Y, agent, flip_dimensions)

        # Create spline data
        spline_data = self.spline_data(X, Y, mask_values_X, mask_values_Y, flip_dimensions, spline_interval, spline)

        # Plot the data if required
        self.Plot_data(X, Y, spline_data, plot_input, plot_spline)

        return X, Y, agent_order, spline_data
    
    
    def Plot_data(self, X, Y, spline_data, plot_input, plot_spline):
        # Return early if plotting is not requested
        if plot_input == False:
            return
        
        # Iterate over each example in the data
        for i in range(X.shape[0]):
            plt.figure(figsize=(20,10))
            # Plot the spline data
            if plot_spline:
                plt.plot(spline_data[i,:,0], spline_data[i,:,1], marker='o', color='m', label='Spline plot',markersize=4,alpha=0.2)

            # Plot the input data
            for j in range(X.shape[1]):
                # Plot the past and future positions of the target and ego agents
                if j != X.shape[1]-1:
                    plt.plot(X[i,j,:,0], X[i,j,:,1], linestyle='-',linewidth=3, color='y', label='Past target agent')
                    plt.plot(Y[i,j,:-1,0], Y[i,j,:-1,1], linestyle='dashed',linewidth=3, color='y', label='Future target agent')
                    plt.plot((X[i,j,-1,0],Y[i,j,0,0]), (X[i,j,-1,1],Y[i,j,0,1]), linestyle='dashed',linewidth=3, color='y')
                else:
                    plt.plot(X[i,j,:,0], X[i,j,:,1], linestyle='-',linewidth=3, color='b',label='Past ego agent')
                    plt.plot(Y[i,j,:-1,0], Y[i,j,:-1,1], linestyle='dashed',linewidth=3, color='b', label='Future ego agent')
                    plt.plot((X[i,j,-1,0],Y[i,j,0,0]), (X[i,j,-1,1],Y[i,j,0,1]),linewidth=3, linestyle='dashed', color='b')
            
            # Set the plot limits and road lines
            offset = 10
            min_value_x = np.inf
            max_value_x = -np.inf
            min_value_y = np.inf
            max_value_y = -np.inf

            # Find plot limits
            for j in range(X.shape[1]):
                min_value_x = min(min_value_x, np.min(X[i,j,:,0]))
                min_value_x = min(min_value_x, np.min(Y[i,j,:,0]))

                max_value_x = max(max_value_x, np.max(X[i,j,:,0]))
                max_value_x = max(max_value_x, np.max(Y[i,j,:,0]))

                min_value_y = min(min_value_y, np.min(X[i,j,:,1]))
                min_value_y = min(min_value_y, np.min(Y[i,j,:,1]))

                max_value_y = max(max_value_y, np.max(X[i,j,:,1]))
                max_value_y = max(max_value_y, np.max(Y[i,j,:,1]))

            # Plot the arrow 
            plt.annotate('', xy=(Y[i,0,-1,0], Y[i,0,-1,1]), xytext=(Y[i,0,-2,0], Y[i,0,-2,1]),
                 size=20,arrowprops=dict(arrowstyle='-|>',linestyle=None,color="y",lw=3))
            plt.annotate('', xy=(Y[i,1,-1,0], Y[i,1,-1,1]), xytext=(Y[i,1,-2,0], Y[i,1,-2,1]),
                size=20,arrowprops=dict(arrowstyle='-|>',linestyle=None,color="b",lw=3))

            # Plot the dashed road lines
            y_dash = [0,0]
            x_min_dash = [min_value_x - offset, 4.5]
            x_max_dash = [-4.5, max_value_x + offset]

            x_dash = [0,0]
            y_min_dash = [min_value_y - 2 * offset, 4.5]
            y_max_dash = [-4.5, max_value_y + 2 * offset]

            plt.hlines(y_dash,x_min_dash,x_max_dash, linestyle='dashed', colors='k',linewidth=0.75)
            plt.vlines(x_dash,y_min_dash,y_max_dash, linestyle='dashed', colors='k',linewidth=0.75)
            
            # Plot the solid road lines
            y_solid = [-3.5, -3.5, 3.5, 3.5]
            x_min_solid = [min_value_x - offset, 3.5, min_value_x - offset, 3.5]
            x_max_solid = [-3.5, max_value_x + offset, -3.5, max_value_x + offset]

            x_solid = [-3.5, 3.5, 3.5, -3.5]
            y_min_solid = [min_value_y - 2 * offset, min_value_y - 2 * offset, 3.5, 3.5]
            y_max_solid = [-3.5, -3.5, max_value_y + 2 * offset, max_value_y + 2 * offset]

            plt.hlines(y_solid,x_min_solid,x_max_solid, linestyle="solid", colors='k')
            plt.vlines(x_solid,y_min_solid,y_max_solid, linestyle="solid", colors='k')

            # Set the plot limits
            plt.xlim(min_value_x - offset, max_value_x + offset)  
            plt.ylim(min_value_y - 2 * offset, max_value_y + 2 * offset)
            plt.axis('equal')
            plt.legend()
            plt.show()

    def flip_dimensions(self, X, Y, agent, flip_dimensions):
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
    
    def spline_data(self, X, Y, mask_values_X, mask_values_Y, flip_dimensions, spline_interval, spline):
        if spline == False:
            return None
        
        # Combine historical and future data
        Spline_input_values = np.concatenate((X,Y),axis=2)
        
        # Check edge case scenarios where agent is standing still
        if flip_dimensions:
            for batch_idx in range(X.shape[0]):
                # Check if the target agent is standing still in historical data
                if mask_values_X[batch_idx] == True:
                    # check if the agent moves in the future data -> store the first and last point of historical data -> store all future data -> set the rest to nan
                    if mask_values_Y[batch_idx] == False:
                            Spline_input_values[batch_idx,0,1,:] = Spline_input_values[batch_idx,0,X.shape[2],:]
                            Spline_input_values[batch_idx,0,2:2+Y.shape[2],:] = Spline_input_values[batch_idx,0,X.shape[2]:,:]
                            Spline_input_values[batch_idx, 0, 2+Y.shape[2]:,:] = np.nan

                    # check if the agent is standing still in the future data -> store the lastest future point -> set the rest to nan (bug in data recording)
                    else:
                        for i in reversed(range(Spline_input_values.shape[2])):
                            # Check if the new positions in the future data are smaller than the last position in the historical data
                            if Spline_input_values[batch_idx, 0, i, 0] < Spline_input_values[batch_idx, 0, 0, 0]:
                                Spline_input_values[batch_idx,0,1,:] = Spline_input_values[batch_idx,0,i,:]
                                Spline_input_values[batch_idx, 0, 2:,:] = np.nan
                                break
                
                # Remove values that are not monotonic For the spline function
                i = 1
                while i < Spline_input_values.shape[2]:
                    if Spline_input_values[batch_idx, 0, i, 0] > Spline_input_values[batch_idx, 0, i - 1, 0]:
                        Spline_input_values[batch_idx, 0, i-1:,:] = np.nan
                        break
                    else:
                        i += 1

        # Initialize spline data
        spline_data = np.zeros((X.shape[0],spline_interval,2))

        # Spline historical data
        for i in range(X.shape[0]):
            # Extract the spline data
            sample_spline = Spline_input_values[i,0,:,:]

            # Remove NaN values
            sample_spline = sample_spline[~np.isnan(sample_spline).any(axis=1)]

            # Flip data if it is not in the correct order for cubic spline funciton
            if sample_spline[0,0] > sample_spline[-1,0]:
                sample_spline[:,0] = np.flip(sample_spline[:,0])
            if sample_spline[0,1] > sample_spline[-1,1]:
                sample_spline[:,1] = np.flip(sample_spline[:,1])

            # Sample from cubic spline function
            Spline_function = CubicSpline(sample_spline[:,0], sample_spline[:,1])
            xs = np.linspace(sample_spline[0,0], sample_spline[-1,0], spline_interval)
            spline_data[i,:,0] = xs
            spline_data[i,:,1] = Spline_function(xs)

        return spline_data

    def convert_to_tensor(self, X, Y, spline_data):
        # Convert all inputs to tensors
        X = self.to_cuda_tensor(X)
        Y = self.to_cuda_tensor(Y)

        # Handle the optional spline data
        if spline_data is not None:
            spline_data = self.to_cuda_tensor(spline_data)
        else:
            spline_data = None
        
        return X, Y, spline_data
    
    def to_cuda_tensor(self,np_array):
        # Helper function to convert numpy arrays to CUDA tensors
        return torch.from_numpy(np_array).to(dtype=torch.float32, device='cuda')
    
    def get_control_actions(self, X, mask_values_X, flip_dimensions, new_input):
        # Initialize control action
        control_action = torch.zeros_like(new_input)

        # Initialize heading and velocity
        heading_init = torch.zeros(X.shape[0]).to(device='cuda')
        velocity_init = torch.zeros(X.shape[0]).to(device='cuda')

        for batch_idx in range(X.shape[0]):
            # Check if the target agent is standing still set all control actions to zero
            if mask_values_X[batch_idx] == True and flip_dimensions == True:
                # update initial velocity 
                velocity_init[batch_idx] = 0

                # update initial heading
                heading_init[batch_idx] = self.compute_heading(X, batch_idx, 0)

                # update control actions
                control_action[batch_idx,:,:,:] = 0 
                          
            else:
                # Retreive the time step size
                dt = self.kwargs['data_param']['dt']

                # Initialize storage for velocity and heading
                velocity = torch.zeros(X.shape[2]).to(device='cuda')
                angle = torch.zeros(X.shape[2]).to(device='cuda')

                # update initial velocity 
                velocity_init[batch_idx] = velocity[0] = self.compute_velocity(X, batch_idx, dt, 0)

                # update initial heading
                heading_init[batch_idx] = angle[0] = self.compute_heading(X, batch_idx,0)

                for i in range(X.shape[2]-1):
                    # Calculate velocity for next time step
                    velocity[i+1] = self.compute_velocity(X, batch_idx, dt, i)

                    # Update the control actions
                    control_action[batch_idx,0,i,0] = (velocity[i+1] - velocity[i]) / dt 

                    # Calculate the heading for the next time step   
                    angle[i+1] = self.compute_heading(X, batch_idx,i)

                    # Calculate the change of heading for the next time step
                    d_yaw_rate = (angle[i+1] - angle[i]) / dt

                    # Calculate the curvature 
                    curvature = d_yaw_rate / velocity[i]
                    control_action[batch_idx,0,i-1,1] = curvature 

        control_action[torch.isinf(control_action)] = 1e-6
        control_action.requires_grad = True 

        return control_action, heading_init, velocity_init
    
    def compute_heading(self, X, batch_idx, index):
        # Calculate dx and dy
        dx = X[batch_idx, 0, index + 1, 0] - X[batch_idx, 0, index, 0]  
        dy = X[batch_idx, 0, index + 1, 1] - X[batch_idx, 0, index, 1]  
        return torch.atan2(dy, dx) 
    
    def compute_velocity(self, X, batch_idx, dt, index):
        return torch.linalg.norm(X[batch_idx,0,index + 1,:] - X[batch_idx,0,index,:] , dim=-1, ord = 2) / dt
    

    def dynamical_model(self, new_input, control_action, velocity_init, heading_init):
        # Retreive the time step size
        dt = self.kwargs['data_param']['dt']

        # Adversarial position storage
        adv_position = new_input.clone().detach()

        # Update adversarial position based on dynamical model
        acc = control_action[:,0,:-1,0]
        cur = control_action[:,0,:-1,1]

        # Calculate the velocity for all time steps
        Velocity = torch.cumsum(acc, dim=1) * dt + velocity_init.unsqueeze(1)

        # Calculte the change of heading for all time steps
        D_yaw_rate = Velocity * cur
        D_yaw_rate = torch.cat((D_yaw_rate[:, -1:], D_yaw_rate[:, :-1]), dim=1)

        # Calculate Heading for all time steps
        Heading = torch.cumsum(D_yaw_rate, dim=1) * dt + heading_init.unsqueeze(1)

        # Calculate the new position for all time steps
        adv_position[:, 0, 1:, 0] = torch.cumsum(Velocity * torch.cos(Heading), dim=1) * dt + adv_position[:, 0, 0, 0].unsqueeze(1)
        adv_position[:, 0, 1:, 1] = torch.cumsum(Velocity * torch.sin(Heading), dim=1) * dt + adv_position[:, 0, 0, 1].unsqueeze(1)

        return adv_position
    
    def calculate_loss(self,X,X_new,Y,Y_new,Pred_t,ADE_loss,ADE_loss_barrier,ADE_loss_adv_future,ADE_loss_adv_future_barrier,collision_loss,collision_loss_barrier,fake_collision_loss,hide_collision_loss,log_barrier,ADVDO_barrier,spline_barrier,distance_threshold,log_value,spline_data):

        # Add  regularization loss to adversarial input using barrier function
        if log_barrier:
            barrier_output = self.barrier_log_function(distance_threshold, X_new, X, log_value)
        elif ADVDO_barrier:
            barrier_output = self.AVDDO_barrier_function(X_new, X, distance_threshold)
        elif spline_barrier:
            barrier_output = self.barrier_log_function_spline(distance_threshold, X_new, spline_data, log_value)
            

        # Calculate the total loss
        if ADE_loss:
            losses = self.ADE_loss_function(self, Y, Pred_t)
        elif ADE_loss_barrier:
            losses = self.ADE_loss_function(self, Y, Pred_t) + barrier_output
        elif ADE_loss_adv_future:
            losses = self.ADE_adv_future(self, Y_new, Pred_t)
        elif ADE_loss_adv_future_barrier:
            losses = self.ADE_adv_future(self, Y_new, Pred_t) + barrier_output
        elif collision_loss:
            losses = -self.collision_loss_function(Y, Pred_t)
        elif collision_loss_barrier:
            losses = -self.collision_loss_function(Y, Pred_t) + barrier_output
        elif fake_collision_loss:
            losses = -self.collision_loss_function(Y, Pred_t) + barrier_output 
        elif hide_collision_loss:
            losses = -self.collision_loss_adv_future(Y_new, Y) + barrier_output 

        return losses
    
    def ADE_loss_function(self, Y, Pred_t):
        return torch.mean(torch.mean(torch.linalg.norm(Y[:,0,:,:].unsqueeze(1) - Pred_t[:,:,:,:], dim=-1 , ord = 2), dim=-1),dim=-1)
    
    def ADE_adv_future(self, Y_new, Pred_t):
        return torch.mean(torch.mean(torch.linalg.norm(Y_new[:,0,:,:].unsqueeze(1) - Pred_t[:,:,:,:], dim=-1 , ord = 2), dim=-1),dim=-1)
    
    def collision_loss_function(self, Y, Pred_t):
        return torch.mean(torch.linalg.norm(Y[:,1,:,:].unsqueeze(1) - Pred_t[:,:,:,:], dim=-1 , ord = 2).min(dim=-1).values,dim=-1)
    
    def collision_loss_adv_future(self, Y_new, Y):
        return torch.mean(torch.linalg.norm(Y[:,1,:,:] - Y_new[:,0,:,:], dim=-1 , ord = 2).min(dim=-1).values,dim=-1)

    def barrier_log_function(self, distance_threshold, X_new, X, log_value):
        barrier_norm = torch.norm(X_new[:,0,:,:] - X[:,0,:,:], dim = -1)
        barrier_log = torch.log(distance_threshold - barrier_norm)
        barrier_log_new = barrier_log / torch.log(torch.tensor(log_value))
        barrier_output = torch.mean(barrier_log_new,dim=-1)
        return barrier_output
    
    def AVDDO_barrier_function(self, X_new, X, distance_threshold_past):
        barrier_norm = torch.norm(X_new[:,0,:,:] - X[:,0,:,:], dim = -1)
        barrier_output = -((barrier_norm / distance_threshold_past) - torch.sigmoid(barrier_norm / distance_threshold_past) + 0.5).sum(dim=-1)
        return barrier_output
    
    def barrier_log_function_spline(self, distance_threshold_past, X_new, spline_data, spline_value_past):
        distance = torch.cdist(X_new[:,0,:,:], spline_data, p=2)
        min_indices = torch.argmin(distance, dim=-1)
        closest_points = torch.gather(spline_data.unsqueeze(1).expand(-1, X_new.shape[2], -1, -1), 2, min_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, spline_data.size(-1)))
        closest_points = closest_points.squeeze(2)
        barrier_norm = torch.norm(X_new[:,0,:,:] - closest_points, dim = -1)
        barrier_output = torch.log(distance_threshold_past - barrier_norm)
        barrier_log_new = barrier_output / torch.log(torch.tensor(spline_value_past))
        barrier_output = torch.mean(barrier_log_new,dim=-1)
        return barrier_output
    
    def detach_tensor(self, X_new_adv, Y_new_adv, Pred_t):
        # Detach tensors, move them to CPU, and convert them to NumPy arrays
        X_new_pert = X_new_adv.detach().cpu().numpy()
        Y_new_pert = Y_new_adv.detach().cpu().numpy()
        Pred_t = Pred_t.detach().cpu().numpy()

        # Calculate the mean of the predicted future positions
        Pred_t = np.mean(Pred_t, axis=1)
        
        return X_new_pert, Y_new_pert, Pred_t
    
    def plot_results(self, X_copy, X_new_pert, Y_copy, Y_new_pert, Pred_t, loss_store, plot_loss, future_action,static_adv_scene,animated_adv_scene):
        # Plot the loss over the iterations
        if plot_loss:
            plt.figure(0)
            plt.plot(loss_store, marker='o', linestyle='-')
            plt.title('Loss for samples')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.show()

        # Plot the static adversarial scene
        if static_adv_scene:
            for i in range(X_copy.shape[0]):
                plt.figure()
                plt.plot(X_copy[i,0,:,0], X_copy[i,0,:,1], marker='o', linestyle='-', color='b', label='Past original adverserial')
                plt.plot(X_copy[i,1,:,0], X_copy[i,1,:,1], marker='o', linestyle='-', color='g',label='Past original other')
                plt.plot(X_new_pert[i,0,:,0], X_new_pert[i,0,:,1], marker='o', linestyle='-', color='r', label='Past perturbed adverserial')
                plt.plot(Pred_t[i,:,0], Pred_t[i,:,1], marker='o', linestyle='-', color='c', label='Future adversarial predict')
                plt.plot([X_new_pert[i,0,-1,0],Pred_t[i,0,0]],[X_new_pert[i,0,-1,1],Pred_t[i,0,1]], marker='o', linestyle='-', color='c')
                plt.plot(Y_copy[i,0,:,0], Y_copy[i,0,:,1], marker='o', linestyle='-', color='m',label='Future original adversarial')
                plt.plot([X_copy[i,0,-1,0],Y_copy[i,0,0,0]],[X_copy[i,0,-1,1],Y_copy[i,0,0,1]], marker='o', linestyle='-', color='m')
                plt.plot(Y_copy[i,1,:,0], Y_copy[i,1,:,1], marker='o', linestyle='-', color='y', label='Future original other')
                plt.plot([X_copy[i,1,-1,0],Y_copy[i,1,0,0]],[X_copy[i,1,-1,1],Y_copy[i,1,0,1]], marker='o', linestyle='-', color='y')

                if future_action:
                    plt.plot(Y_new_pert[i,0,:,0], Y_new_pert[i,0,:,1], marker='o', linestyle='-', color='k', label='Future adversarial expected')
                    plt.plot([X_new_pert[i,0,-1,0],Y_new_pert[i,0,0,0]],[X_new_pert[i,0,-1,1],Y_new_pert[i,0,0,1]], marker='o', linestyle='-', color='k')
                plt.axis('equal')
                plt.legend()
                plt.show()

        # Plot the animated adversarial scene  
        if animated_adv_scene:
            for i in range(X_copy.shape[0]):
                #smooth line
                num_interpolations = 5

                ego_original = np.concatenate((X_copy[i,0,:,:],Y_eval[i,0,:,:]),axis=0)
                ego_original, flip_value = self.delete_after_flip_x(ego_original)
                ego_original, length = self.interpolate_points(ego_original,flip_value, num_interpolations,X_copy.shape[2])

                x_1 = ego_original[:,0]
                y_1 = ego_original[:,1]

                tar_original = np.concatenate((X_copy[i,1,:,:],Y_eval[i,1,:,:]),axis=0)
                tar_original, flip_value = self.delete_after_flip_x(tar_original)
                tar_original,_ = self.interpolate_points(tar_original,flip_value, num_interpolations,X_copy.shape[2])

                x_2 = tar_original[:,0]
                y_2 = tar_original[:,1]

                adversarial = np.concatenate((X_new_pert[i,0,:,:],Pred_t[i,:,:]),axis=0)
                adversarial, flip_value = self.delete_after_flip_x(adversarial)
                adversarial, _ = self.interpolate_points(adversarial,flip_value, num_interpolations,X_copy.shape[2])

                x_3 = adversarial[:,0]
                y_3 = adversarial[:,1]

                rect_width = 4.1
                rect_height = 1.7
                rect1 = patches.Rectangle((0, 0), rect_width, rect_height, edgecolor='none', facecolor='blue',label='Ego-agent')
                rect2 = patches.Rectangle((0, 0), rect_width, rect_height, edgecolor='none', facecolor='green',label='Target-agent')
                rect3 = patches.Rectangle((0, 0), rect_width, rect_height, edgecolor='none', facecolor='red',label='Adversarial agent')

                fig = plt.figure(figsize = (18,12), dpi=1920/16)

                ax = fig.add_subplot(2,2,1)
                ax1 = fig.add_subplot(2,2,2)
                ax2 = fig.add_subplot(2,1,2)
                # line1, = ax.plot(x_1, y_1, color='b',label='Ego-agent')
                # line2, = ax.plot(x_2, y_2, color='g',label='Target-agent')
                # line3, = ax.plot(x_3, y_3, color='r',label='Adversarial agent')

                ax.add_patch(rect1)
                ax.add_patch(rect2)
                ax.add_patch(rect3)

                # def update(num, x_1, y_1, x_2, y_2, x_3, y_3, line1,line2,line3):

                def update(num, x_1, y_1, x_2, y_2, x_3, y_3):
                    # if num < length:
                    #     line1.set_data(x_1[:num], y_1[:num])
                    #     line2.set_data(x_2[:num], y_2[:num])
                    #     line3.set_data(x_3[:num], y_3[:num])
                    # if num >= length:
                    #     line1.set_linestyle('dashed')
                    #     line2.set_linestyle('dashed')
                    #     line3.set_linestyle('dashed')
                    #     line1.set_data(x_1[length:num], y_1[length:num])
                    #     line2.set_data(x_2[length:num], y_2[length:num])
                    #     line3.set_data(x_3[length:num], y_3[length:num])
                    #     ax.plot(ego_original[:length,0], ego_original[:length,1], color='b')
                    #     ax.plot(tar_original[:length,0], tar_original[:length,1], color='g')
                    #     ax.plot(adversarial[:length,0], adversarial[:length,1], color='r')
                        

                    # Set position based on bottom left corner (assuming this is your anchor point)

                    # Update rectangle 1 (ego-original)
                    dx = x_1[num + 1] - x_1[num]
                    dy = y_1[num + 1] - y_1[num]
                    angle_rad = np.arctan2(dy, dx)

                    shift_x_1 = (rect_height / 2) * np.sin(angle_rad) - (rect_width / 2) * np.cos(angle_rad)
                    shift_y_1 = -(rect_height / 2) * np.cos(angle_rad) - (rect_width / 2) * np.sin(angle_rad)
                    rect1.set_xy([x_1[num-1] + shift_x_1, y_1[num-1] + shift_y_1])
                    angle = np.arctan2(dy, dx) * (180 / np.pi)  # Convert radians to degrees
                    rect1.set_angle(angle)

                    #Calculate direction vector
                    dx = x_2[num + 1] - x_2[num]
                    dy = y_2[num + 1] - y_2[num]

                    angle_rad = np.arctan2(dy, dx) 

                    shift_x_2 = (rect_height/2) * np.sin(angle_rad) - (rect_width/2) * np.cos(angle_rad)
                    shift_y_2 = -(rect_height/2) * np.cos(angle_rad) - (rect_width/2) * np.sin(angle_rad )

                    rect2.set_xy([x_2[num-1] + shift_x_2, y_2[num-1] + shift_y_2])
                    dx = x_2[num + 1] - x_2[num]
                    dy = y_2[num + 1] - y_2[num]
                    angle = np.arctan2(dy, dx) * (180 / np.pi)  # Convert radians to degrees

                    # Apply rotation around the bottom left corner
                    rect2.set_angle(angle)

                    # Calculate direction vector for adversarial trajectory
                    dx = x_3[num + 1] - x_3[num]
                    dy = y_3[num + 1] - y_3[num]
                    angle_rad = np.arctan2(dy, dx)

                    # Update rectangle 3 (adversarial)
                    shift_x_3 = (rect_height / 2) * np.sin(angle_rad) - (rect_width / 2) * np.cos(angle_rad)
                    shift_y_3 = -(rect_height / 2) * np.cos(angle_rad) - (rect_width / 2) * np.sin(angle_rad)
                    rect3.set_xy([x_3[num-1] + shift_x_3, y_3[num-1] + shift_y_3])
                    angle = np.arctan2(dy, dx) * (180 / np.pi)  # Convert radians to degrees
                    rect3.set_angle(angle)

                    return 

                min_value_x = min(np.min(x_1), np.min(x_2), np.min(x_3))
                max_value_x = max(np.max(x_1), np.max(x_2), np.max(x_3))
                min_value_y = min(np.min(y_1), np.min(y_2), np.min(y_3))
                max_value_y = max(np.max(y_1), np.max(y_2), np.max(y_3))

                offset = 10

                y1 = [0,0]
                x_min1 = [min_value_x - offset, +4.5]
                x_max1 = [-4.5, max_value_x + offset]

                x1 = [0,0]
                y_min1 = [min_value_y - 2 * offset, 4.5]
                y_max1 = [-4.5, max_value_y + 2 * offset]
                
                ax.hlines(y1,x_min1,x_max1, linestyle='dashed', colors='k',linewidth=0.75)
                ax.vlines(x1,y_min1,y_max1, linestyle='dashed', colors='k',linewidth=0.75)


                y = [-3.5, -3.5, 3.5, 3.5]

                x_min = [min_value_x - offset, 3.5, min_value_x - offset, 3.5]
                x_max = [-3.5, max_value_x + offset, -3.5, max_value_x + offset]

                x = [-3.5, 3.5, 3.5, -3.5]

                y_min = [min_value_y - 2 * offset, min_value_y - 2 * offset, 3.5, 3.5]
                y_max = [-3.5, -3.5, max_value_y + 2 * offset, max_value_y + 2 * offset]

                ax.hlines(y,x_min,x_max, linestyle="solid", colors='k')
                ax.vlines(x,y_min,y_max, linestyle="solid", colors='k')
                
                ax.set_aspect('equal')
                ax.set_xlim(min_value_x - offset, max_value_x + offset)   # Set x-axis limits
                ax.set_ylim(min_value_y - 2 * offset, max_value_y + 2 * offset)
                ax.legend()
                ax.set_title('Animation of the adversarial scene')

                # ani = animation.FuncAnimation(fig, update, len(x_1)-1, fargs=[x_1, y_1, x_2, y_2, x_3, y_3, line1, line2, line3],
                #                             interval=100/num_interpolations, blit=False)
                
                ani = animation.FuncAnimation(fig, update, len(x_1)-1, fargs=[x_1, y_1, x_2, y_2, x_3, y_3],
                                            interval=100/num_interpolations, blit=False)
                
                ax1.plot(X_new_pert[i,0,:,0], X_new_pert[i,0,:,1], linestyle='-', color='r',label='Adversarial agent')
                ax1.plot(X_copy[i,1,:,0], X_copy[i,1,:,1], linestyle='-', color='g',label='Target agent')
                ax1.plot(Y_eval[i,1,:-1,0], Y_eval[i,1,:-1,1], linestyle='dashed', color='g',alpha=0.7)
                ax1.plot([X_copy[i,1,-1,0],Y_eval[i,1,0,0]],[X_copy[i,1,-1,1],Y_eval[i,1,0,1]], linestyle='dashed', color='g',alpha=0.7)
                ax1.plot(X_new_pert[i,0,:,0], X_new_pert[i,0,:,1], linestyle='dotted', color='r')
                ax1.plot(Pred_t[i,:-1,0], Pred_t[i,:-1,1], linestyle='dashdot', color='r',alpha=0.7)
                ax1.plot([X_new_pert[i,0,-1,0],Pred_t[i,0,0]],[X_new_pert[i,0,-1,1],Pred_t[i,0,1]], linestyle='dashdot', color='r',alpha=0.7)
                ax1.set_aspect('equal')

                # Second legend 'imaginary' lines
                line_solid = mlines.Line2D([], [], color='black', linestyle='-', label=f'Benign past')
                line_dashed = mlines.Line2D([], [], color='black', linestyle='dashed', label=f'Benign future')
                line_dotted = mlines.Line2D([], [], color='black', linestyle='dotted', label=f'Adversarial past')
                line_dashdot = mlines.Line2D([], [], color='black', linestyle='dashdot', label=f'Adversarial future')

                leg1 = ax1.legend()
                leg2 = ax1.legend(handles=[line_solid, line_dashed,line_dotted,line_dashdot], loc='best')
                leg2.set_frame_on(False)
                ax1.add_artist(leg1)
                ax1.set_title('Adversarial past scene')

                ax1.hlines(y1,x_min1,x_max1, linestyle='dashed', colors='k',linewidth=0.75)
                ax1.vlines(x1,y_min1,y_max1, linestyle='dashed', colors='k',linewidth=0.75)

                ax1.hlines(y,x_min,x_max, linestyle="solid", colors='k')
                ax1.vlines(x,y_min,y_max, linestyle="solid", colors='k')

                offset = 0.5

                min_x_new = min(np.min(X_copy[i,1,:,0]), np.min(X_new_pert[i,0,:,0]))
                max_x_new = max(np.max(X_copy[i,1,:,0]), np.max(X_new_pert[i,0,:,1]))

                min_y_new = min(np.min(X_copy[i,1,:,1]), np.min(X_new_pert[i,0,:,0]))
                max_y_new = max(np.max(X_copy[i,1,:,1]), np.max(X_new_pert[i,0,:,1]))

                ax1.set_xlim(min_x_new - offset, max_x_new + offset)   # Set x-axis limits
                ax1.set_ylim(min_y_new - offset, max_y_new + offset)
                
                ax2.plot(X_copy[i,0,:,0], X_copy[i,0,:,1], linestyle='-', color='b',label='Ego agent')
                ax2.plot(X_copy[i,1,:,0], X_copy[i,1,:,1], linestyle='-', color='g',label='Target agent')
                ax2.plot(X_new_pert[i,0,:,0], X_new_pert[i,0,:,1], linestyle='-', color='r',label='Adversarial agent')
                ax2.plot(Pred_t[i,:-1,0], Pred_t[i,:-1,1], linestyle='dashdot', color='r')
                ax2.plot([X_new_pert[i,0,-1,0],Pred_t[i,0,0]],[X_new_pert[i,0,-1,1],Pred_t[i,0,1]], linestyle='dashdot', color='r')
                ax2.plot(Y_eval[i,0,:-1,0], Y_eval[i,0,:-1,1], linestyle='dashed', color='b')
                ax2.plot([X_copy[i,0,-1,0],Y_eval[i,0,0,0]],[X_copy[i,0,-1,1],Y_eval[i,0,0,1]], linestyle='-', color='b')
                ax2.plot(Y_eval[i,1,:-1,0], Y_eval[i,1,:-1,1], linestyle='dashed', color='g')
                ax2.plot([X_copy[i,1,-1,0],Y_eval[i,1,0,0]],[X_copy[i,1,-1,1],Y_eval[i,1,0,1]], linestyle='dashed', color='g')

                ax2.annotate('', xy=(Y_eval[i,0,-1,0], Y_eval[i,0,-1,1]), xytext=(Y_eval[i,0,-2,0], Y_eval[i,0,-2,1]),
                 size=20,arrowprops=dict(arrowstyle='-|>',linestyle=None,color="b"))
                ax2.annotate('', xy=(Y_eval[i,1,-1,0], Y_eval[i,1,-1,1]), xytext=(Y_eval[i,1,-2,0], Y_eval[i,1,-2,1]),
                 size=20,arrowprops=dict(arrowstyle='-|>',linestyle=None,color="g"))
                ax2.annotate('', xy=(Pred_t[i,-1,0], Pred_t[i,-1,1]), xytext=(Pred_t[i,-2,0], Pred_t[i,-2,1]),
                 size=20,arrowprops=dict(arrowstyle='-|>',linestyle=None,color="r"))

                ax2.set_aspect('equal')
                offset = 2
                ax2.set_xlim(min_value_x - offset, max_value_x + offset)   # Set x-axis limits
                ax2.set_ylim(min_value_y - offset, max_value_y + offset)
                
                ax2.hlines(y1,x_min1,x_max1, linestyle='dashed', colors='k',linewidth=0.75)
                ax2.vlines(x1,y_min1,y_max1, linestyle='dashed', colors='k',linewidth=0.75)

                ax2.hlines(y,x_min,x_max, linestyle="solid", colors='k')
                ax2.vlines(x,y_min,y_max, linestyle="solid", colors='k')

                ax2.legend()
                ax2.set_title('Adversarial scene static')

                ani.save(f'basic_animation_new-{np.random.rand(1)}.mp4')
                
                                
                plt.show()

        
    def delete_after_flip_x(self, arr):
        # Check if the array is empty or has only one element
        if arr.shape[0] <= 1:
            return arr

        # Determine if the array is initially increasing or decreasing along the x-axis
        initial_trend = arr[0, 0] < arr[1, 0]

        offset = 0.01

        # Iterate through the array along the x-axis starting from the second element
        for i in range(1, arr.shape[0]):
            # Check if the trend flips
            current_trend = arr[i-1, 0] < arr[i, 0]
            if current_trend != initial_trend:
                arr1 =  arr[:i]  # Return the array up to the flip point
                arr2 = arr[i-1:]  # Return the array after the flip point
                
                return arr1, arr2
                    
        return arr, None  # Return the original array if no flip is found
    
        
                
    def interpolate_points(self,data,flip_value, num_interpolations,shape):
        """
        Interpolate points between each existing point in the dataset.

        Parameters:
            data (numpy.ndarray): Array containing the original dataset with shape (n, 2), where n is the number of points.
            num_interpolations (int): Number of points to interpolate between each existing point.

        Returns:
            numpy.ndarray: Array containing the interpolated points.
        """

        Flip = False

        length = None

        if data[0,0] > data[-1,0]:
            data = np.flip(data, axis=0)
            Flip = True

        if flip_value is None:
            # Create a spline interpolation function using all points
            # spline = interp1d(data[:, 0], data[:, 1], kind = 'cubic',bounds_error=False)
            spline = CubicSpline(data[:, 0], data[:, 1])
        else:
            # Create a spline interpolation function using all points
            # spline1 = interp1d(data[:, 0], data[:, 1], kind = 'cubic',bounds_error=False)
            # spline2 = interp1d(flip_value[:, 0], flip_value[:, 1], kind = 'cubic',bounds_error=False)
            spline1 = CubicSpline(data[:, 0], data[:, 1])
            spline2 = CubicSpline(flip_value[:, 0], flip_value[:, 1])

        # Interpolate points between each existing point
        interpolated_points = []
        if flip_value is None:
            for i in range(len(data[:, 0]) - 1):
                x_interval = np.linspace(data[i, 0], data[i+1, 0], num_interpolations + 1)
                y_interval = spline(x_interval)
                if i == len(data[:, 0]) - 1:
                    interpolated_points.extend(zip(x_interval, y_interval))
                else:
                    interpolated_points.extend(zip(x_interval[:-1], y_interval[:-1]))
                
                if shape-1 == i:
                    length = len(interpolated_points)
        else:
            spline_1_list = []
            for i in range(len(data[:, 0]) - 1):
                x_interval_1 = np.linspace(data[i, 0], data[i+1, 0], num_interpolations + 1)
                y_interval_1 = spline1(x_interval_1)
                spline_1_list.extend(zip(x_interval_1[:-1], y_interval_1[:-1]))

                if shape-1 == i:
                    length = len(spline_1_list)
                
            spline_2_list = []
            for i in range(len(flip_value) - 1):
                x_interval_2 = np.linspace(flip_value[i, 0], flip_value[i+1, 0], num_interpolations + 1)
                y_interval_2 = spline2(x_interval_2)
                if i == len(flip_value) - 1:
                    spline_2_list.extend(zip(x_interval_2, y_interval_2))
                else: 
                    spline_2_list.extend(zip(x_interval_2[:-1], y_interval_2[:-1]))

            if Flip:
                spline_1 = np.flip(np.array(spline_1_list),axis=0)
            else:
                spline_1 = np.array(spline_1_list)
            spline_2 = np.array(spline_2_list)

            return np.concatenate((spline_1,spline_2),axis=0), length

        # Convert the list of tuples to a NumPy array
        if Flip:
            interpolated_points = np.flip(np.array(interpolated_points),axis=0)
        else:
            interpolated_points = np.array(interpolated_points)

        return interpolated_points, length

    
    def adversarial_smoothing(self, X_pert, X, Y_pert_prediction, Y, T, Domain):
        X_pert_copy = X_pert.copy()

        flip_dimensions = False

        if flip_dimensions:
            X_pert[:, [1, 0], :, :] = X_pert[:, [0, 1], :, :]
            X_pert_copy[:, [1, 0], :, :] = X_pert_copy[:, [0, 1], :, :]
            X[:, [1, 0], :, :] = X[:, [0, 1], :, :]
            Y_pert_prediction[:, [1, 0], :, :] = Y_pert_prediction[:, [0, 1], :, :]
            Y[:, [1, 0], :, :] = Y[:, [0, 1], :, :]

        X_pert = torch.from_numpy(X_pert).to(dtype = torch.float32)
        X = torch.from_numpy(X).to(dtype = torch.float32)
        Y = torch.from_numpy(Y).to(dtype = torch.float32)

            
        
        # Setting adversarial smoothing parameters
        num_noised_samples = 5
        sigmas = [0.1, 0.2]
        analyze_pert = False
        num_steps = Y_pert_prediction.shape[2]
        num_steps = 15 #fix this

        # Plot settings
        plot_figure = True
        limit_X = False
        limit_Y = False

        # Create list to store values for all sigmas
        outputs_per_sigma = [[] for _ in sigmas]
        perturbation_per_sigma = [[] for _ in sigmas]

        # Apply randomized adversarial smoothing
        for i, sigma in enumerate(sigmas):
            for _ in range(num_noised_samples):
                if analyze_pert:
                    noise = torch.randn_like(X_pert) * sigma
                    input_data = X_pert +  noise
                else:
                    noise = torch.randn_like(X) * sigma
                    input_data = X +  noise

                Pred_t = self.pert_model.predict_batch_tensor(input_data, T, Domain, num_steps)
                Pred_t = torch.mean(Pred_t, dim=1)

                outputs_per_sigma[i].append(Pred_t.detach().cpu().numpy())
                perturbation_per_sigma[i].append(input_data.detach().cpu().numpy())  

        Y = Y.detach().cpu().numpy()

        if plot_figure:
            for i in range(X_pert_copy.shape[0]):
                fig, axes = plt.subplots(len(sigmas), 1, figsize=(8, 6 * len(sigmas)))
                for j, sigma_outputs in enumerate(outputs_per_sigma):
                    sigma_outputs = np.array(sigma_outputs)
                    outputs = np.mean(sigma_outputs, axis=0)

                    filtered_outputs = []
                    
                    for k in range(sigma_outputs.shape[0]):
                        if limit_Y:
                            if np.abs(sigma_outputs[k,i, :, 1])[-1] < 3.5:
                                axes[j].plot(sigma_outputs[k,i, :, 0], sigma_outputs[k,i, :, 1], marker='o', linestyle='-', color='b',ms=1, label='Prediction with smoothing')
                                filtered_outputs.append(sigma_outputs[k,i,:,:])
                        elif limit_X:
                            if np.abs(sigma_outputs[k,i, :, 0])[-1] < 3.5:
                                axes[j].plot(sigma_outputs[k,i, :, 0], sigma_outputs[k,i, :, 1], marker='o', linestyle='-', color='b',ms=1, label='Prediction with smoothing')
                                filtered_outputs.append(sigma_outputs[k,i,:,:])
                        else:
                            axes[j].plot(sigma_outputs[k,i, :, 0], sigma_outputs[k,i, :, 1], marker='o', linestyle='-', color='b',ms=1)

                    axes[j].plot(sigma_outputs[-1,i, :, 0], sigma_outputs[k,i, :, 1], marker='o', linestyle='-', color='b',ms=1, label='Prediction with smoothing')

                    if limit_X or limit_Y:
                        filtered_outputs = np.array(filtered_outputs)
                        mean_output = np.mean(filtered_outputs, axis=0)
                        axes[j].plot(mean_output[:, 0], mean_output[:, 1], marker='o', linestyle='-', color='k', ms=1, label='Mean Prediction')

                    axes[j].plot(outputs[i, :, 0], outputs[i, :, 1], marker='o', linestyle='-', color='k',ms=1, label='mean randomized_smoothing')
                    axes[j].plot(Y[i, 0, :, 0], Y[i, 0, :, 1], marker='o', linestyle='-', color='r',ms=1, label='Future original')

                    if analyze_pert:
                        axes[j].plot(X_pert_copy[i, 0, :, 0], X_pert_copy[i, 0, :, 1], marker='o', linestyle='-', color='g',ms=1)
                    else:
                        axes[j].plot(X[i, 0, :, 0], X[i, 0, :, 1], marker='o', linestyle='-', color='g',ms=1,label='Past original')

                    if analyze_pert:
                        axes[j].plot(Y_pert_prediction[i, 0, :, 0], Y_pert_prediction[i, 0, :, 1], marker='o', linestyle='-', color='y',ms=1)


                    axes[j].axis('equal')
                    axes[j].set_title(f'Sigma = {sigmas[j]}')
                    axes[j].set_xlabel('Sample')
                    axes[j].grid(True)
                    axes[j].legend()
                
                plt.tight_layout()
                plt.show()

        plot_debug = True

        if plot_debug:
            for i in range(X_pert_copy.shape[0]):
                for j, (sigma_outputs, sigma_perturbations) in enumerate(zip(outputs_per_sigma,perturbation_per_sigma)):
                    sigma_outputs = np.array(sigma_outputs)
                    sigma_perturbations = np.array(sigma_perturbations)
                    for k in range(len(sigma_outputs)):
                        plt.figure()
                        plt.plot(sigma_outputs[k,i, :, 0], sigma_outputs[k,i, :, 1], marker='o', linestyle='-', color='b', label='Prediction with smoothing')
                        plt.plot(sigma_perturbations[k,i,0, :, 0], sigma_perturbations[k,i,0, :, 1], marker='o', linestyle='-', color='g', label='Perturbation')
                        plt.plot(Y[i, 0, :, 0], Y[i, 0, :, 1], marker='o', linestyle='-', color='r', label='Original')
                        plt.axis('equal')
                        plt.title(f'ADE for sigma = {sigmas[j]}, sample {i} and perturbation {k}')
                        plt.legend()
                        plt.show()

        return outputs


    def set_batch_size(self):
        '''
        This function sets the batch size for the perturbation method.

        It must add a attribute self.batch_size to the class.

        Returns
        -------
        None.

        '''

        self.batch_size = 1

        # TODO: Implement this function, you can decide here if you somehow rely on self.pert_model, if possible, or instead use a fixed value

        # raise AttributeError('This function has to be implemented in the actual perturbation method.')

    
    def requirerments(self):
        '''
        This function returns the requirements for the data to be perturbed.

        It returns a dictionary, that may contain the following keys:

        n_I_max : int (optional)
            The number of maximum input timesteps.
        n_I_min : int (optional)
            The number of minimum input timesteps.

        n_O_max : int (optional)
            The number of maximum output timesteps.
        n_O_min : int (optional)
            The number of minimum output timesteps.

        dt : float (optional)
            The time step of the data.
        

        Returns
        -------
        dict
            A dictionary with the required keys and values.

        '''

        # TODO: Implement this function, use self.pert_model to get the requirements of the model.

        return {}
    
# mask = ~torch.isnan(adv_position[:, 0, i + 1, 0])  
    
                # d_yaw_rate = velocity[mask] * control_action[mask, 0, i, 1] 
                # heading[mask] = heading[mask] + d_yaw_rate * dt

                # adv_position[mask, 0, i+1, 0] = velocity[mask] * torch.cos(heading[mask]) * dt + adv_position[mask, 0, i, 0]
                # adv_position[mask, 0, i+1, 1] = velocity[mask] * torch.sin(heading[mask]) * dt + adv_position[mask, 0, i, 1]

                # velocity[mask] = control_action[mask, 0, i, 0] * dt + velocity[mask] 