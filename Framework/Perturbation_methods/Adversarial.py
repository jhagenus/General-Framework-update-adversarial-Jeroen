from perturbation_template import perturbation_template
import pandas as pd
import os
import numpy as np
import importlib
from Data_sets.data_interface import data_interface
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib import gridspec

from Adversarial_classes.control_action import Control_action
from Adversarial_classes.helper import Helper
from Adversarial_classes.loss import Loss
from Adversarial_classes.plot import Plot
from Adversarial_classes.smoothing import Smoothing
from Adversarial_classes.spline import Spline

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
        # Plot input data and spline (if plot is True -> plot_spline can be set on True) 
        plot_input = True
        plot_spline = True

        # Spline settings
        spline = True 
        spline_interval = 100

        # Plot the loss over the iterations
        plot_loss = True
        loss_store = []

        # Plot the adversarial scene
        static_adv_scene = True
        animated_adv_scene = True

        # Car size
        car_length = 4.1
        car_width = 1.7

        # validation settings
        Helper.validate_plot_settings(plot_input, plot_spline)
        Helper.validate_plot_settings(spline, plot_spline)

        # Change ego and tar vehicle -> (Important to keep this on True to perturb the agent that turns left)
        flip_dimensions = True

        # Initialize parameters
        iter_num = 2
        epsilon_acc = 7
        epsilon_curv = 0.2

        # Learning decay
        learning_rate_decay = True
        gamma = 1

        # Learning rate
        alpha_acc = 3
        alpha_curv = 0.02

        # Randomized smoothing 
        smooth_perturbed_data = True
        smooth_unperturbed_data = True
        num_samples = 2
        sigmas = [0.02]
        plot_smoothing = True

        # remove nan from input and remember old shape
        Y_shape = Y.shape
        Y = Helper.remove_nan_values(Y)

        # Select loss function ()
        #ADE loss
        ADE_loss = False
        ADE_loss_barrier = False
        ADE_loss_adv_future = False
        ADE_loss_adv_future_barrier = False

        # Collision loss
        collision_loss = True
        collision_loss_barrier = False
        fake_collision_loss = False
        fake_collision_loss_barrier = False
        hide_collision_loss = False
        hide_collision_loss_barrier = False

        # check if only one loss function is activated
        Helper.assert_only_one_true(
                ADE_loss,
                ADE_loss_barrier,
                ADE_loss_adv_future,
                ADE_loss_adv_future_barrier,
                collision_loss,
                collision_loss_barrier,
                fake_collision_loss,
                fake_collision_loss_barrier,
                hide_collision_loss,
                hide_collision_loss_barrier
            )

        # Barrier function
        log_barrier = True
        ADVDO_barrier = False
        spline_barrier = False

        # check if only one barrier function is activated
        Helper.assert_only_one_true(
                log_barrier,
                ADVDO_barrier,
                spline_barrier
            )

        # Barrier function parameters
        distance_threshold = 1
        log_value = 1.2

        # Check edge case scenarios where agent is standing still
        mask_values_X, mask_values_Y = Helper.masked_data(X, Y)
        
        # Create data and plot data if required
        X, Y, agent_order, spline_data = self.create_data_plot(X, Y, agent, mask_values_X, mask_values_Y, flip_dimensions, spline_interval, spline, plot_input,plot_spline)

        # Make copy of the original data for plots
        X_copy = X.copy()
        Y_copy = Y.copy()

        # Convert to tensor
        X, Y, spline_data = Helper.convert_to_tensor(X, Y, spline_data)

        # Check if future action is required
        if ADE_loss_adv_future or ADE_loss_adv_future_barrier or fake_collision_loss or hide_collision_loss:
            future_action = True
            new_input = torch.cat((X,Y),dim=2)
        else:
            future_action = False
            new_input = X

        # Calculate initial control actions
        dt = self.kwargs['data_param']['dt']
        control_action, heading_init, velocity_init = Control_action.get_control_actions(X, mask_values_X, flip_dimensions, new_input,dt)

        # Storage for adversarial position
        X_new_adv = torch.zeros_like(X)
        Y_new_adv = torch.zeros_like(Y)

        # Start the optimization of the adversarial attack
        for i in range(iter_num):
            # Reset gradients
            control_action.grad = None

            # Calculate updated adversarial position
            adv_position = Control_action.dynamical_model(new_input, control_action, velocity_init, heading_init,dt)

            # check if cotrol action are converted correctly
            if i == 0:
                Helper.check_conversion_control_action(adv_position, X, Y, future_action)

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
                
            # Forward pass through the model
            Pred_t = self.pert_model.predict_batch_tensor(X_new,T,Domain, num_steps)

            # Calculate the loss
            losses = Loss.calculate_loss(
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
                        fake_collision_loss_barrier,
                        hide_collision_loss,
                        hide_collision_loss_barrier,
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

        # Gaussian smoothing module
        X_pert_smoothed, Pred_pert_smoothed, X_unpert_smoothed, Pred_unpert_smoothed = Smoothing.randomized_smoothing(
                X,
                X_new_adv,
                smooth_perturbed_data,
                smooth_unperturbed_data,
                num_samples,
                sigmas,
                T,
                Domain, 
                num_steps,
                self.pert_model
                )

        # Detach the tensor and convert to numpy
        X_new_pert, Y_new_pert, Pred_t = Helper.detach_tensor(X_new_adv, Y_new_adv, Pred_t)

        # Plot the results
        Plot.plot_results(
                X_copy, 
                X_new_pert, 
                Y_copy, 
                Y_new_pert, 
                Pred_t,
                loss_store,
                plot_loss,  
                future_action,
                static_adv_scene,
                animated_adv_scene,
                car_length,
                car_width,
                mask_values_X, 
                mask_values_Y,
                plot_smoothing,
                X_pert_smoothed, 
                Pred_pert_smoothed, 
                X_unpert_smoothed, 
                Pred_unpert_smoothed,
                sigmas
            )

        # Return Y to old shape
        Y_new_pert = Helper.return_to_old_shape(Y_new_pert, Y_shape)

        if flip_dimensions:
            agent_order_inverse = np.argsort(agent_order)
            X_new_pert = X_new_pert[:, agent_order_inverse, :, :]
            Y_new_pert = Y_new_pert[:, agent_order_inverse, :, :]
        
        return X_new_pert, Y_new_pert
    
    def create_data_plot(self, X, Y, agent, mask_values_X, mask_values_Y, flip_dimensions, spline_interval, spline, plot_input, plot_spline):
        # Flip the dimensions of the data if required
        X, Y, agent_order = Helper.flip_dimensions(X, Y, agent, flip_dimensions)

        # Create spline data
        spline_data = Spline.spline_data(X, Y, mask_values_X, mask_values_Y, flip_dimensions, spline_interval, spline)

        # Plot the data if required
        Plot.Plot_data(X, Y, spline_data, plot_input, plot_spline)

        return X, Y, agent_order, spline_data

    def set_batch_size(self):
        '''
        This function sets the batch size for the perturbation method.

        It must add a attribute self.batch_size to the class.

        Returns
        -------
        None.

        '''

        self.batch_size = 2

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
    