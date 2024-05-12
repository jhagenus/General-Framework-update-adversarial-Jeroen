from perturbation_template import perturbation_template
import pandas as pd
import os
import numpy as np
import importlib
from Data_sets.data_interface import data_interface
import torch

from Adversarial_classes.control_action import Control_action
from Adversarial_classes.helper import Helper
from Adversarial_classes.loss import Loss
from Adversarial_classes.plot import Plot
from Adversarial_classes.smoothing import Smoothing
from Adversarial_classes.spline import Spline

from PIL import Image

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

        # Setting animated scene
        control_action_bar = False
        control_action_graph = True

        # check if only one control action type is selected
        Helper.assert_only_one_true_barrier(
                control_action_bar,
                control_action_graph
            )
        
        # Car size
        car_length = 4.1
        car_width = 1.7
        wheelbase = 2.7

        # Change ego and tar vehicle -> (Important to keep this on True to perturb the agent that turns left)
        flip_dimensions = True

        # Initialize parameters
        num_samples = 20 # Defined as (K) in paper
        iter_num = 20
        epsilon_acc = 6
        epsilon_curv = 0.2

        # Learning decay
        learning_rate_decay = True
        gamma = 1
        alpha = 0.01

        # Randomized smoothing 
        smooth_perturbed_data = True
        smooth_unperturbed_data = True
        num_samples_smoothing = 5 # Defined as (R) in paper
        num_samples_used_smoothing = 15 # Defined as (M) in paper -> need to be lower than (num_samples * num_samples_smoothing)
        sigmas = [0.05,0.1]
        plot_smoothing = True
        smoothing_method = 'control_action'   # 'position' or 'control_action'

        # remove nan from input and remember old shape
        Y_shape = Y.shape
        Y = Helper.remove_nan_values(Y)

        # Select loss function (set 1 on True and the rest on False)
        #ADE loss
        ADE_loss = False
        ADE_loss_adv_future_GT = False
        ADE_loss_adv_future_pred = True

        # Collision loss
        collision_loss = False
        fake_collision_loss_GT = False
        fake_collision_loss_Pred = False
        hide_collision_loss_GT = False
        hide_collision_loss_Pred = False

        Name_attack = Helper.retrieve_name_attack(
            ADE_loss,
            ADE_loss_adv_future_GT,
            ADE_loss_adv_future_pred,
            collision_loss,
            fake_collision_loss_GT,
            fake_collision_loss_Pred,
            hide_collision_loss_GT,
            hide_collision_loss_Pred
        )

        # check if only one loss function is activated
        Helper.assert_only_one_true(
                ADE_loss,
                ADE_loss_adv_future_GT,
                ADE_loss_adv_future_pred,
                collision_loss,
                fake_collision_loss_GT,
                fake_collision_loss_Pred,
                hide_collision_loss_GT,
                hide_collision_loss_Pred
            )

        # Barrier function (set 1 on true if barrier is activated and the rest on false, or all on false if no barrier is activated)
        log_barrier = False
        ADVDO_barrier = False
        spline_barrier = True

        # check if only one barrier function is activated
        Helper.assert_only_one_true_barrier(
                log_barrier,
                ADVDO_barrier,
                spline_barrier
            )
        
        # validation settings
        Helper.validate_plot_settings(plot_input, plot_spline)
        Helper.validate_plot_settings(spline, plot_spline)
        Helper.validate_plot_settings(spline, spline_barrier)

        # Barrier function parameters
        distance_threshold = 1
        log_value = 1.2
         
        # Check edge case scenarios where agent is standing still
        mask_values_X, mask_values_Y = Helper.masked_data(X, Y)
        
        # Create data and plot data if required
        X, Y, agent_order, spline_data = self.create_data_plot(X, Y, agent, mask_values_X, mask_values_Y, flip_dimensions, spline_interval, spline, plot_input,plot_spline)
        
        # Load images 
        # img, img_m_per_px = self.load_images(X,Domain)
        img, img_m_per_px = None, None

        # Show image
        # plot_img = Image.fromarray(img[0,0,:],'RGB')
        # plot_img.show()
        
        # Make copy of the original data for plots
        X_copy = X.copy()
        Y_copy = Y.copy()

        # Convert to tensor
        X, Y, spline_data = Helper.convert_to_tensor(X, Y, spline_data)

        # Check if future action is required
        new_input, future_action = Helper.create_new_input(X,Y, ADE_loss_adv_future_GT,ADE_loss_adv_future_pred, fake_collision_loss_GT, fake_collision_loss_Pred, hide_collision_loss_GT, hide_collision_loss_Pred)

        # Calculate initial control actions
        dt = self.kwargs['data_param']['dt']
        control_action, heading_init, velocity_init = Control_action.Reversed_Dynamical_Model(X, mask_values_X, flip_dimensions, new_input,dt)
        control_action.requires_grad = True 

        # Storage for adversarial position
        X_new_adv = torch.zeros_like(X)
        Y_new_adv = torch.zeros_like(Y)
        Pred_iter_1 = torch.zeros((Y.shape[0], num_samples, Y.shape[2], 2))

        # Clamp the control actions relative to ground truth (Not finished yet)
        tensor_addition = torch.zeros_like(control_action)
        tensor_addition[:,0] = epsilon_acc
        tensor_addition[:,1] = epsilon_curv

        control_actions_clamp_low = control_action - tensor_addition
        control_actions_clamp_high = control_action + tensor_addition
        
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
            X_new, Y_new, X_new_adv, Y_new_adv = Helper.return_data(adv_position, X, Y, future_action)

            # Output forward pass
            num_steps = Y.shape[2]
                
            # Forward pass through the model
            Pred_t = self.pert_model.predict_batch_tensor(X_new,T,Domain,img, img_m_per_px,num_steps,num_samples)

            # Store the first prediction
            if i == 0:
                Pred_iter_1 = Pred_t.detach()

            # Calculate the loss
            losses = Loss.calculate_loss(
                        X,
                        X_new,
                        Y,
                        Y_new,
                        Pred_t,
                        Pred_iter_1,
                        ADE_loss,
                        ADE_loss_adv_future_GT,
                        ADE_loss_adv_future_pred,
                        collision_loss,
                        fake_collision_loss_GT,
                        fake_collision_loss_Pred,
                        hide_collision_loss_GT,
                        hide_collision_loss_Pred,
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
                alpha = alpha * (gamma**i)

            # Update Control inputs
            with torch.no_grad():
                control_action[:,0,:,0].subtract_(grad[:,0,:,0], alpha=alpha)
                control_action[:,0,:,1].subtract_(grad[:,0,:,1], alpha=alpha)
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
                num_samples_smoothing,
                self.pert_model,
                mask_values_X,
                flip_dimensions,
                dt,
                epsilon_acc,
                epsilon_curv,
                smoothing_method,
                img,
                img_m_per_px,
                num_samples_used_smoothing
                )

        # Detach the tensor and convert to numpy
        X_new_pert, Y_new_pert, Pred_t, Pred_iter_1 = Helper.detach_tensor(X_new_adv, Y_new_adv, Pred_t, Pred_iter_1)

        # Plot the results
        Plot.plot_results(
                X_copy, 
                X_new_pert, 
                Y_copy, 
                Y_new_pert, 
                Pred_t,
                Pred_iter_1,
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
                sigmas,
                smoothing_method,
                dt,
                flip_dimensions,
                epsilon_acc,
                epsilon_curv,
                control_action_bar,
                control_action_graph,
                wheelbase,
                Name_attack
            )

        # Return Y to old shape
        Y_new_pert = Helper.return_to_old_shape(Y_new_pert, Y_shape)

        if flip_dimensions:
            agent_order_inverse = np.argsort(agent_order)
            X_new_pert = X_new_pert[:, agent_order_inverse, :, :]
            Y_new_pert = Y_new_pert[:, agent_order_inverse, :, :]
        
        return X_new_pert, Y_new_pert

    def load_images(self,X,Domain):
        Img_needed = np.zeros(X.shape[:2], bool)
        Img_needed[:,0] = True
        
        if self.data.includes_images():
            if self.pert_model.grayscale:
                channels = 1
            else:
                channels = 3
            img          = np.zeros((*Img_needed.shape, self.pert_model.target_height, self.pert_model.target_width, channels), np.uint8)
            img_m_per_px = np.ones(Img_needed.shape, np.float32) * np.nan

            centre = X[Img_needed, -1,:]
            x_rel = centre - X[Img_needed, -2,:]
            rot = np.angle(x_rel[:,0] + 1j * x_rel[:,1]) 
            domain_needed = Domain.iloc[np.where(Img_needed)[0]]
            
            img[Img_needed] = self.data.return_batch_images(domain_needed, centre, rot,
                                                            target_height = self.pert_model.target_height, 
                                                            target_width = self.pert_model.target_width,
                                                            grayscale = self.pert_model.grayscale,
                                                            Imgs_rot = img[Img_needed],
                                                            Imgs_index = np.arange(Img_needed.sum()))
            
            img_m_per_px[Img_needed] = self.data.Images.Target_MeterPerPx.loc[domain_needed.image_id]
        else:
            img = None
            img_m_per_px = None

        return img, img_m_per_px
    
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
    