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
        assert 'data_set_dict' in kwargs.keys(
        ), "Adverserial model dataset is missing (required key: 'data_set_dict')."
        assert 'data_param' in kwargs.keys(
        ), "Adverserial model data param is missing (required key: 'data_param')."
        assert 'splitter_dict' in kwargs.keys(
        ), "Adverserial model splitter is missing (required key: 'splitter_dict')."
        assert 'model_dict' in kwargs.keys(
        ), "Adverserial model is missing (required key: 'model_dict')."
        assert 'exp_parameters' in kwargs.keys(
        ), "Adverserial model experiment parameters are missing (required key: 'exp_parameters')."

        assert kwargs['exp_parameters'][6] == 'predefined', "Perturbed datasets can only be used if the agents' roles are predefined."

        self.kwargs = kwargs
        self.initialize_settings()

        # Check that in splitter dict the length of repetition is only 1 (i.e., only one splitting method)
        if isinstance(kwargs['splitter_dict']['repetition'], list):

            if len(kwargs['splitter_dict']['repetition']) > 1:
                raise ValueError("The splitting dictionary neccessary to define the trained model used " +
                                 "for the adversarial attack can only contain one singel repetition " +
                                 "(i.e, the value assigned to the key 'repetition' CANNOT be a list with a lenght larger than one).")

            kwargs['splitter_dict']['repetition'] = kwargs['splitter_dict']['repetition'][0]

        # Load the perturbation model
        pert_data_set = data_interface(
            kwargs['data_set_dict'], kwargs['exp_parameters'])
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
        pert_splitter = pert_splitter_class(
            pert_data_set, pert_splitter_tp, pert_splitter_rep)
        pert_splitter.split_data()

        # Extract per model dict
        if isinstance(kwargs['model_dict'], str):
            pert_model_name = kwargs['model_dict']
            pert_model_kwargs = {}
        elif isinstance(kwargs['model_dict'], dict):
            assert 'model' in kwargs['model_dict'].keys(
            ), "No model name is provided."
            assert isinstance(kwargs['model_dict']['model'],
                              str), "A model is set as a string."
            pert_model_name = kwargs['model_dict']['model']
            if not 'kwargs' in kwargs['model_dict'].keys():
                pert_model_kwargs = {}
            else:
                assert isinstance(
                    kwargs['model_dict']['kwargs'], dict), "The kwargs value must be a dictionary."
                pert_model_kwargs = kwargs['model_dict']['kwargs']
        else:
            raise TypeError("The provided model must be string or dictionary")

        # Get model class
        pert_model_module = importlib.import_module(pert_model_name)
        pert_model_class = getattr(pert_model_module, pert_model_name)

        # Initialize the model
        self.pert_model = pert_model_class(
            pert_model_kwargs, pert_data_set, pert_splitter, True)

        # TODO: Check if self.pert_model can call the function that is needed later in perturb_batch (i.e., self.pert_model.adv_generation())

        # Train the model on the given training set
        self.pert_model.train()

        # Define the name of the perturbation method
        self.name = self.pert_model.model_file.split(os.sep)[-1][:-4]

    def initialize_settings(self):
        # Plot input data and spline (if plot is True -> plot_spline can be set on True)
        self.plot_input = False

        # Spline settings
        self.total_spline_values = 100

        # Plot the loss over the iterations
        self.plot_loss = False

        # Plot the adversarial scene
        self.static_adv_scene = False
        self.animated_adv_scene = True

        # Setting animated scene
        self.control_action_graph = True

        # Car size
        self.car_length = 4.1
        self.car_width = 1.7
        self.wheelbase = 2.7

        # Change ego and tar vehicle -> (Important to keep this on True to perturb the agent that turns left)
        self.flip_dimensions = True

        # Select which agent in datasets to attack
        self.tar_agent_index = 0
        self.ego_agent_index = 1

        # Initialize parameters
        self.num_samples = 20  # Defined as (K) in our paper
        self.max_number_iterations = 20

        # absolute clamping values
        self.epsilon_acc_absolute = 6
        self.epsilon_curv_absolute = 0.2

        # relative clamping values
        self.epsilon_acc_relative = 2
        self.epsilon_curv_relative = 0.05

        # Learning decay
        self.gamma = 1
        self.alpha = 0.001

        # Randomized smoothing
        self.smoothing = False
        self.num_samples_used_smoothing = 15 # Defined as .. in paper
        self.sigma_acceleration = [0.05, 0.1]
        self.sigma_curvature = [0.01, 0.05]
        self.plot_smoothing = True
        self.smoothing_method = 'control_action'   # 'position' or 'control_action'

        # For ADE attack select: 'ADE', 'ADE_new_GT', 'ADE_new_pred'
        # For Collision attack select: 'Collision', 'Fake_collision_GT', 'Fake_collision_Pred', 'Hide_collision_GT', 'Hide_collision_Pred'
        self.loss_function = 'ADE'

        # For barrier function select: 'Log', 'Spline' or None
        self.barrier_function = 'Spline'

        # Barrier function parameters
        self.distance_threshold = 1
        self.log_value = 1.2

        # Time step
        self.dt = self.kwargs['data_param']['dt']

        # Do a assertion check on settings
        self.assertion_check()

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
        # torch.autograd.set_detect_anomaly(True)

        # Prepare the data (ordering/spline/edge_cases)
        X, X_copy, X_shape, Y, Y_copy, Y_shape, agent_order, spline_data = self.prepare_data(X, Y, agent)

        # Prepare data for adversarial attack (tensor/image prediction model)
        X, Y, spline_data, positions_perturb, future_action_included, mask_data, img, img_m_per_px, Y_Pred_iter_1, num_steps_predict = self.prepare_data_attack(X, Y, spline_data)

        # Calculate initial control actions
        control_action, heading, velocity = Control_action.inverse_Dynamical_Model(positions_perturb=positions_perturb, dt=self.dt)

        # set to device
        control_action, heading, velocity = Helper.set_device(self.pert_model.device,control_action, heading, velocity)

        # Create a tensor for the perturbation
        perturbation = torch.zeros_like(control_action)
        perturbation.requires_grad = True

        # Create relative clamping limits
        control_actions_relative_low, control_actions_relative_high = self.relative_clamping(control_action)

        # Store the loss for plot
        loss_store = []

        # Start the optimization of the adversarial attack
        for i in range(self.max_number_iterations):
            # Reset gradients
            control_action.grad = None

            # Calculate updated adversarial position
            adv_position = Control_action.dynamical_model(control_action + perturbation, positions_perturb, heading, velocity, self.dt)

            # Split the adversarial position back to X and Y
            X_new, Y_new = Helper.return_data(adv_position, X, Y, future_action_included)

            # Forward pass through the model
            Y_Pred = self.pert_model.predict_batch_tensor(X=X_new, T=T, Domain=Domain, img=img, img_m_per_px=img_m_per_px, 
                                                          num_steps=num_steps_predict, num_samples=self.num_samples)

            if i == 0:
                # check conversion
                Helper.check_conversion(adv_position, positions_perturb)

                # Store the first prediction
                Y_Pred_iter_1 = Y_Pred.detach()

            losses = self.loss_module(X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, spline_data)
            
            # Store the loss for plot
            loss_store.append(losses.detach().cpu().numpy())
            print(losses)

            # Calulate gradients
            losses.sum().backward()
            grad = perturbation.grad

            # Update Control inputs
            with torch.no_grad():
                perturbation.subtract_(grad, alpha=self.alpha)
                perturbation[:,:,:,0].clamp_(-self.epsilon_acc_absolute, self.epsilon_acc_absolute)
                perturbation[:,:,:,1].clamp_(-self.epsilon_curv_absolute, self.epsilon_curv_absolute)
                perturbation.clamp_(control_actions_relative_low, control_actions_relative_high)
                perturbation[:, 1:] = 0.0

            # Update the step size
            self.alpha *= self.gamma


        # Gaussian smoothing module
        X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv = self.smoothing_module(X, X_new, T, Domain, img, img_m_per_px, num_steps_predict, 
                                                                                                 control_actions_relative_low, control_actions_relative_high)

        # Detach the tensor and convert to numpy
        X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, spline_data = Helper.detach_tensor(X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1,spline_data)

        # Plot the data

        self.ploting_module(X=X,X_new=X_new,Y=Y,Y_new=Y_new,Y_Pred=Y_Pred,Y_Pred_iter_1=Y_Pred_iter_1,spline_data=spline_data,loss_store=loss_store,future_action=future_action_included, control_actions_relative_low=control_actions_relative_low, control_actions_relative_high=control_actions_relative_high)

        # Plot the results
        # Plot.plot_results(
        #     X_copy,
        #     X_new_pert,
        #     Y_copy,
        #     Y_new_pert,
        #     Y_Pred,
        #     Y_Pred_iter_1,
        #     loss_store,
        #     self.plot_loss,
        #     future_action_included,
        #     self.static_adv_scene,
        #     self.animated_adv_scene,
        #     self.car_length,
        #     self.car_width,
        #     mask_values_X,
        #     mask_values_Y,
        #     self.plot_smoothing,
        #     X_pert_smoothed,
        #     Pred_pert_smoothed,
        #     X_unpert_smoothed,
        #     Pred_unpert_smoothed,
        #     self.sigmas,
        #     self.dt,
        #     self.flip_dimensions,
        #     self.epsilon_acc_absolute,
        #     self.epsilon_curv_absolute,
        #     self.control_action_bar,
        #     self.control_action_graph,
        #     self.wheelbase,
        #     self.loss_function
        # )

        # Return Y to old shape
        Y_new_pert = Helper.return_to_old_shape(Y_new_pert, Y_shape)

        # Flip dimensions back
        X_new_pert, Y_new_pert = Helper.flip_dimensions_2(self.flip_dimensions, X_new_pert, Y_new_pert, agent_order)

        return X_new_pert, Y_new_pert
    

    def ploting_module(self,X,X_new,Y,Y_new,Y_Pred,Y_Pred_iter_1,spline_data,loss_store,future_action,control_actions_relative_low, control_actions_relative_high):
        # Plot the input/spline data if required
        if self.plot_input:
            Plot.plot_static_data(X=X, X_new=None, Y=Y, Y_new=None, Y_Pred=None, Y_Pred_iter_1=None, spline_data=spline_data, future_action=False, plot_input=self.plot_input)

        # Plot the loss over the iterations
        if self.plot_loss:
            Plot.plot_loss_over_iterations(loss_store)

        # Plot the static adversarial scene
        if self.static_adv_scene:
            Plot.plot_static_data(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1, spline_data=spline_data, future_action=future_action, plot_input=False)

        # Plot the animated adversarial scene  
        if self.animated_adv_scene:
            Plot.plot_animated_adv_scene(X=X,X_new=X_new,Y=Y,Y_new=Y_new,Y_Pred=Y_Pred,Y_Pred_iter_1=Y_Pred_iter_1,dt=self.dt,epsilon_acc_absolute=self.epsilon_acc_absolute,epsilon_curv_absolute=self.epsilon_curv_absolute,car_length=self.car_length,car_width=self.car_width,wheelbase=self.wheelbase,Name_attack=self.loss_function,future_action=future_action,control_action_graph=self.control_action_graph,tar_agent=self.tar_agent_index,device=self.pert_model.device,control_actions_relative_low=control_actions_relative_low, control_actions_relative_high=control_actions_relative_high)


        # Plot the randomized smoothing
        if self.plot_smoothing:
            Plot.plot_smoothing(X,X_new_pert,Y,Y_new_pert,Pred_t,Pred_iter_1,future_action,sigmas,X_pert_smoothed,Pred_pert_smoothed,X_unpert_smoothed,Pred_unpert_smoothed,smoothing_method)


    
    def loss_module(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, spline_data):
        # calculate the loss
        losses = Loss.calculate_loss(X=X,
                                    X_new=X_new,
                                    Y=Y,
                                    Y_new=Y_new,
                                    Y_Pred=Y_Pred,
                                    Y_Pred_iter_1=Y_Pred_iter_1,
                                    distance_threshold=self.distance_threshold,
                                    log_value=self.log_value,
                                    spline_data=spline_data,
                                    loss_function=self.loss_function,
                                    barrier_function=self.barrier_function,
                                    tar_agent=self.tar_agent_index,
                                    ego_agent=self.ego_agent_index)
        
        return losses
    
    def smoothing_module(self,X, X_new, T, Domain, img, img_m_per_px, num_steps, control_actions_relative_low, control_actions_relative_high):
        #initialize smoothing
        smoothing = Smoothing(dt=self.dt,
                              tar_agent=self.tar_agent_index,
                              num_samples_smoothing=self.num_samples_used_smoothing,
                              sigma_acceleration=self.sigma_acceleration,
                              sigma_curvature=self.sigma_curvature,
                              epsilon_acc_absolute=self.epsilon_acc_absolute,
                              epsilon_curv_absolute=self.epsilon_curv_absolute,
                              control_actions_relative_low=control_actions_relative_low,
                              control_actions_relative_high=control_actions_relative_high,
                              pert_model=self.pert_model,
                              Domain=Domain,
                              T=T,
                              img=img,
                              img_m_per_px=img_m_per_px,
                              num_steps=num_steps)
        
        # Randomized smoothing
        X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv = smoothing.randomized_smoothing(X, X_new, self.smoothing)

        return X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv

    def relative_clamping(self,control_action):
        # Clamp the control actions relative to ground truth (Not finished yet)
        tensor_addition = torch.zeros_like(control_action)
        tensor_addition[:, 0] = self.epsilon_acc_relative
        tensor_addition[:, 1] = self.epsilon_curv_relative

        # JULIAN: Those function need to be done for the relative clamping
        control_actions_clamp_low = control_action - tensor_addition
        control_actions_clamp_high = control_action + tensor_addition

        return control_actions_clamp_low, control_actions_clamp_high

    def assertion_check(self):
        # check if the size of both sigmas are the same
        Helper.check_size_list(self.sigma_acceleration, self.sigma_curvature)


    def load_images(self, X, Domain):
        Img_needed = np.zeros(X.shape[:2], bool)
        Img_needed[:, 0] = True

        if self.data.includes_images():
            if self.pert_model.grayscale:
                channels = 1
            else:
                channels = 3
            img = np.zeros((*Img_needed.shape, self.pert_model.target_height,
                            self.pert_model.target_width, channels), np.uint8)
            img_m_per_px = np.ones(Img_needed.shape, np.float32) * np.nan

            centre = X[Img_needed, -1, :]
            x_rel = centre - X[Img_needed, -2, :]
            rot = np.angle(x_rel[:, 0] + 1j * x_rel[:, 1])
            domain_needed = Domain.iloc[np.where(Img_needed)[0]]

            img[Img_needed] = self.data.return_batch_images(domain_needed, centre, rot,
                                                            target_height=self.pert_model.target_height,
                                                            target_width=self.pert_model.target_width,
                                                            grayscale=self.pert_model.grayscale,
                                                            Imgs_rot=img[Img_needed],
                                                            Imgs_index=np.arange(Img_needed.sum()))

            img_m_per_px[Img_needed] = self.data.Images.Target_MeterPerPx.loc[domain_needed.image_id]
        else:
            img = None
            img_m_per_px = None

        return img, img_m_per_px

    def prepare_data(self, X, Y, agent):
        # Remove nan from input and remember old shape
        Y = Helper.remove_nan_values(data=Y)
        X_shape = X.shape
        Y_shape = Y.shape

        # Flip dimensions agents
        X, Y, agent_order = Helper.flip_dimensions(X=X, Y=Y, agent=agent, flip_dimensions=self.flip_dimensions)

        # Make copy of the original flipped data for plots
        X_copy = X.copy()
        Y_copy = Y.copy()

        # Create spline data
        spline_data = Spline.spline_data(X=X, 
                                         Y=Y,  
                                         total_spline_values=self.total_spline_values)

        return X, X_copy, X_shape, Y, Y_copy, Y_shape, agent_order, spline_data

    def prepare_data_attack(self, X, Y, spline_data):
        # Convert to tensor
        X, Y, spline_data = Helper.convert_to_tensor(self.pert_model.device, X, Y, spline_data)
        
        # Check if future action is required
        positions_perturb, future_action_included = Helper.create_data_to_perturb(X=X, Y=Y, loss_function=self.loss_function)

        mask_data = Helper.compute_mask_values_tensor(positions_perturb)

        # Load images for adversarial attack
        # img, img_m_per_px = self.load_images(X,Domain)
        img, img_m_per_px = None, None

        # Show image
        # plot_img = Image.fromarray(img[0,0,:],'RGB')
        # plot_img.show()

        # Create storage for the adversarial prediction on nominal setting
        Y_Pred_iter_1 = torch.zeros((Y.shape[0], self.num_samples, Y.shape[2], Y.shape[3]))
        
        # number of steps to predict
        num_steps_predict = Y.shape[2]

        return X, Y, spline_data, positions_perturb, future_action_included, mask_data, img, img_m_per_px, Y_Pred_iter_1, num_steps_predict

    def set_batch_size(self):
        '''
        This function sets the batch size for the perturbation method.

        It must add a attribute self.batch_size to the class.

        Returns
        -------
        None.

        '''

        self.batch_size = 1

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
