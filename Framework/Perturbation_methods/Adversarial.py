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
        if 'repetition' in kwargs['splitter_dict'].keys():
            pert_splitter_rep = [kwargs['splitter_dict']['repetition']]
        else:
            pert_splitter_rep = [(0,)]
        if 'test_part' in kwargs['splitter_dict'].keys():
            pert_splitter_tp = kwargs['splitter_dict']['test_part']
        else:
            pert_splitter_tp = 0.2

        if 'train_pert' in kwargs['splitter_dict'].keys():
            pert_splitter_train_pert = kwargs['splitter_dict']['train_pert']
        else:
            pert_splitter_train_pert = False
        if 'test_pert' in kwargs['splitter_dict'].keys():
            pert_splitter_test_pert = kwargs['splitter_dict']['test_pert']
        else:
            pert_splitter_test_pert = False

        pert_splitter_module = importlib.import_module(pert_splitter_name)
        pert_splitter_class = getattr(pert_splitter_module, pert_splitter_name)

        # Initialize and apply Splitting method
        pert_splitter = pert_splitter_class(
            pert_data_set, pert_splitter_tp, pert_splitter_rep, pert_splitter_train_pert, pert_splitter_test_pert)
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
        self.plot_input = True

        # Spline settings
        self.total_spline_values = 100

        # Plot the loss over the iterations
        self.plot_loss = True

        # Plot the adversarial scene
        self.static_adv_scene = True
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
        # JULIAN: flip_dimensions() in helper.py suggests that this is must be the case.
        # JULIAN: So maybe not make this seems as something you can easily change, especially not make
        # JULIAN: This a parameter in the functions that you pass this to
        # JULIAN: Instead, if you want to use this varaible, pass self (i.e., Adversarial) to the function, and then
        # JULIAN: then use Adversarial.tar_agent_index and Adversarial.ego_agent_index in those functions
        # JULIAN: This has the added benefit ofg reducing the number of function parameters, which should always
        # JULIAN: be a goal when designing a function
        self.tar_agent_index = 0
        self.ego_agent_index = 1

        # Initialize parameters
        self.num_samples = 5  # Defined as (K) in our paper
        self.max_number_iterations = 5

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
        self.smoothing = True
        self.num_samples_used_smoothing = 15  # Defined as .. in paper
        self.sigma_acceleration = [0.05, 0.1]
        self.sigma_curvature = [0.01, 0.05]
        self.plot_smoothing = True

        # For ADE attack select: 'ADE', 'ADE_new_GT', 'ADE_new_pred'
        # For Collision attack select: 'Collision', 'Fake_collision_GT', 'Fake_collision_Pred', 'Hide_collision_GT', 'Hide_collision_Pred'
        self.loss_function = 'Fake_collision_GT'

        # For barrier function select: 'Log', 'Log_V2' or None
        self.barrier_function = 'Log_V2'

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
        X, Y = self.prepare_data(X, Y, T, agent, Domain)

        # Prepare data for adversarial attack (tensor/image prediction model)
        X, Y, positions_perturb, Y_Pred_iter_1, data_barrier = self.prepare_data_attack(
            X, Y)

        # Calculate initial control actions
        control_action, heading, velocity = Control_action.inverse_Dynamical_Model(
            positions_perturb=positions_perturb, dt=self.dt, device=self.pert_model.device)

        # Create a tensor for the perturbation
        perturbation = torch.zeros_like(control_action)
        perturbation.requires_grad = True

        # Store the loss for plot
        loss_store = []

        # Start the optimization of the adversarial attack
        for i in range(self.max_number_iterations):
            # Reset gradients
            perturbation.grad = None

            # Calculate updated adversarial position
            adv_position = Control_action.dynamical_model(
                control_action + perturbation, positions_perturb, heading, velocity, self.dt, device=self.pert_model.device)

            # Split the adversarial position back to X and Y
            X_new, Y_new = Helper.return_data(
                adv_position, X, Y, self.future_action_included)

            # Forward pass through the model
            Y_Pred = self.pert_model.predict_batch_tensor(X=X_new, T=T, Domain=Domain, img=self.img, img_m_per_px=self.img_m_per_px,
                                                          num_steps=self.num_steps_predict, num_samples=self.num_samples)

            if i == 0:
                # check conversion
                # Helper.check_conversion(adv_position, positions_perturb)

                # Store the first prediction
                Y_Pred_iter_1 = Y_Pred.detach()

            losses = self.loss_module(
                X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier)

            # Store the loss for plot
            loss_store.append(losses.detach().cpu().numpy())
            print(losses)

            # Calulate gradients
            losses.sum().backward()
            grad = perturbation.grad

            # Update Control inputs
            with torch.no_grad():
                perturbation.subtract_(grad, alpha=self.alpha)
                perturbation[:, :, :,
                             0].clamp_(-self.epsilon_acc_relative, self.epsilon_acc_relative)
                perturbation[:, :, :, 1].clamp_(
                    -self.epsilon_curv_relative, self.epsilon_curv_relative)

                control_action_perturbed = control_action + perturbation
                control_action_perturbed[:, :, :, 0].clamp_(
                    -self.epsilon_acc_absolute, self.epsilon_acc_absolute)
                control_action_perturbed[:, :, :, 1].clamp_(
                    -self.epsilon_curv_absolute, self.epsilon_curv_absolute)

                perturbation.copy_(control_action_perturbed - control_action)
                #perturbation + controlaction
                perturbation[:, 1:] = 0.0

            # Update the step size
            self.alpha *= self.gamma

        # Calculate the final adversarial position
        adv_position = Control_action.dynamical_model(
            control_action + perturbation, positions_perturb, heading, velocity, self.dt, device=self.pert_model.device)

        # Split the adversarial position back to X and Y
        X_new, Y_new = Helper.return_data(
            adv_position, X, Y, self.future_action_included)

        # Forward pass through the model
        Y_Pred = self.pert_model.predict_batch_tensor(X=X_new, T=T, Domain=Domain, img=self.img, img_m_per_px=self.img_m_per_px,
                                                      num_steps=self.num_steps_predict, num_samples=self.num_samples)

        # Gaussian smoothing module
        self.X_smoothed, self.X_smoothed_adv, self.Y_pred_smoothed, self.Y_pred_smoothed_adv = self.smoothing_module(
            X, Y, control_action, perturbation, adv_position, velocity, heading)

        # Detach the tensor and convert to numpy
        X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier = Helper.detach_tensor(
            X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier)

        # Plot the data
        self.ploting_module(X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1,
                            data_barrier, loss_store, control_action, perturbation)

        # Return Y to old shape
        Y_new = Helper.return_to_old_shape(Y_new, self.Y_shape)

        # Flip dimensions back
        X_new_pert, Y_new_pert = Helper.flip_dimensions_2(
            self.flip_dimensions, X_new, Y_new, self.agent_order)

        return X_new_pert, Y_new_pert

    def ploting_module(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier, loss_store, control_action, perturbation):
        # Initialize the plot class
        plot = Plot(self)

        # Plot the input/barrier data if required
        if self.plot_input:
            plot.plot_static_data(X=X, X_new=None, Y=Y, Y_new=None, Y_Pred=None,
                                  Y_Pred_iter_1=None, data_barrier=data_barrier, plot_input=self.plot_input)

        # Plot the loss over the iterations
        if self.plot_loss:
            plot.plot_loss_over_iterations(loss_store)

        # Plot the static adversarial scene
        if self.static_adv_scene:
            plot.plot_static_data(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred,
                                  Y_Pred_iter_1=Y_Pred_iter_1, data_barrier=data_barrier, plot_input=False)

        # Plot the animated adversarial scene
        if self.animated_adv_scene:
            plot.plot_animated_adv_scene(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1,
                                         control_action=control_action, perturbed_control_action=control_action+perturbation)

        # Plot the randomized smoothing
        if self.plot_smoothing:
            plot.plot_smoothing(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1,
                                X_smoothed=self.X_smoothed, X_smoothed_adv=self.X_smoothed_adv, Y_pred_smoothed=self.Y_pred_smoothed, Y_pred_smoothed_adv=self.Y_pred_smoothed_adv)

    def loss_module(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier):
        # calculate the loss
        losses = Loss.calculate_loss(self,
                                     X=X,
                                     X_new=X_new,
                                     Y=Y,
                                     Y_new=Y_new,
                                     Y_Pred=Y_Pred,
                                     Y_Pred_iter_1=Y_Pred_iter_1,
                                     barrier_data=data_barrier
                                     )

        return losses

    def smoothing_module(self, X, Y, control_action, perturbation, adv_position, velocity, heading):
        # initialize smoothing
        smoothing = Smoothing(self,
                              control_action=control_action,
                              control_action_perturbed=control_action+perturbation,
                              adv_position=adv_position,
                              velocity=velocity,
                              heading=heading,
                              X=X,
                              Y=Y
                              )

        # Randomized smoothing
        X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv = smoothing.randomized_smoothing()

        return X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv

    def assertion_check(self):
        # check if the size of both sigmas are the same
        Helper.check_size_list(self.sigma_acceleration, self.sigma_curvature)

        Helper.validate_settings_order(self.smoothing, self.plot_smoothing)

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

    def prepare_data(self, X, Y, T, agent, Domain):
        # Remove nan from input and remember old shape
        self.Y_shape = Y.shape
        Y = Helper.remove_nan_values(data=Y)

        # Flip dimensions agents
        X, Y, self.agent_order = Helper.flip_dimensions(
            X=X, Y=Y, agent=agent, flip_dimensions=self.flip_dimensions)

        self.T = T
        self.Domain = Domain

        return X, Y

    def prepare_data_attack(self, X, Y):
        # Convert to tensor
        X, Y = Helper.convert_to_tensor(self.pert_model.device, X, Y)

        # Check if future action is required
        positions_perturb, self.future_action_included = Helper.create_data_to_perturb(
            X=X, Y=Y, loss_function=self.loss_function)

        # data for barrier function
        data_barrier = torch.cat((X, Y), dim=2)

        self.mask_data = Helper.compute_mask_values_tensor(positions_perturb)

        # Load images for adversarial attack
        # img, img_m_per_px = self.load_images(X,Domain)
        self.img, self.img_m_per_px = None, None

        # Show image
        # plot_img = Image.fromarray(img[0,0,:],'RGB')
        # plot_img.show()

        # Create storage for the adversarial prediction on nominal setting
        Y_Pred_iter_1 = torch.zeros(
            (Y.shape[0], self.num_samples, Y.shape[2], Y.shape[3]))

        # number of steps to predict
        self.num_steps_predict = Y.shape[2]

        return X, Y, positions_perturb, Y_Pred_iter_1, data_barrier

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
