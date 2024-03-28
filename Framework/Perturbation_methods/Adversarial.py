from perturbation_template import perturbation_template
import pandas as pd
import os
import numpy as np
import importlib
from Data_sets.data_interface import data_interface
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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

        torch.autograd.set_detect_anomaly(True)

        # Remove nan values
        Y_shape = Y.shape
        max_length = np.min(np.sum(~np.isnan(Y[:,:,:,0]), axis=2)[:,0])
        Y = Y[:,:,:max_length,:]

        # Change ego and tar vehicle
        flip_dimensions = False

        i_agent_perturbed = np.where(agent == 'tar')[0][0]
        i_agent_collision = np.where(agent == 'ego')[0][0]
        other_agents = np.arange(Y.shape[1])
        other_agents = np.delete(other_agents, [i_agent_perturbed, i_agent_collision])
        agent_order = np.array([i_agent_perturbed, i_agent_collision, *other_agents])

        if flip_dimensions:
            X = X[:, agent_order, :, :]
            Y = Y[:, agent_order, :, :]

        # Spline settings
        spline_X_Y = True
        spline_interval = 100
        spline_data = np.zeros((X.shape[0],spline_interval,2))

        if spline_X_Y:
            Spline_input_values = np.concatenate((X,Y),axis=2)
        else:
            Spline_input_values = X.copy()

        if flip_dimensions:
            for batch_idx in range(X.shape[0]):
                i = 1
                while i < Spline_input_values.shape[2]:
                    if Spline_input_values[batch_idx, 0, i, 0] > Spline_input_values[batch_idx, 0, i - 1, 0]:
                        Spline_input_values[batch_idx, 0, i-1:,:] = np.nan
                        break
                    else:
                        i += 1

        # Spline historical data
        for i in range(X.shape[0]):
            sample_spline = Spline_input_values[i,0,:,:]
            sample_spline = sample_spline[~np.isnan(sample_spline).any(axis=1)]
            if sample_spline[0,0] > sample_spline[-1,0]:
                sample_spline[:,0] = np.flip(sample_spline[:,0])
            if sample_spline[0,1] > sample_spline[-1,1]:
                sample_spline[:,1] = np.flip(sample_spline[:,1])
            Spline_function = CubicSpline(sample_spline[:,0], sample_spline[:,1])
            xs = np.linspace(sample_spline[0,0], sample_spline[-1,0], spline_interval)
            spline_data[i,:,0] = xs
            spline_data[i,:,1] = Spline_function(xs)

        # Convert to tensor
        X_copy = X.copy()
        X = torch.from_numpy(X).to(dtype = torch.float32).to(device='cuda')
        Y = torch.from_numpy(Y).to(dtype = torch.float32).to(device='cuda')
        spline_data = torch.from_numpy(spline_data).to(dtype = torch.float32).to(device='cuda')

        Y_eval = Y.clone()

        # Investigate specific sample of batch
        specific_sample = False

        if specific_sample:
            sample_num = 0
            sample_num_help = sample_num + 1
            X = X[sample_num:sample_num_help,...]
            Y = Y[sample_num:sample_num_help,...]
            T = T[sample_num:sample_num_help,...] 

        # Initialize parameters
        iter_num = 1
        epsilon_acc = 10
        epsilon_curv = 0.2
        distance_threshold_past = 1
        Distance_threshold_future = 2

        # Learning rate
        learning_rate_decay = True
        gamma = 0.98

        # straight
        alpha_acc = 0.0007
        alpha_curv = 0.00001

        # Corner
        # alpha_acc = 10
        # alpha_curv = 0.0000001

        # Select loss function
        #ADE loss
        ADE_loss = True
        ADE_loss_barrier = False
        ADE_exp_pred_loss = False
        ADE_exp_pred_loss_barrier = False

        # Collision loss
        collision_loss = False
        collision_loss_barrier = False
        fake_collision_loss = False
        hide_collision_loss = False

        # Barrier function
        log_barrier = True
        ADVDO_barrier = False
        spline_barrier_past = False

        # Barrier function parameters
        log_value = 1.4
        spline_value_past = 1.05
        log_value_future = 1.1

        # Create new input for the optimization
        if ADE_exp_pred_loss or ADE_exp_pred_loss_barrier or fake_collision_loss or hide_collision_loss:
            new_input = torch.cat((X,Y),dim=2)
        else:
            new_input = X

        # Initialize control action
        control_action = torch.zeros_like(new_input)

        # Select if to plot figure and loss
        plot_figure = True
        loss_store = []
        X_new_print = torch.zeros_like(X)
        Y_new_print = torch.zeros_like(Y)

        # Get the initial velocity and heading
        dt = self.kwargs['data_param']['dt']
        velocity_init = torch.linalg.norm(X[:,0,1,:] - X[:,0,0,:] , dim=-1, ord = 2) / dt

        dx = X[:, 0, 1, 0] - X[:, 0, 0, 0]  
        dy = X[:, 0, 1, 1] - X[:, 0, 0, 1]  
        heading_init = torch.atan2(dy, dx) 

        # Initialize control actions
        velocity = torch.zeros(X.shape[0],X.shape[2]).to(device='cuda')
        angle = torch.zeros(X.shape[0],X.shape[2]).to(device='cuda')

        for i in range(X.shape[2]-1):
            velocity[:,0] = velocity_init 
            velocity[:,i+1] = torch.linalg.norm(X[:,0,i+1,:] - X[:,0,i,:] , dim=-1, ord = 2) / dt 
            control_action[:,0,i,0] = (velocity[:,i+1] - velocity[:,i]) / dt
            if i == 0:
                control_action[:,0,i,1] = 0
                angle[:,i] = heading_init
            else:
                dx = X[:, 0, i+1, 0] - X[:, 0, i, 0]
                dy = X[:, 0, i+1, 1] - X[:, 0, i, 1]
                angle[:,i] = torch.atan2(dy, dx).to(device='cuda')
                d_yaw_rate = (angle[:,i] - angle[:,i-1]) / dt
                curvature = d_yaw_rate / velocity[:,i]
                # mask_nan = torch.isnan(curvature)
                # mask_large_curvature = torch.abs(curvature) > epsilon_curv
                control_action[:,0,i-1,1] = curvature
                # control_action[:,0,i-1,1][mask_nan | mask_large_curvature] = 0
                

        control_action[torch.isinf(control_action)] = 1e-6
        control_action.requires_grad = True 

        print(control_action)


        # Start the optimization of the adversarial attack
        for i in range(iter_num):
            # Reset gradients
            control_action.grad = None

            # Adversarial position storage
            adv_position = new_input.clone().detach()

            # Update adversarial position based on dynamical model
            acc = control_action[:,0,:-1,0]
            cur = control_action[:,0,:-1,1]
            
            acc_accumulated = torch.cumsum(acc, dim=1) * dt
            acc_accumulated = torch.cat((torch.zeros_like(acc_accumulated[:,0:1]), acc_accumulated[:,:-1]), dim=1)

            Velocity = torch.cumsum(acc, dim=1) * dt + velocity_init.unsqueeze(1)
            D_yaw_rate = Velocity * cur
            D_yaw_rate_accumulated = torch.cumsum(D_yaw_rate, dim=1) * dt
            D_yaw_rate_accumulated = torch.cat((torch.zeros_like(D_yaw_rate_accumulated[:,0:1]), D_yaw_rate_accumulated[:,:-1]), dim=1)

            Heading = D_yaw_rate_accumulated + heading_init.unsqueeze(1)

            adv_position[:, 0, 1:, 0] = torch.cumsum(Velocity * torch.cos(Heading), dim=1) * dt + adv_position[:, 0, 0, 0].unsqueeze(1)
            adv_position[:, 0, 1:, 1] = torch.cumsum(Velocity * torch.sin(Heading), dim=1) * dt + adv_position[:, 0, 0, 1].unsqueeze(1)

            # Split the adversarial position back to X and Y
            if ADE_exp_pred_loss or ADE_exp_pred_loss_barrier or fake_collision_loss or hide_collision_loss:
                X_new, Y_new = torch.split(adv_position, [X.shape[2], Y.shape[2]], dim=2)
                X_new_print = X_new
                Y_new_print = Y_new
            else: 
                X_new_print = adv_position
                Y_new_print = Y
                X_new = adv_position
                Y_new = Y

            # Output forward pass
            num_steps = Y.shape[2]
            print(X_new)
            Pred_t = self.pert_model.predict_batch_tensor(X_new,T,Domain, num_steps)

            # Calculate ADE loss
            ADE_adv_future = torch.mean(torch.mean(torch.linalg.norm(Y_eval[:,0,:,:].unsqueeze(1) - Pred_t[:,:,:,:], dim=-1 , ord = 2), dim=-1),dim=-1)
            ADE_feasibility = torch.mean(torch.linalg.norm(Y_new[:,0,:,:] - Y_eval[:,0,:,:], dim=-1 , ord = 2), dim=-1)
            ADE_exp_pred = torch.mean(torch.mean(torch.linalg.norm(Y_new[:,0,:,:].unsqueeze(1) - Pred_t[:,:,:,:], dim=-1 , ord = 2), dim=-1),dim=-1)

            # Calculate collision loss
            collision_adv_prediction_future = torch.mean(torch.linalg.norm(Y_eval[:,1,:,:].unsqueeze(1) - Pred_t[:,:,:,:], dim=-1 , ord = 2).min(dim=-1).values,dim=-1)
            collision_adv_perturb_future = torch.mean(torch.linalg.norm(Y_eval[:,1,:,:] - Y_new[:,0,:,:], dim=-1 , ord = 2).min(dim=-1).values,dim=-1)

            # Calculate no collision loss
            # no_collision_adv_prediction_future = torch.log(torch.mean(torch.linalg.norm(Y_eval[:,1,:,:].unsqueeze(1) - Pred_t[:,:,:,:], dim=-1 , ord = 2).min(dim=-1).values,dim=-1)-collision_threshold_future)
            # no_collision_adv_perturb_future = torch.log(torch.linalg.norm(Y_eval[:,1,:,:] - Y_new[:,0,:,:], dim=-1 , ord = 2).min(dim=-1).values - collision_threshold_future)

            # Add  regularization loss to adversarial input using barrier function
            if log_barrier:
                barrier_norm = torch.norm(X_new[:,0,:,:] - X[:,0,:,:], dim = -1)
                barrier_log = torch.log(distance_threshold_past - barrier_norm)
                barrier_log_new = barrier_log / torch.log(torch.tensor(log_value))
                barrier_output = torch.mean(barrier_log_new,dim=-1)
            elif ADVDO_barrier:
                barrier_norm = torch.norm(X_new[:,0,:,:] - X[:,0,:,:], dim = -1)
                barrier_output = -((barrier_norm / distance_threshold_past) - torch.sigmoid(barrier_norm / distance_threshold_past) + 0.5).sum(dim=-1)
            elif spline_barrier_past:
                distance = torch.cdist(X_new[:,0,:,:], spline_data, p=2)
                min_indices = torch.argmin(distance, dim=-1)
                closest_points = torch.gather(spline_data.unsqueeze(1).expand(-1, X_new.shape[2], -1, -1), 2, min_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, spline_data.size(-1)))
                closest_points = closest_points.squeeze(2)
                barrier_norm = torch.norm(X_new[:,0,:,:] - closest_points, dim = -1)
                barrier_output = torch.log(distance_threshold_past - barrier_norm)
                barrier_log_new = barrier_output / torch.log(torch.tensor(spline_value_past))
                barrier_output = torch.mean(barrier_log_new,dim=-1)

            
            # if fake_collision_loss or hide_collision_loss:
            #     if fake_collision_loss:
            #         distance_future = torch.cdist(Y_new[:,0,:,:], spline_data, p=2)
            #     else:
            #         distance_future = torch.cdist(np.mean(Pred_t, axis=1), spline_data, p=2)

            #     min_indices_future = torch.argmin(distance_future, dim=-1)

            #     closest_points_future = torch.gather(spline_data.unsqueeze(1).expand(-1, Y_new.shape[2], -1, -1), 2, min_indices_future.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, spline_data.size(-1)))
            #     closest_points_future = closest_points_future.squeeze(2)

            #     barrier_norm_future = torch.norm(Y_new[:,0,:,:] - closest_points_future, dim = -1)
            #     barrier_output_future = torch.log(Distance_threshold_future - barrier_norm_future)
            #     barrier_log_new_future = barrier_output_future / torch.log(torch.tensor(log_value_future))
            #     barrier_output_future = torch.mean(barrier_log_new_future,dim=-1)

            # Calculate the total loss
            if ADE_loss:
                losses = ADE_adv_future
            elif ADE_exp_pred_loss:
                losses = ADE_exp_pred
            elif ADE_exp_pred_loss_barrier:
                losses = ADE_exp_pred + barrier_output
            elif ADE_loss_barrier:
                losses = ADE_adv_future + barrier_output
            elif collision_loss:
                losses = -collision_adv_prediction_future
            elif collision_loss_barrier:
                losses = -collision_adv_prediction_future + barrier_output
            elif fake_collision_loss:
                losses = -collision_adv_prediction_future + barrier_output + barrier_output_future
            elif hide_collision_loss:
                losses = -collision_adv_perturb_future + barrier_output + barrier_output_future

            # Store the loss for plot
            loss_store.append(losses.detach().cpu().numpy())

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

        X_new = X_new_print.detach().cpu().numpy()
        Y_new = Y.detach().cpu().numpy()
        Pred_t = Pred_t.detach().cpu().numpy()
        Y_new_print = Y_new_print.detach().cpu().numpy()

        Pred_t = np.mean(Pred_t, axis=1)

        if plot_figure:
            plt.figure(0)
            plt.plot(loss_store, marker='o', linestyle='-')
            plt.title('Loss for samples')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.show()

            for i in range(X.shape[0]):
                plt.figure()
                plt.plot(X_copy[i,0,:,0], X_copy[i,0,:,1], marker='o', linestyle='-', color='b', label='Past original adverserial')
                plt.plot(X_copy[i,1,:,0], X_copy[i,1,:,1], marker='o', linestyle='-', color='g',label='Past original other')
                plt.plot(X_new[i,0,:,0], X_new[i,0,:,1], marker='o', linestyle='-', color='r', label='Past perturbed adverserial')
                plt.plot(Pred_t[i,:,0], Pred_t[i,:,1], marker='o', linestyle='-', color='c', label='Future adversarial predict')
                plt.plot(Y_new[i,0,:,0], Y_new[i,0,:,1], marker='o', linestyle='-', color='m', label='Future original adversarial')
                plt.plot(Y_new[i,1,:,0], Y_new[i,1,:,1], marker='o', linestyle='-', color='y', label='Future original other')
                if ADE_exp_pred_loss or ADE_exp_pred_loss_barrier or fake_collision_loss or hide_collision_loss:
                    plt.plot(Y_new_print[i,0,:,0], Y_new_print[i,0,:,1], marker='o', linestyle='-', color='k', label='Future adversarial expected')
                plt.axis('equal')
                plt.legend()
                plt.show()

        # Return to old shape
        Y_pred = np.expand_dims(Pred_t, axis=1)
        
        nan_array = np.full((Y_shape[0], Y_shape[1], Y_shape[2]-Y_new.shape[2], Y_shape[3]), np.nan)
        Y_output = np.concatenate((Y_new, nan_array), axis=2)

        if flip_dimensions:
            agent_order_inverse = np.argsort(agent_order)
            X_new_pert = X_new[:, agent_order_inverse, :, :]
            Y_new_pert = Y_output[:, agent_order_inverse, :, :]

        return X_new_pert, Y_new_pert
    
    # def adversarial_smoothing(self, X, Y_new, Y, T, Domain):
    
    def adversarial_smoothing(self, X_pert, X, Y_pert_prediction, Y, T, Domain):
        X_pert_copy = X_pert.copy()

        flip_dimensions = True

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
        num_noised_samples = 40
        sigmas = [0.05, 0.10, 0.15, 0.20]
        analyze_pert = True
        num_steps = Y_pert_prediction.shape[2]

        # Plot settings
        plot_figure = True
        limit_X = False
        limit_Y = False

        # Create list to store values for all sigmas
        outputs_per_sigma = [[] for _ in sigmas]

        # Apply randomized adversarial smoothing
        for i, sigma in enumerate(sigmas):
            for _ in range(num_noised_samples):
                if analyze_pert:
                    noise = torch.randn_like(X_pert) * sigma
                    input_data = X_pert + noise
                else:
                    noise = torch.randn_like(X) * sigma
                    input_data = X + noise

                Pred_t, _ = self.pert_model.predict_batch_tensor(input_data, Y, T, Domain, num_steps)
                Pred_t = torch.mean(Pred_t, dim=1)

                outputs_per_sigma[i].append(Pred_t.detach().cpu().numpy())

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
                            axes[j].plot(sigma_outputs[k,i, :, 0], sigma_outputs[k,i, :, 1], marker='o', linestyle='-', color='b',ms=1, label='Prediction with smoothing')

                    if limit_X or limit_Y:
                        filtered_outputs = np.array(filtered_outputs)
                        mean_output = np.mean(filtered_outputs, axis=0)
                        axes[j].plot(mean_output[:, 0], mean_output[:, 1], marker='o', linestyle='-', color='k', ms=1, label='Mean Prediction')

                    axes[j].plot(outputs[i, :, 0], outputs[i, :, 1], marker='o', linestyle='-', color='k',ms=1)
                    axes[j].plot(X_pert_copy[i, 0, :, 0], X_pert_copy[i, 0, :, 1], marker='o', linestyle='-', color='g',ms=1)
                    axes[j].plot(Y[i, 0, :, 0], Y[i, 0, :, 1], marker='o', linestyle='-', color='r',ms=1)

                    if analyze_pert:
                        axes[j].plot(Y_pert_prediction[i, 0, :, 0], Y_pert_prediction[i, 0, :, 1], marker='o', linestyle='-', color='y',ms=1)


                    axes[j].axis('equal')
                    axes[j].set_title(f'ADE for sigma = {sigmas[j]}')
                    axes[j].set_xlabel('Sample')
                    axes[j].grid(True)
                
                plt.tight_layout()
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

        self.batch_size = 4

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