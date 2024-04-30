import torch
import numpy as np

from Adversarial_classes.control_action import Control_action

class Smoothing:
    @staticmethod
    def randomized_smoothing(X, X_new_adv, smooth_perturbed_data, smooth_unperturbed_data, num_samples, sigmas,T,Domain, num_steps,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method):
        if not smooth_perturbed_data and not smooth_unperturbed_data:
            return None, None, None, None
        
        # Storage for the outputs
        outputs_per_sigma_pert = [[] for _ in sigmas]
        outputs_per_sigma_unpert = [[] for _ in sigmas]

        # Storage for the inputs with gaussian noise
        perturbated_X_per_sigma_pert = [[] for _ in sigmas]
        perturbated_X_per_sigma_unpert = [[] for _ in sigmas]

        # Apply randomized adversarial smoothing
        for i, sigma in enumerate(sigmas):
            for _ in range(num_samples):
                # Smooth perturbed data
                if smooth_perturbed_data and not smooth_unperturbed_data:
                    # Add gaussian noise to the perturbed data
                    smoothed_input_data_pert, Pred_pert_smoothed = Smoothing.forward_pass_smoothing(X_new_adv,sigma,T, Domain, num_steps,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method)

                    # append the smoothed perturbed data
                    perturbated_X_per_sigma_pert[i].append(smoothed_input_data_pert)
                    outputs_per_sigma_pert[i].append(Pred_pert_smoothed)

                # Smooth unperturbed data
                elif smooth_unperturbed_data and not smooth_perturbed_data:
                    # Add gaussian noise to the unperturbed data
                    smoothed_input_data_unpert, Pred_unpert_smoothed = Smoothing.forward_pass_smoothing(X,sigma,T, Domain, num_steps,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method)

                    # append the smoothed unperturbed data
                    perturbated_X_per_sigma_unpert[i].append(smoothed_input_data_unpert)
                    outputs_per_sigma_unpert[i].append(Pred_unpert_smoothed)

                # Smooth both perturbed and unperturbed data
                else:
                    # Add gaussian noise to the perturbed data
                    smoothed_input_data_pert, Pred_pert_smoothed = Smoothing.forward_pass_smoothing(X_new_adv,sigma,T, Domain, num_steps,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method)

                    # append the smoothed perturbed data
                    perturbated_X_per_sigma_pert[i].append(smoothed_input_data_pert)
                    outputs_per_sigma_pert[i].append(Pred_pert_smoothed)
    
                    # Add gaussian noise to the unperturbed data
                    smoothed_input_data_unpert, Pred_unpert_smoothed = Smoothing.forward_pass_smoothing(X,sigma,T, Domain, num_steps,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method)

                    # append the smoothed unperturbed data
                    perturbated_X_per_sigma_unpert[i].append(smoothed_input_data_unpert)
                    outputs_per_sigma_unpert[i].append(Pred_unpert_smoothed)
        
        # Return to array
        outputs_per_sigma_pert = np.array(outputs_per_sigma_pert)
        outputs_per_sigma_unpert = np.array(outputs_per_sigma_unpert)
        perturbated_X_per_sigma_pert = np.array(perturbated_X_per_sigma_pert)
        perturbated_X_per_sigma_unpert = np.array(perturbated_X_per_sigma_unpert)

        if smooth_perturbed_data and not smooth_unperturbed_data:
            return perturbated_X_per_sigma_pert, outputs_per_sigma_pert, None, None
        elif smooth_unperturbed_data and not smooth_perturbed_data:
            return None, None, perturbated_X_per_sigma_unpert, outputs_per_sigma_unpert
        else:
            return perturbated_X_per_sigma_pert, outputs_per_sigma_pert, perturbated_X_per_sigma_unpert, outputs_per_sigma_unpert

    @staticmethod
    def forward_pass_smoothing(input_data,sigma,T, Domain, num_steps,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method):
        # Add noise to the target agent
        if smoothing_method == 'position':
            input_data_noised = Smoothing.add_noise_positions(input_data,sigma)
        elif smoothing_method == 'control_action':
            input_data_noised = Smoothing.add_noise_control_actions(input_data, mask_values_X, flip_dimensions, dt, sigma,epsilon_acc,epsilon_curv)
        else:
            raise ValueError("Please give either 'position' or 'control_action' as input to the randomized smoothing.")

        # Make the prediction and calculate the expectation
        Pred_smoothed = pert_model.predict_batch_tensor(input_data_noised, T, Domain, num_steps)
        Pred_smoothed = torch.mean(Pred_smoothed, dim=1)

        # Detach tensors
        input_data_noised = input_data_noised.detach().cpu().numpy()
        Pred_smoothed = Pred_smoothed.detach().cpu().numpy()

        return input_data_noised, Pred_smoothed
    
    @staticmethod
    def add_noise_positions(input_data,sigma):
        noise_data = torch.randn_like(input_data) * sigma
        noise_data[:,1:] = 0.0
        input_data_noised = input_data + noise_data

        return input_data_noised

    @staticmethod
    def add_noise_control_actions(input_data, mask_values_X, flip_dimensions, dt, sigma,epsilon_acc,epsilon_curv):
        control_action, heading_init, velocity_init = Control_action.Reversed_Dynamical_Model(input_data, mask_values_X, flip_dimensions, input_data, dt)

        noise_data = torch.randn_like(control_action) * sigma
        noise_data[:,1:] = 0.0
        control_action = control_action + noise_data

        with torch.no_grad():
            control_action[:,0,:,0].clamp_(-epsilon_acc, epsilon_acc)
            control_action[:,0,:,1].clamp_(-epsilon_curv, epsilon_curv)

        input_data_noised = Control_action.dynamical_model(input_data, control_action, velocity_init, heading_init, dt)

        return input_data_noised
