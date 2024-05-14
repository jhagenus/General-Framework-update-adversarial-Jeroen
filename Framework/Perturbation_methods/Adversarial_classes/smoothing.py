import torch
import numpy as np
import matplotlib.pyplot as plt

from Adversarial_classes.control_action import Control_action

class Smoothing:
    # FREDERIK: This might seem a little pedantic, but you really need to improve the consistency in your code.
    # https://peps.python.org/pep-0008/ <- The one you do all the time is lack of space after commas in function arguments.
    # It might seem inconsequential, but it makes the code harder to read, and if you are working in a team, it is important to have a consistent style.
    # A code review in a professional setting would have caught this, and they wouldn't even have looked at the actual code before you fixed this.
    # Another thing is that your lines tend to be very long and not fit onto a standard screen. This is also a readability issue. Break the lines
    # at meaningful places, like after commas or operators. This is also a common code review comment.

    @staticmethod
    def randomized_smoothing(X, X_new_adv, smooth_perturbed_data, smooth_unperturbed_data, num_samples, sigmas,T,Domain, num_steps,num_samples_smoothing,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method,img,img_m_per_px,num_samples_used_smoothing):
        if not smooth_perturbed_data and not smooth_unperturbed_data:
            return None, None, None, None
        
        # Storage for the outputs
        outputs_per_sigma_pert = [[] for _ in sigmas]
        outputs_per_sigma_unpert = [[] for _ in sigmas]

        # Storage for the inputs with gaussian noise
        perturbated_X_per_sigma_pert = [[] for _ in sigmas]
        perturbated_X_per_sigma_unpert = [[] for _ in sigmas]

        # Apply randomized adversarial smoothing
        # FREDERIK: Be careful with applying the same amount of smoothing for both acceleration and curvature.
        # Semantically, it is not very meaningful. IMO, it is better to design the smoothing to specify different
        # sigmas for acceleration and curvature, and then, if needed, apply the same sigma for both.
        for i, sigma in enumerate(sigmas):
            for _ in range(num_samples_smoothing):
                # FREDERIK: One if-else for perturbed data and one for unperturbed data.
                # It will be way more readable.

                # Smooth perturbed data
                if smooth_perturbed_data and not smooth_unperturbed_data:
                    # Add gaussian noise to the perturbed data
                    smoothed_input_data_pert, Pred_pert_smoothed = Smoothing.forward_pass_smoothing(X_new_adv,sigma,T, Domain, num_samples,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method,img,img_m_per_px,num_steps)

                    # append the smoothed perturbed data
                    perturbated_X_per_sigma_pert[i].append(smoothed_input_data_pert)
                    outputs_per_sigma_pert[i].append(Pred_pert_smoothed)

                # Smooth unperturbed data
                elif smooth_unperturbed_data and not smooth_perturbed_data:
                    # Add gaussian noise to the unperturbed data
                    smoothed_input_data_unpert, Pred_unpert_smoothed = Smoothing.forward_pass_smoothing(X,sigma,T, Domain, num_samples,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method,img,img_m_per_px,num_steps)

                    # append the smoothed unperturbed data
                    perturbated_X_per_sigma_unpert[i].append(smoothed_input_data_unpert)
                    outputs_per_sigma_unpert[i].append(Pred_unpert_smoothed)

                # Smooth both perturbed and unperturbed data
                else:
                    # Add gaussian noise to the perturbed data
                    smoothed_input_data_pert, Pred_pert_smoothed = Smoothing.forward_pass_smoothing(X_new_adv,sigma,T, Domain, num_samples,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method,img,img_m_per_px,num_steps)

                    # append the smoothed perturbed data
                    perturbated_X_per_sigma_pert[i].append(smoothed_input_data_pert)
                    outputs_per_sigma_pert[i].append(Pred_pert_smoothed)
    
                    # Add gaussian noise to the unperturbed data
                    smoothed_input_data_unpert, Pred_unpert_smoothed = Smoothing.forward_pass_smoothing(X,sigma,T, Domain, num_samples,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method,img,img_m_per_px,num_steps)

                    # append the smoothed unperturbed data
                    perturbated_X_per_sigma_unpert[i].append(smoothed_input_data_unpert)
                    outputs_per_sigma_unpert[i].append(Pred_unpert_smoothed)
        
        # Return to array
        perturbated_X_per_sigma_pert = np.array(perturbated_X_per_sigma_pert)
        perturbated_X_per_sigma_unpert = np.array(perturbated_X_per_sigma_unpert)
        outputs_per_sigma_pert = np.array(outputs_per_sigma_pert)
        outputs_per_sigma_unpert = np.array(outputs_per_sigma_unpert)

        # Return randomly selected data
        perturbated_X_per_sigma_pert_selection, outputs_per_sigma_pert_selection = Smoothing.randomly_select_samples_smoothing(perturbated_X_per_sigma_pert,outputs_per_sigma_pert, num_samples_used_smoothing, num_samples)
        perturbated_X_per_sigma_unpert_selection, outputs_per_sigma_unpert_selection = Smoothing.randomly_select_samples_smoothing(perturbated_X_per_sigma_unpert,outputs_per_sigma_unpert, num_samples_used_smoothing, num_samples)

        if smooth_perturbed_data and not smooth_unperturbed_data:
            return perturbated_X_per_sigma_pert_selection, outputs_per_sigma_pert_selection, None, None
        elif smooth_unperturbed_data and not smooth_perturbed_data:
            return None, None, perturbated_X_per_sigma_unpert_selection, outputs_per_sigma_unpert_selection
        else:
            return perturbated_X_per_sigma_pert_selection, outputs_per_sigma_pert_selection, perturbated_X_per_sigma_unpert_selection, outputs_per_sigma_unpert_selection

    @staticmethod
    def forward_pass_smoothing(input_data,sigma,T, Domain, num_samples,pert_model,mask_values_X,flip_dimensions,dt,epsilon_acc,epsilon_curv,smoothing_method,img,img_m_per_px,num_steps):
        # Add noise to the target agent
        if smoothing_method == 'position':
            input_data_noised = Smoothing.add_noise_positions(input_data,sigma)
        elif smoothing_method == 'control_action':
            input_data_noised = Smoothing.add_noise_control_actions(input_data, mask_values_X, flip_dimensions, dt, sigma,epsilon_acc,epsilon_curv)
        else:
            raise ValueError("Please give either 'position' or 'control_action' as input to the randomized smoothing.")

        # Make the prediction and calculate the expectation
        Pred_smoothed = pert_model.predict_batch_tensor(input_data_noised, T, Domain,img, img_m_per_px,num_steps, num_samples)

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
    
    @staticmethod
    def randomly_select_samples_smoothing(array_past,array_future, num_samples_used,num_samples):
        # flatten data
        shape_future = array_future.shape
        new_shape = (shape_future[0],shape_future[2], shape_future[1] * shape_future[3]) + shape_future[4:]
        new_array_future = array_future.transpose(0,2,1,3,4,5).reshape(new_shape)


        # Select random samples
        random_samples = np.random.choice(new_array_future.shape[2], size=num_samples_used, replace=False) + 1
        selected_samples_future = new_array_future[:, :, random_samples]

        # Select the correct input samples
        random_samples_past = np.ceil(random_samples/num_samples) - 1 
        random_samples_past = random_samples_past.astype(int)

        selected_samples_past = []

        for i in range(len(random_samples_past)):
            selected_samples_past.append(array_past[:,random_samples_past[i],:,:,:,:])
            
        selected_samples_past = np.array(selected_samples_past)

        return selected_samples_past, selected_samples_future
    
    @staticmethod
    def draw_arrow(data_X, data_Y, figure_input, color, linewidth,line_style_input,line_style_output, label_input,label_output,alpha_input,alpha_output):
        figure_input.plot(data_X[:,0], data_X[:,1], linestyle=line_style_input,linewidth=linewidth, color=color, label=label_input,alpha=alpha_input)
        figure_input.plot((data_X[-1,0],data_Y[0,0]), (data_X[-1,1],data_Y[0,1]), linestyle=line_style_output,linewidth=linewidth, color=color,alpha=alpha_output)
        figure_input.plot(data_Y[:-1,0], data_Y[:-1,1], linestyle=line_style_output,linewidth=linewidth, color=color,alpha=alpha_output,label=label_output)
        figure_input.annotate('', xy=(data_Y[-1,0], data_Y[-1,1]), xytext=(data_Y[-2,0], data_Y[-2,1]),
                size=20,arrowprops=dict(arrowstyle='-|>',linestyle=None,color=color,lw=linewidth,alpha=alpha_output))

    
