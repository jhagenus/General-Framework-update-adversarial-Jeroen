import torch

class Control_action:
    @staticmethod
    def get_control_actions(X, mask_values_X, flip_dimensions, new_input,dt):
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
                heading_init[batch_idx] = Control_action.compute_heading(X, batch_idx, 0)

                # update control actions
                control_action[batch_idx,:,:,:] = 0 
                          
            else:
                # Initialize storage for velocity and heading
                velocity = torch.zeros(new_input.shape[2]).to(device='cuda')
                angle = torch.zeros(new_input.shape[2]).to(device='cuda')

                # update initial velocity 
                velocity_init[batch_idx] = velocity[0] = Control_action.compute_velocity(new_input, batch_idx, dt, 0)

                # update initial heading
                heading_init[batch_idx] = angle[0] = Control_action.compute_heading(new_input, batch_idx,0)

                for i in range(new_input.shape[2]-1):
                    # Calculate velocity for next time step
                    velocity[i+1] = Control_action.compute_velocity(new_input, batch_idx, dt, i)

                    # Update the control actions
                    control_action[batch_idx,0,i,0] = (velocity[i+1] - velocity[i]) / dt 

                    # Calculate the heading for the next time step   
                    angle[i+1] = Control_action.compute_heading(new_input, batch_idx,i)

                    # Calculate the change of heading for the next time step
                    d_yaw_rate = (angle[i+1] - angle[i]) / dt

                    # Calculate the curvature 
                    curvature = d_yaw_rate / velocity[i]
                    control_action[batch_idx,0,i-1,1] = curvature 

        control_action[torch.isinf(control_action)] = 1e-6
        control_action.requires_grad = True 

        return control_action, heading_init, velocity_init
    
    @staticmethod
    def compute_heading(X, batch_idx, index):
        # Calculate dx and dy
        dx = X[batch_idx, 0, index + 1, 0] - X[batch_idx, 0, index, 0]  
        dy = X[batch_idx, 0, index + 1, 1] - X[batch_idx, 0, index, 1]  
        return torch.atan2(dy, dx) 
    
    @staticmethod
    def compute_velocity(X, batch_idx, dt, index):
        return torch.linalg.norm(X[batch_idx,0,index + 1,:] - X[batch_idx,0,index,:] , dim=-1, ord = 2) / dt
    
    @staticmethod
    def dynamical_model(new_input, control_action, velocity_init, heading_init, dt):
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