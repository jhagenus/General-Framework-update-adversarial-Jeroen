import torch

class Control_action:
    @staticmethod
    def Reversed_Dynamical_Model(X, mask_values_X, flip_dimensions, new_input,dt):
        # JULIAN: We can simplify this a lot by implementing the following:
        # Replace X with new_input, as X.shape[0] == new_input.shape[0]
        # Rename the parameters to something more meaningful, as somebody reading this code might 
        # not remember what new_input etc. where in the higher level code
        # For example, we can rename mask_values_X to mask, and new_input to positions

        # JULIAN: Write some preamble for the function, explaining what the parameters do

        # Initialize control action
        control_action = torch.zeros_like(new_input)

        # Initialize heading and velocity
        heading_init = torch.zeros(X.shape[0]).to(device='cuda')
        velocity_init = torch.zeros(X.shape[0]).to(device='cuda')

        # JULIAN: Instead of using a for loop, we can instead use something like 
        # control_action[mask_values_X] = 0, or control_action[~mask_values_X] = 0
        for batch_idx in range(X.shape[0]):
            # Check if the target agent is standing still set all control actions to zero
            # JULIAN: why do we we have flip dimensions here? I the agent is not moving, does it really matter if we flip the dimensions or not?
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

                # JULIAN: It is liekly possible to vectorize this code, as the control actions are independent of each other
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
                    control_action[batch_idx,0,i,1] = curvature 

        # JULIAN: Using torch.nan_to_num to replace inf values is likely better, as you can distinguish between inf and -inf values
        control_action[torch.isinf(control_action)] = 1e-6
        # JULIAN: Right now, you have a last timestep in control_action that you never use. Therefore, it might be better if
        # you initialize control action to be 1 step shorter along the time dimension.
        # JULIAN: Additionally, as we only perturb the first agent (i.e., [:,0]), it might be easier to simply remove the agent dimension 
        # from control action

        return control_action, heading_init, velocity_init
    
    @staticmethod
    def compute_heading(X, batch_idx, index):
        # JULIAN: It might be more comprehensible, if we assume that X has no agent dimension, and instead 
        # put in the [:,0] when we call the function

        # Calculate dx and dy
        dx = X[batch_idx, 0, index + 1, 0] - X[batch_idx, 0, index, 0]  
        dy = X[batch_idx, 0, index + 1, 1] - X[batch_idx, 0, index, 1]  
        return torch.atan2(dy, dx) 
    
    @staticmethod
    def compute_velocity(X, batch_idx, dt, index):
        # JULIAN: See the comment in compute_heading
        return torch.linalg.norm(X[batch_idx,0,index + 1,:] - X[batch_idx,0,index,:] , dim=-1, ord = 2) / dt
    
    @staticmethod
    def dynamical_model(new_input, control_action, velocity_init, heading_init, dt):
        # JULIAN: Ass mentioned above, code gets more comprehensible if we eliminate
        # unneeded function arguments (such as new_input, we can use control_action instead to get the required shape)
        
        # Adversarial position storage
        adv_position = new_input.clone().detach()

        # Update adversarial position based on dynamical model
        acc = control_action[:,0,:-1,0]
        cur = control_action[:,0,:-1,1]

        # Calculate the velocity for all time steps
        Velocity_set = torch.cumsum(acc, dim=1) * dt + velocity_init.unsqueeze(1)
        Velocity = torch.cat((velocity_init.unsqueeze(1), Velocity_set), dim=1)

        # Calculte the change of heading for all time steps
        D_yaw_rate = Velocity[:,:-1] * cur

        # Calculate Heading for all time steps
        Heading = torch.cumsum(D_yaw_rate, dim=1) * dt + heading_init.unsqueeze(1)
        Heading = torch.cat((heading_init.unsqueeze(1), Heading), dim=1)

        # Calculate the new position for all time steps
        adv_position[:, 0, 1:, 0] = torch.cumsum(Velocity[:,1:] * torch.cos(Heading[:,1:]), dim=1) * dt + adv_position[:, 0, 0, 0].unsqueeze(1)
        adv_position[:, 0, 1:, 1] = torch.cumsum(Velocity[:,1:] * torch.sin(Heading[:,1:]), dim=1) * dt + adv_position[:, 0, 0, 1].unsqueeze(1)

        return adv_position