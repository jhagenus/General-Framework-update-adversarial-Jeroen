# Adverarial attack generation

## General setting

### Adversarial attack
For the generation of adversarial attacks, four important settings need to be configured:
- Setting the number of predictions:
```
self.num_samples = 20 
```
- Setting the max number of iterations:
```
self.max_number_iterations = 50
```
- Setting an exponential decay learning rate ($\alpha = \gamma ^{iter} \cdot \alpha_{0}$):
```
self.gamma = 1
self.alpha = 0.01
```

### Car size
To modify the size of the car ($m$) used in the animation, the length and width can be adjusted accordingly:
```
self.car_length = 4.1
self.car_width = 1.7
self.wheelbase = 2.7
```

### Clamping
For the adversarial attack strategy, 'Adversarial_Control_Action', the perturbed control action values are clamped absolute and relative.
```
self.epsilon_curv_absolute = 0.2
```
```
self.epsilon_acc_relative = 2
self.epsilon_curv_relative = 0.05
```

## Attack function
The attack function can be selected (See table):
```
self.loss_function_1 = 'ADE_Y_GT_Y_Pred_Max' # Mandotary
self.loss_function_2 = None # Option if not used set to None
```


| Type attack   | First input   | Second input  | Objective     | Formula       | Name framework (str)  | 
| ------------- | ------------- | ------------- | ------------- | ------------- |   -------------       |
| ADE           | $`{Y_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Maximize distance | $-\frac{1}{T} \sum_{t=1}^{T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - {Y}_{\text{tar}}^{t} \right\|_2}`$ | 'ADE_Y_GT_Y_Pred_Max' |
| ADE           | $Y_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Minimize distance | $\frac{1}{T} \sum_{t=1}^{T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - {Y}_{\text{tar}}^{t} \right\|_2$ | 'ADE_Y_GT_Y_Pred_Min' |
| FDE           | $Y_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Maximize distance | $- \left\| \hat{\tilde{Y}}_{\text{tar}}^{T} - {Y}_{\text{tar}}^{T} \right\|_2$ | 'FDE_Y_GT_Y_Pred_Max' |
| FDE           | $Y_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Minimize distance | $ \left\| \hat{\tilde{Y}}_{\text{tar}}^{T} - {Y}_{\text{tar}}^{T} \right\|_2$ | 'FDE_Y_GT_Y_Pred_Min' |
| ADE           | $\tilde{Y}_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Maximize distance | $-\frac{1}{T} \sum_{t=1}^{T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - \tilde{Y}_{\text{tar}}^{t} \right\|_2$ | 'ADE_Y_Perturb_Y_Pred_Max' |
| ADE           | $\tilde{Y}_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Minimize distance | $\frac{1}{T} \sum_{t=1}^{T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - \tilde{Y}_{\text{tar}}^{t} \right\|_2$ | 'ADE_Y_Perturb_Y_Pred_Min' |
| FDE           | $\tilde{Y}_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Maximize distance | $- \left\| \hat{\tilde{Y}}_{\text{tar}}^{T} - \tilde{Y}_{\text{tar}}^{T} \right\|_2$ | 'FDE_Y_Perturb_Y_Pred_Max' |
| FDE           | $\tilde{Y}_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Minimize distance | $ \left\| \hat{\tilde{Y}}_{\text{tar}}^{T} - \tilde{Y}_{\text{tar}}^{T} \right\|_2$ | 'FDE_Y_Perturb_Y_Pred_Min' |
| ADE           | $\tilde{Y}_{\text{tar}}$  | ${Y}_{\text{tar}}$ | Maximize distance | $-\frac{1}{T} \sum_{t=1}^{T} \left\| {Y}_{\text{tar}}^{t} - \tilde{Y}_{\text{tar}}^{t} \right\|_2$ | 'ADE_Y_Perturb_Y_GT_Max' |
| ADE           | $\tilde{Y}_{\text{tar}}$  | ${Y}_{\text{tar}}$ | Minimize distance | $\frac{1}{T} \sum_{t=1}^{T} \left\| {Y}_{\text{tar}}^{t} - \tilde{Y}_{\text{tar}}^{t} \right\|_2$ | 'ADE_Y_Perturb_Y_GT_Min' |
| FDE           | $\tilde{Y}_{\text{tar}}$  | ${Y}_{\text{tar}}$ | Maximize distance | $- \left\| {Y}_{\text{tar}}^{T} - \tilde{Y}_{\text{tar}}^{T} \right\|_2$ | 'FDE_Y_Perturb_Y_GT_Max' |
| FDE           | $\tilde{Y}_{\text{tar}}$  | ${Y}_{\text{tar}}$ | Minimize distance | $\ \left\| {Y}_{\text{tar}}^{T} - \tilde{Y}_{\text{tar}}^{T} \right\|_2$ | 'FDE_Y_Perturb_Y_GT_Min' |
| ADE           | $\hat{Y}_{\text{tar}}$  | $\tilde{Y}_{\text{tar}}$ | Maximize distance | $-\frac{1}{T} \sum_{t=1}^{T} \left\| \tilde{Y}_{\text{tar}}^{t} - \hat{Y}_{\text{tar}}^{t} \right\|_2$ | 'ADE_Y_pred_iteration_1_and_Y_Perturb_Max' |
| ADE           | $\hat{Y}_{\text{tar}}$  | $\tilde{Y}_{\text{tar}}$ | Minimize distance | $\frac{1}{T} \sum_{t=1}^{T} \left\| \tilde{Y}_{\text{tar}}^{t} - \hat{Y}_\text{tar}^{t} \right\|_2$ | 'ADE_Y_pred_iteration_1_and_Y_Perturb_Min' |
| FDE           | $\hat{Y}_{\text{tar}}$  | $\tilde{Y}_{\text{tar}}$ | Maximize distance | $-\left\| \tilde{Y}_{\text{tar}}^{T} - \hat{Y}_{\text{tar}}^{T} \right\|_2$ | 'FDE_Y_pred_iteration_1_and_Y_Perturb_Max' |
| FDE           | $\hat{Y}_{\text{tar}}$  | $\tilde{Y}_{\text{tar}}$ | Minimize distance | $\left\| \tilde{Y}_{\text{tar}}^{T} - \hat{Y}_{\text{tar}}^{T} \right\|_2$ | 'FDE_Y_pred_iteration_1_and_Y_Perturb_Min' |
| ADE           | $\hat{\tilde{Y}}_{\text{tar}}$  | $\hat{Y}_{\text{tar}}$ | Maximize distance | $-\frac{1}{T} \sum_{t=1}^{T} \left\| \hat{Y}_{\text{tar}}^{t} - \hat{\tilde{Y}}_{\text{tar}}^{t} \right\|_2$ | 'ADE_Y_pred_and_Y_pred_iteration_1_Max' |
| ADE           | $\hat{\tilde{Y}}_{\text{tar}}$  | $\hat{Y}_{\text{tar}}$ | Minimize distance | $\frac{1}{T} \sum_{t=1}^{T} \left\| \hat{Y}_{\text{tar}}^{t} - \hat{\tilde{Y}}_\text{tar}^{t} \right\|_2$ | 'ADE_Y_pred_and_Y_pred_iteration_1_Min' |
| FDE           | $\hat{\tilde{Y}}_{\text{tar}}$  | $\hat{Y}_{\text{tar}}$ | Maximize distance | $- \left\| \hat{Y}_{\text{tar}}^{T} - \hat{\tilde{Y}}_{\text{tar}}^{T} \right\|_2$ | 'ADE_Y_pred_and_Y_pred_iteration_1_Max' |
| FDE           | $\hat{\tilde{Y}}_{\text{tar}}$  | $\hat{Y}_{\text{tar}}$ | Minimize distance | $ \left\| \hat{Y}_{\text{tar}}^{T} - \hat{\tilde{Y}}_\text{tar}^{T} \right\|_2$ | 'ADE_Y_pred_and_Y_pred_iteration_1_Min' |
| Collision     | $\hat{\tilde{Y}}_{\text{tar}}$  | ${Y}_{\text{ego}}$ | Minimize smallest distance | $\min_{t \in \{1, \ldots, T\}} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - Y_{\text{ego}}^{t} \right\|_2$ | 'Collision_Y_pred_tar_Y_GT_ego' |
| Collision     | $\tilde{Y}_{\text{tar}}$  | ${Y}_{\text{ego}}$ | Minimize smallest distance | $\min_{t \in \{1, \ldots, T\}} \left\| \tilde{Y}_{\text{tar}}^{t} - Y_{\text{ego}}^{t} \right\|_2$ | 'Collision_Y_Perturb_tar_Y_GT_ego' |

## Barrier function
The barrier function can be selected (See table):
```
 self.barrier_function_past = 'Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific' or None
 self.barrier_function_future = 'Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific' or None
```

For the barrier function a distance threshold can be selected ($D_{\text{Adversarial threshold}}$):
```
 self.distance_threshold_past = 1
 self.distance_threshold_future = 1
```

For the barrier function the weight of the penalty can be changed:
```
self.log_value_past = 1.5
self.log_value_future = 1.5
```

| Type Regularization  | First input   | Second input  | Third input |Objective     | Formula       | Name framework (str)  | 
| ------------- | ------------- | ------------- | ------------- | ------------- |   ------------- | ------------- |
| Time specific | ${X}_{\text{tar}}$ | $\tilde{X}_{\text{tar}}$ |  | Past states | $-\frac{1}{H} \sum_{t=-H+1}^{0} \log_{e} ( D_{\text{Adversarial\;threshold}} - \left\| \tilde{X}_{\text{tar}}^{t} -  X_{\text{tar}}^{t} \right\|_2)$ | 'Time_specific'|
| Trajectory specific | ${X}_{\text{tar}}$ | $\tilde{X}_{\text{tar}}$ | $Z_{\text{tar}}$ | Past states | $-\frac{1}{H} \sum_{t=-H+1}^{0} \log_{e} ( D_{\text{Adversarial\;threshold}} - \min (\min_{t_{\text{eval}} \in \{-H+1, \ldots, 0\}} \left\| \tilde{X}^{t}_{\text{tar}} -  X^{t_{\text{eval}}}_{\text{tar}} \right\|_2, \min_{z \in \{1, \ldots, H-1\}} (d_{\perp} (\tilde{X}_{\text{tar}}^{t},  Z_{\text{tar}}^{z})))$ | 'Trajectory_specific' |
| Time and Trajectory specific | ${X}_{\text{tar}}$ | $\tilde{X}_{\text{tar}}$ | $Z_{\text{tar}}$ | Past states | $-\frac{1}{H} \sum_{t=-H+1}^{0} \log_{e} ( D_{\text{Adversarial\;threshold}} - \min (\min_{t_{\text{eval}} \in \{-H+1, \ldots, 0\}} \left\| \tilde{X}^{t}_{\text{tar}} -  X^{t_{\text{eval}}}_{\text{tar}} \right\|_2, \min_{z \in \{1, \ldots, H-1\}} (d_{\perp} (\tilde{X}_{\text{tar}}^{t},  Z_{\text{tar}}^{z}))))  - \log_{e} ( D_{\text{Adversarial\;threshold}} - \left\| \tilde{X}_{\text{tar}}^{T} -  X_{\text{tar}}^{T} \right\|_2)$ | 'Time_Trajectory_specific' |
|                               |
| Time specific | ${Y}_{\text{tar}}$ | $\tilde{Y}_{\text{tar}}$ | |  Future states | $-\frac{1}{T} \sum_{t=1}^{T} \log_{e} ( D_{\text{Adversarial\;threshold}} - \left\| \tilde{Y}_{\text{tar}}^{t} -  Y_{\text{tar}}^{t} \right\|_2)$ | 'Time_specific' | 
| Trajectory specific | ${Y}_{\text{tar}}$ | $\tilde{Y}_{\text{tar}}$ | $Z_{\text{tar}}$ | Future states | $-\frac{1}{T} \sum_{t=1}^{T} \log_{e} ( D_{\text{Adversarial\;threshold}} - \min (\min_{t_{\text{eval}} \in \{1, \ldots, T\}} \left\| \tilde{Y}^{t}_{\text{tar}} -  Y^{t_{\text{eval}}}_{\text{tar}} \right\|_2, \min_{z \in \{1, \ldots, T-1\}} (d_{\perp} (\tilde{Y}_{\text{tar}}^{t},  Z_{\text{tar}}^{z})))) $ | 'Trajectory_specific' |
| Time and Trajectory specific | ${Y}_{\text{tar}}$ | $\tilde{Y}_{\text{tar}}$ | $Z_{\text{tar}}$ | Future states | $-\frac{1}{T} \sum_{t=1}^{T} \log_{e} ( D_{\text{Adversarial\;threshold}} - \min (\min_{t_{\text{eval}} \in \{1, \ldots, T\}} \left\| \tilde{Y}^{t}_{\text{tar}} -  Y^{t_{\text{eval}}}_{\text{tar}} \right\|_2, \min_{z \in \{1, \ldots, T-1\}} (d_{\perp} (\tilde{Y}_{\text{tar}}^{t},  Z_{\text{tar}}^{z})))) - \log_{e} ( D_{\text{Adversarial\;threshold}} - \left\| \tilde{Y}_{\text{tar}}^{T} -  Y_{\text{tar}}^{T} \right\|_2)$ | 'Time_Trajectory_specific' |

## Gaussian smoothing

### Settings
- To select the total number of smoothed paths and predictions (1 per smoothed trajectory):
```
self.num_samples_used_smoothing = 15
```
- To select the sigmas to analyze, a list can be filled:
```
self.sigmas = [0.05,0.1]
```
Specific for the attack type, 'Adversarial_Control_Action', sigmas specific for their control action can be selected:
```
self.sigma_acceleration = [0.05, 0.1]
self.sigma_curvature = [0.01, 0.05]
```

## Plot the data

### General
- Plot the loss over the iterations:
```
self.plot_loss = True
```
- Plot the image used for the neural network:
```
self.image_neural_network = True
```

### Dataset specific (left turns)
- Plot the input data:
```
self.plot_input = True
```
- Plot the adversarial scene (static):
```
self.static_adv_scene = True
```
- Plot the adversarial scene (animated):
```
self.animated_adv_scene = True
```
Set the smoothnes of the animation (higher is better):
```
self.total_spline_values = 100
```
For the attack type, 'Adversarial_Control_Action', control action can be animated:
```
self.control_action_graph = True
```