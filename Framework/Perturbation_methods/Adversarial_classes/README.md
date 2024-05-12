# Adverarial attack generation on Left-Turn dataset

## General setting
### Adversarial attack
For the generation of adversarial attacks, four important settings need to be configured:
- Setting the number of iterations:
```
iter_num = 50
```
- Setting the number of predictions:
```
num_samples = 20 # defined as (K) in paper
```
- Setting the bounded control actions for acceleration ($m/s^2$) and curvature ($\kappa$):
```
epsilon_acc = 7
epsilon_curv = 0.20
```
- Setting an exponential decay learning rate ($\alpha = \gamma ^{iter} \cdot \alpha_{0}$) for both acceleration ($m/s^2$) and curvature ($\kappa$):
```
learning_rate_decay = True
gamma = 1
alpha = 0.01
```

### Selecting the Agent to Perturb
Two different agents in the scene can be perturbed:
- When perturbing the car that takes the corner:
```
flip_dimensions = True
```
- When perturbing the car that drives straight:
```
flip_dimensions = False
```

### Car size
To modify the size of the car ($m$) used in the animation, the length and width can be adjusted accordingly:
```
car_length = 4.1
car_width = 1.7
```

## Barrier function
The barrier function utilized in the paper takes two inputs: the ground truth values of $X$ and spline values $S$ collected from $X$ and $Y$. For both, a maximum distance ($m$) can be set using:
```
distance_threshold = 1
```
**Note:** Only one can be set to True. When both are set to False, no barrier function is utilized.
### Barrier - Ground truth
$$
r_{\text{barrier}} = -\frac{1}{H} \sum_{t=1}^{H} \log \left( D_{\text{Adversarial\;threshold}} - \left\| \tilde{X}_{\text{tar}}^{t} -  X_{\text{tar}}^{t} \right\|_2 \right)
$$ <br />
To utilize the regularization term for the observed states ($X{tar}$), two settings are available:

- To utilize this barrier function, set:
```
log_barrier = True
```
- To change the weight of the regularization term, change the following value:
```
log_value = np.e
```
### Barrier - Spline
$r_{\text{barrier spline}} = -\frac{1}{H} \sum_{t=1 \dots H} \log ( D_{\text{Adversarial\;threshold}} - \left\| \tilde{X}_{\text{tar}}^{t} -  S_{\text{tar}} \right\|_2)$ <br />
To utilize the regularization term for the observed states ($X{tar}$), three settings are available:

- To utilize this barrier function, set:
```
spline_barrier = True
```
- To change the weight of the regularization term, change the following value:
```
log_value = np.e
```
- To determine how many points are interpolated on the target trajectory ($X_{tar},Y_{tar}$), change the following value:
```
spline_interval = 100
```

## Loss function
The loss functions are used to create adversarial trajectory to a certain direction. **Note** Only one of the loss functions can be set on True.

### Loss - Average Displacement Error
$l_{\text{ADE}} = -\frac{1}{T} \sum_{t=1 \dots T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - {Y}_{\text{tar}}^{t} \right\|_2$ <br />
To utilize the ADE loss function, set:
```
ADE_loss = True
```
### Loss - Average Displacement Error (new) - Ground truth
$$l_{\text{ADE\;(new)}} = -\frac{1}{T} \sum_{t=1 \dots T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - \tilde{Y}_{\text{tar}}^{t} \right\|_2 + \frac{1}{T} \sum_{t=1 \dots T} \left\| \tilde{Y}_{\text{tar}}^{t} -  Y_{\text{tar}}^{t} \right\|_2$$ <br />
To utilize the ADE loss function using a perturbed future ($\tilde{Y}{tar}$) closely resembling the ground truth future ($\hat{Y}_{tar}$), set:
```
ADE_loss_adv_future_GT = True
```
### Loss - Average Displacement Error (new) - Prediction
$l_{\text{ADE\;(new)}} = -\frac{1}{T} \sum_{t=1 \dots T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - \tilde{Y}_{\text{tar}}^{t} \right\|_2 + \frac{1}{T} \sum_{t=1 \dots T} \left\| \tilde{Y}_{\text{tar}}^{t} -  \hat{Y}_{\text{tar}}^{t} \right\|_2$ <br />
To utilize the ADE loss function using a perturbed future ($\tilde{Y}_{tar}$) closely resembling the ground truth future (${Y}_{tar}$), set:
```
ADE_loss_adv_future_pred = True
```

### Loss - Collision
$l_{\text{Collision}} =  \min_{t} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - Y_{\text{ego}}^{t} \right\|_2$ <br />
To utlize the collision loss function, set:
```
collision_loss = True
```

### Loss - Fake collision - Ground truth
$l_{\text{Fake\;collision}} =  \min_{t} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - Y_{\text{ego}}^{t} \right\|_2 + \frac{1}{T} \sum_{t=1 \dots T} \left\| \tilde{Y}_{\text{tar}}^{t} -  Y_{\text{tar}}^{t} \right\|_2$ <br />
To utilize the fake collision function using a perturbed future ($\tilde{Y}_{tar}$) closely resembling the ground truth future ($Y_{tar}$), set:
```
fake_collision_loss_GT = True
```

### Loss - Fake collision - Prediction
$l_{\text{Fake\;collision}} =  \min_{t} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - Y_{\text{ego}}^{t} \right\|_2 + \frac{1}{T} \sum_{t=1 \dots T} \left\| \tilde{Y}_{\text{tar}}^{t} -  \hat{Y}_{\text{tar}}^{t} \right\|_2$ <br />
To utilize the fake collision function using a perturbed future ($\tilde{Y}_{tar}$) closely resembling the predicted future ($\hat{Y}_{tar}$), set:
```
fake_collision_loss_pred = True
```

### Loss - Hide collision - Ground truth
$l_{\text{Hide\;Collision}} =  \min_{t} \left\| \tilde{Y}_{\text{tar}}^{t} - Y_{\text{ego}}^{t} \right\|_2 + \frac{1}{T} \sum_{t=1 \dots T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} -  Y_{\text{tar}}^{t} \right\|_2$ <br />
To utilize the fake collision function using a predicted perturbed future ($\hat{\tilde{Y}}_{tar}$) closely resembling the ground truth future ($Y_{tar}$), set:
```
hide_collision_loss_GT = True
```

### Loss - Hide collision - Prediction
$l_{\text{Hide\;Collision}} =  \min_{t} \left\| \tilde{Y}_{\text{tar}}^{t} - Y_{\text{ego}}^{t} \right\|_2 + \frac{1}{T} \sum_{t=1 \dots T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} -  \hat{Y}_{\text{tar}}^{t} \right\|_2$ <br />
To utilize the fake collision function using a predicted perturbed future ($\hat{\tilde{Y}}_{tar}$) closely resembling the predicted future ($\hat{Y}_{tar}$), set:
```
hide_collision_loss_pred = True
```

## Gaussian smoothing

### Settings
For the application of Gaussian smoothing, five settings are available:
- For the smoothing strategy, two options can be selected to either smooth the Cartesian coordinates or the control actions:
```
smoothing_method = 'position' or 'control_action'
```

- To select the total number of smoothed paths:
```
num_samples_smoothing = 18 # defined as (R) in paper
```
- To select the number of randomly selected predictions:
```
num_samples_used_smoothing = 20 # defined as (M) in paper
```
- To select the sigmas to analyze, a list can be filled:
```
sigmas = [0.05,0.1]
```
- To smooth the nominal observed states ($X_{tar}$), set:
```
smooth_unperturbed_data = True
```
- To smooth the adversarial observed states ($\tilde{X}_{tar}$), set:
```
smooth_perturbed_data = True
```

## Plot the data

### Plot - benign scene

To plot the unperturbed data ($X_{tar},Y_{tar}$) and ($X_{ego},Y_{ego}$), set:

```
plot_input = True
```

### Plot - spline

To plot the splined data ($S$) used in the barrier function, set:
```
plot_spline = True
```
**Note:** This only works when bening scene is also plotted.

### Plot - Loss

To plot the loss of the adversarial attack for all the iteration, set:
```
plot_loss = True
```

### Plot - Adversarial scene - Static

To plot the adversarial scene without animations, set:
```
static_adv_scene = True
```

### Plot - Adversarial scene - animated

To plot the adversarial scene with animation, set:

```
animated_adv_scene = True
```

To add control actions using bar visualisation:
```
control_action_bar = True
```
To add control actions using graph visualisation:
```
control_action_graph = True
```
**Note:** Only one can be set on True, when bot are False control actions are not visualised in animation.
### Plot - Gaussian smoothing

To plot the gaussian smoothed trajectories and predictions, set:
```
plot_smoothing = True
```


