#%%
import copy
from experiment import Experiment

# Draw latex figure
Experiment_name = 'Perturbations'
new_experiment = Experiment(Experiment_name)


#%% Select modules


# Select the models to be trained
Models = [{'model': 'trajectron_salzmann_old','kwargs': {'seed': 0, 'predict_ego': False}}]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.1, 'num_timesteps_in': (12,12), 'num_timesteps_out': (12, 12)}] 

# Select the datasets
Data_sets = []
preturbation = {'attack': None,
                'data_set_dict': {'scenario': 'CoR_left_turns', 'max_num_agents': None, 't0_type': 'col_set', 'conforming_t0_types': []},
                'data_param': Data_params[0],
                'splitter_dict': {'Type': 'no_split', 'repetition': 0, 'train_pert': False, 'test_pert': False},
                'model_dict': Models[0],
                'num_samples_perturb': 20,
                'max_number_iterations': 100,
                'alpha': 0.01,
                'gamma': 0.99,
                'loss_function_1': None,
                'loss_function_2': None,
                'barrier_function_past': None,
                'barrier_function_future': None,
                'distance_threshold_past': 0.9,
                'distance_threshold_future': 0.9,
                'log_value_past': 2.5,
                'log_value_future': 2.5,
                'GT_data': 'no'}

for attack in ['Adversarial_Control_Action', 'Adversarial_Position', 'Adversarial_Search']:
    for loss_function_1, loss_function_2 in [('ADE_Y_GT_Y_Pred_Max', None),
                                             ('FDE_Y_GT_Y_Pred_Max', None),
                                             ('Collision_Y_pred_tar_Y_GT_ego', None),
                                             ('Collision_Y_Perturb_tar_Y_GT_ego', 'ADE_Y_pred_and_Y_pred_iteration_1_Min')]:
        for barrier_function_past in ['Time_specific', 'Time_Trajectory_specific']:
            for barrier_function_future in [None, 'Trajectory_specific']:
                if barrier_function_future is not None:
                    if attack != 'Adversarial_Control_Action':
                        continue
                    if loss_function_2 is not None:
                        continue
                
                # Define specific perturbation
                perturbation_i = copy.deepcopy(preturbation)
                perturbation_i['attack'] = attack
                perturbation_i['loss_function_1'] = loss_function_1
                perturbation_i['loss_function_2'] = loss_function_2
                perturbation_i['barrier_function_past'] = barrier_function_past
                perturbation_i['barrier_function_future'] = barrier_function_future
                dataset = {'scenario': 'CoR_left_turns',  'max_num_agents': None, 't0_type': 'col_set', 'conforming_t0_types': [], 'perturbation': perturbation_i}
                Data_sets.append(dataset)

# Select the spitting methods to be considered
Splitters = [{'Type': 'no_split', 'repetition': 0, 'train_pert': False, 'test_pert': True}]

# Select the metrics to be used
Metrics = ['ADE20_indep', 'FDE20_indep', 'Collision_rate_indep']


new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)

#%% Other settings
# Set the number of different trajectories to be predicted by trajectory prediction models.
num_samples_path_pred = 20

# Deciding wether to enforce start times (or allow later predictions if not enough input data is available)
enforce_prediction_times = True

# determine if the upper bound for n_O should be enforced, or if prediction can be made without
# underlying output data (might cause training problems)
enforce_num_timesteps_out = True

# Determine if the useless prediction (i.e, prediction you cannot act anymore)
# should be exclude from the dataset
exclude_post_crit = False

# Decide wether missing position in trajectory data can be extrapolated
allow_extrapolation = True

# Use all available agents for predictions
agents_to_predict = 'predefined'

# Determine if allready existing results shoul dbe overwritten, or if not, be used instead
overwrite_results = 'no'

# Determine if the model should be evaluated on the training set as well
evaluate_on_train_set = True

# Determine if predictions should be saved
save_predictions = True

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann_old'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_times, 
                              exclude_post_crit, allow_extrapolation, 
                              agents_to_predict, overwrite_results, 
                              save_predictions, evaluate_on_train_set)

#%% Run experiment
new_experiment.run()

# Load results
Results = new_experiment.load_results()
