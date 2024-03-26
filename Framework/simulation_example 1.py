#%%
from experiment import Experiment

# Draw latex figure
Experiment_name = 'Initial_test'
new_experiment = Experiment(Experiment_name)


#%% Select modules
# Select the datasets
Data_sets = [{'scenario': 'CoR_left_turns', 'max_num_agents': None, 't0_type': 'col_set', 'conforming_t0_types': [], 'perturbation': {'attack': 'Adversarial',
                                                                                                                                        'data_set_dict': {'scenario': 'CoR_left_turns', 'max_num_agents': None, 't0_type': 'col_set', 'conforming_t0_types': []},
                                                                                                                                        'data_param': {'dt': 0.1, 'num_timesteps_in': (12,12), 'num_timesteps_out': (12, 12)},
                                                                                                                                        'splitter_dict': {'Type': 'Random_split', 'repetition': [0], 'test_part': 0.2},
                                                                                                                                        'model_dict': {'model': 'trajectron_salzmann_old','kwargs': {'seed':0}}}}]

# Select the spitting methods to be considered
Splitters = [{'Type': 'Random_split', 'repetition': [0], 'test_part': 0.2}]

# Select the models to be trained
Models = [{'model': 'trajectron_salzmann_old',
	   'kwargs': {'seed': 0}}]

# Select the params for the datasets to be considered
# Data_params = [{'dt': 0.1, 'num_timesteps_in': (7,7), 'num_timesteps_out': (30, 30), 'adv_gen': {'splitter_param': Splitters[0]}}] 

Data_params = [{'dt': 0.1, 'num_timesteps_in': (12,12), 'num_timesteps_out': (12, 12)}] 

# Data_params = [{'dt': 0.1, 'num_timesteps_in': (7,7), 'num_timesteps_out': (22, 22)}] 
          

# Select the metrics to be used
Metrics = ['ADE20_indep', 'minADE20_indep', 'FDE20_indep', 'minFDE20_indep', 'ECE_class', 'AUC_ROC']


new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)

#%% Other settings
# Set the number of different trajectories to be predicted by trajectory prediction models.
num_samples_path_pred = 20

# Deciding wether to enforce start times (or allow later predictions if not enough input data is available)
enforce_prediction_time = True

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
evaluate_on_train_set = False

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann_old'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_time, 
                              exclude_post_crit, allow_extrapolation, 
                              agents_to_predict, overwrite_results, evaluate_on_train_set)




#%% Run experiment
new_experiment.run() 

# Load results
Results, Train_Loss = new_experiment.load_results(plot_if_possible = True, return_train_loss = True)

new_experiment.write_tables(dataset_row = True, use_scriptsize = False, depict_std = True)

new_experiment.plot_paths(load_all = False, plot_similar_futures = False, plot_train = False,
                          only_show_pred_agents = False, likelihood_visualization = False, plot_only_lines = False)