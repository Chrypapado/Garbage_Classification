# Import Libraries and Modules
import sys
import argparse
from pathlib import Path
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace

# Parent Path
project_dir = Path(__file__).resolve().parents[2]

# Arguments
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--num_epochs', type=int, default=10)
args = parser.parse_args(sys.argv[2:])

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Create a Python environment for the experiment
project_env = Environment("training-env")

# Ensure the required packages are installed
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults', 'joblib', 'torch', 'torchvision',
                                                  'matplotlib', 'numpy', 'opencv-python', 'kornia'])
project_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory='./',
                                script='azure_train.py',
                                environment=project_env)

# Submit the experiment run
experiment_name = 'final_project'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)

# Block until the experiment run has completed
run.wait_for_completion()

# Register the trained models
for epoch in range(10):
    run.register_model(model_path=project_dir.joinpath('models/azure/model' + str(epoch) + '.pth'),
                       model_name='model' + str(epoch),
                       tags={'Training context': 'Script'},
                       properties={'Validation Accuracy': run.get_metrics()['Validation Accuracy'][epoch]})
