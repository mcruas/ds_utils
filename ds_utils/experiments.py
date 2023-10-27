import wandb
import itertools

class WandbExperimentLogger:
    
    def __init__(self, project_name, tags=None, notes=None):
        """
        Initialize the logger.

        Parameters:
            project_name (str): The name of the W&B project.
            common_vars (dict): Common variables that are logged in each run.
        """
        self.project_name = project_name
        self.tags = tags
        self.notes = notes

    def log_run(self,  variables={}, metrics={}, run_name=None):
        """
        Log a single run to W&B.

        Parameters:
            run_name (str): The name of the run.
            variables (dict): Run-specific variables.
            metrics (dict): Metrics to log, e.g., 'loss'.
        """
        wandb.init(
            project=self.project_name,
            name=run_name
        )
        
        # Log run-specific variables
        wandb.log({**variables})
        
        # Log metrics
        wandb.log(metrics)

        # End the run
        wandb.finish()


class ExperimentRunner:
    def __init__(self, variables, run_function, logger=None, evaluation_function=None):
        self.variables = variables
        self.run_function = run_function
        self.logger = logger
        self.evaluation_function = evaluation_function

    def flatten_dict(self, d, parent_key='', sep='.'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v if isinstance(v, list) else [v]
        return items

    def generate_combinations(self):
        flat_variables = self.flatten_dict(self.variables)
        keys = flat_variables.keys()
        values = flat_variables.values()
        combinations = [dict(zip(keys, prod)) for prod in itertools.product(*values)]
        return combinations

    def run_experiments(self):
        combinations = self.generate_combinations()
        for i, combination in enumerate(combinations):
            print(f"Running experiment {i+1}/{len(combinations)} with variables {combination}")

            output = self.run_function(**combination)

            eval_metrics = {}
            if self.evaluation_function:
                eval_metrics = self.evaluation_function(output)

            if self.logger:
                self.logger.log_run(variables=combination, metrics=eval_metrics)


def sample_run(**kwargs):
    print(f"Running with {kwargs}")
    return sum(kwargs.values())

def sample_evaluation(output):
    return {"sum": output}

class SimpleLogger:
    def log_run(self, variables, metrics):
        print(f"Logging variables {variables} and metrics {metrics}")

variables = {
    'a': 1,
    'b': [2, 3],
    'c': {'x1': 1, 'x2': [10, 20, 30], 'x3': 100}
}

runner = ExperimentRunner(variables, sample_run, logger=SimpleLogger(), evaluation_function=sample_evaluation)
runner.run_experiments()
