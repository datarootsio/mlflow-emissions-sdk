from mlflow import MlflowClient
from codecarbon import EmissionsTracker


import mlflow
import mlflow.keras
import torch


class EmissionsTrackerMlflow:
    experiment_tracking_params : dict=None
    tracking_uri : str=""
    run_id : str=""
    exp_id : str=""
    flavor : str=""
    emissions : float=0
    emissions_tracker : EmissionsTracker= None
    device : torch.device = None
    model_acc : float = None
    
    def __init__(self):
        self.emissions_tracker = EmissionsTracker()
    

    def read_params(self, experiment_tracking_params : dict) -> None:
        '''
        Reads the parameters and creates a reference to an mlflow client instance. Creates a run in mlflow

        :param experiment_tracking_params dict: ex. { "tracking_uri": "http://127.0.0.1:5000",
                                                      "experiment_name": "keras_test",
                                                      "run_name": "mobile_net_v2",
                                                      "flavor": "keras" 
                                                      }
        '''
        try:
            
            self.experiment_tracking_params = experiment_tracking_params
            # Creates a reference to the mlflow client
            client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
            # Sets the tracking uri of the mlflow 
            mlflow.set_tracking_uri(self.experiment_tracking_params['tracking_uri'])
            # Looks for the experiment with the given name, creates a new one if none are found
            exp_id = dict(mlflow.set_experiment(self.experiment_tracking_params['experiment_name']))['experiment_id']
            self.exp_id = exp_id
            # registers the model flavor
            self.flavor = experiment_tracking_params['flavor']
            # creates a run with the given name and save the run id
            run_id = dict(client.create_run(exp_id, run_name=self.experiment_tracking_params['run_name'] ))
            self.run_id = dict(run_id['info'])['run_id']
            # starts the mlflow run
            mlflow.start_run(self.run_id, exp_id)

            # specific for keras, autologs the model params and some metrics
            if self.flavor == 'keras':
                mlflow.keras.autolog()
            elif self.flavor == 'pytorch':
                mlflow.pytorch.autolog()
        except:
            print("Please refer to a running instance of mlflow ui")
    
    



    def start_training_job(self):
        self.emissions_tracker.start()
        
    def get_default_device(self) -> None:
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")    

    def to_device(self, data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
    
    def predict_image(self, img, model) -> int:
    # Convert to a batch of 1
        xb = self.to_device(img.unsqueeze(0), self.device)
    # Get predictions from model
        yb = model(xb)
    # Pick index with highest probability
        prob, preds = torch.max(yb, dim=1)
    # Retrieve the class label
        return preds[0].item()

    def end_training_job(self):
        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        emissions = self.emissions_tracker.stop()
        
        mlflow.end_run()

        client.log_metric(self.run_id, "emissions", emissions * 1000)
        
        self.emissions = emissions
        


    def evaluate_model_accuracy(self, model, *args) -> float:
        '''
        Evaluates and logs the model's accuracy on testing data

        :param model: a pytorch/keras ml model
        :param args: the testing data the model 
        '''

        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        model_acc = 0
        # 
        if self.flavor == 'pytorch':
            test_data = args[0]
            total_data_count = len(test_data)
            correctly_predicted = 0
            for testd, res in test_data:
                predicted_val = self.predict_image(testd, model)
                if predicted_val == res:
                    correctly_predicted += 1
            model_acc = correctly_predicted / total_data_count
            
        elif self.flavor == 'keras':
            if len(args) == 1:
                model_acc = model.evaluate(test_data)[1]
                self.model_acc = model_acc
            else:
                x_test = args[0]
                y_test = args[1]
                model_acc = model.evaluate(x_test, y_test)[1]
                self.model_acc = model_acc
        self.model_acc = model_acc
        client.log_metric(self.run_id,"accuracy_on_test_data", model_acc)
        return model_acc


    
    def accuracy_per_emission(self, model, *args):
        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        if len(args) == 2:
            x_test = args[0]
            y_test = args[1]
            if self.model_acc == None :
                self.evaluate_model_accuracy(model, x_test, y_test)
            acc_per_emission = (self.model_acc) / (self.emissions * 1000)
            client.log_metric(self.run_id, "accuracy_per_emission", acc_per_emission)
        else:
            test_data = args[0]
            if self.model_acc == None :
                self.evaluate_model_accuracy(model, test_data)
            acc_per_emission = (self.model_acc) / (self.emissions * 1000)
            client.log_metric(self.run_id, "accuracy_per_emission", acc_per_emission)
    


    def emission_per_10_inferences(self, model, test_data):
        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        # For pytorch  
        if self.flavor == 'pytorch':
            self.emissions_tracker.start()
            for i in range(10):
                self.predict_image(test_data[i][0], model)
            emissions_per_10 = self.emissions_tracker.stop()
            client.log_metric(self.run_id, "emissions_per_10_predictions", emissions_per_10 * 1000)

       # For keras models assuming the batch size of the data is 1
        elif self.flavor == 'keras':
            self.emissions_tracker.start()
            model.predict(test_data.take(10))
            emissions_per_10 = self.emissions_tracker.stop()
            client.log_metric(self.run_id, "emissions_per_10_predictions", emissions_per_10 * 1000)

    
    

        


        
       
        
        
            

