from mlflow import MlflowClient
from codecarbon import EmissionsTracker


import mlflow
import mlflow.keras
import torch


class EmissionsTrackerMlflow:
    experiment_tracking_params=None
    tracking_uri=""
    run_id=""
    exp_id=""
    flavor=""
    emissions=0
    emissions_tracker = None
    device = None
    model_acc = None
    
    def __init__(self):
        self.emissions_tracker = EmissionsTracker()
    
    def read_params(self, experiment_tracking_params):
        self.experiment_tracking_params = experiment_tracking_params
        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        mlflow.set_tracking_uri(self.experiment_tracking_params['tracking_uri'])
        exp_id = dict(mlflow.set_experiment(self.experiment_tracking_params['experiment_name']))['experiment_id']
        self.exp_id = exp_id
        self.flavor = experiment_tracking_params['flavor']
        run_id = dict(client.create_run(exp_id, run_name=self.experiment_tracking_params['run_name'] ))
        self.run_id = dict(run_id['info'])['run_id']
        mlflow.start_run(self.run_id, exp_id)
        if self.flavor == 'keras':
            mlflow.keras.autolog()
    
    

    def start_training_job(self):
        #client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        #expm = client.get_experiment_by_name(self.experiment_tracking_params['experiment_name'])
        #if expm:
        #    expm = dict(expm)
        #    exp_id = expm['experiment_id']
        #else:
        #    exp_id = client.create_experiment(self.experiment_tracking_params['experiment_name'])
        #run_id = dict(client.create_run(exp_id, run_name=self.experiment_tracking_params['run_name']))
        
        #print("runid", dict(run_id['info'])['run_id'])
        #self.run_id = dict(run_id['info'])['run_id']
        #mlflow.set_tracking_uri(self.experiment_tracking_params['tracking_uri'])
        #mlflow.set_experiment(self.experiment_tracking_params['experiment_name'])
        self.emissions_tracker.start()
        
    def get_default_device(self):
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
    
    def predict_image(self, img, model):
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
        
    def evaluate_model_accuracy(self, model, test_data):
        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        if self.flavor == 'pytorch':
            total_data_count = len(test_data)
            correctly_predicted = 0
            for testd, res in test_data:
                predicted_val = self.predict_image(testd, model)
                if predicted_val == res:
                    correctly_predicted += 1
            model_acc = correctly_predicted / total_data_count
            self.model_acc = model_acc
            client.log_metric(self.run_id,"Model Accuracy", model_acc * 100)
            return model_acc
        elif self.flavor == 'keras':
            accuracy = model.evaluate(test_data)[1]
            self.model_acc = accuracy
            client.log_metric(self.run_id,"Model Accuracy", accuracy * 100)
            return accuracy
        else:
            raise "Not supported"


    def accuracy_per_emission(self, model, test_data):
        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        if self.model_acc == None:
            self.evaluate_model_accuracy(model, test_data)
        acc_per_emission = (self.model_acc * 100) / (self.emissions * 1000)
        client.log_metric(self.run_id, "accuracy_per_emission", acc_per_emission)

    def emission_per_10_inferences(self, model, test_data):
        client = MlflowClient(tracking_uri=self.experiment_tracking_params['tracking_uri'])
        if self.flavor == 'pytorch':
            self.emissions_tracker.start()
            for i in range(10):
                self.predict_image(test_data[i][0], model)
            emissions_per_10 = self.emissions_tracker.stop()
            client.log_metric(self.run_id, "emissions_per_10_predictions", emissions_per_10 * 1000)
        elif self.flavor == 'keras':
            self.emissions_tracker.start()

            model.predict(test_data.take(10))

            emissions_per_10 = self.emissions_tracker.stop()
            client.log_metric(self.run_id, "emissions_per_10_predictions", emissions_per_10 * 1000)

    
    

        


        
       
        
        
            

