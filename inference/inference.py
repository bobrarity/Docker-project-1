import torch
import torch.nn as nn
import pandas as pd


class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def run_inference():
    try:
        model = IrisNet()
        model.load_state_dict(torch.load('/app/shared_folder/trained_model.pth'))
    except FileNotFoundError:
        print('Trained model file not found. Please ensure the model has been trained and saved correctly.')
        return
    except Exception as e:
        print(f'An error occurred while loading the model: {e}')
        return

    model.eval()

    try:
        inference_data = pd.read_csv('inference.csv')
    except FileNotFoundError:
        print('Inference data file not found. Please check the file path.')
        return
    except pd.errors.EmptyDataError:
        print('Inference data file is empty.')
        return
    except pd.errors.ParserError:
        print('Error parsing the inference data file.')
        return
    except Exception as e:
        print(f'An error occurred while loading inference data: {e}')
        return

    try:
        X = inference_data.drop('target', axis=1).values
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)

        predictions = pd.DataFrame(predicted.numpy(), columns=['Predictions'])

        print('Saving predictions to CSV...')
        predictions.to_csv('/app/shared_folder/inference_results.csv', index=False)
        print('File saved.')
    except Exception as e:
        print(f'An error occurred during inference or saving results: {e}')


if __name__ == '__main__':
    run_inference()
