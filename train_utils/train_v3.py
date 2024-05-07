import time
import random
import torch
from IPython.display import clear_output

from pde.utils.plots_graphs import plot_results
from pde.utils.logs_utils import save_logs


def train(device, model, optimizer, scheduler, trainloader, validloader, dataset_std, dataset_mean,
          regression_function_of_loss, pde_function_of_loss, epochs=3,
          time_prediction=12, name_experiment='model_test', coeff_of_pde=1e6, is_plotting_prediction=True):
    
    train_regression_list = []
    train_pde_list = []

    test_pred_list = []
    test_pde_list = []

    grad_list = []

    for epoch_number in range(epochs):
        time_start = time.time()
        train_regression_full = 0
        train_pde_full = 0
        model.train()
        total_grad = 0
        for x_train, y_train in trainloader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()

            hour_for_prediction = random.randint(1, time_prediction)
            y_next_hour = y_train[:, hour_for_prediction+1]

            time_tensor = hour_for_prediction * torch.ones(x_train.shape[0], device=x_train.device).unsqueeze(-1)
            prediction = model(x_train, time_tensor)

            loss_regression = regression_function_of_loss(prediction, y_next_hour)

            if coeff_of_pde != 0:
                loss_pde = coeff_of_pde * pde_function_of_loss(prediction, y_train[:, hour_for_prediction-1])
                train_pde_full += loss_pde.item()
            else:
                loss_pde = 0

            train_regression_full += loss_regression.item()

            loss = loss_regression + loss_pde

            loss.backward()
            for tag, value in model.named_parameters():
                if value.grad is not None:
                    grad = value.grad.norm()
                    total_grad += grad

            optimizer.step()

        grad_list.append(total_grad.cpu().item())
        grad_by_batch = total_grad.cpu().item() / len(trainloader)

        train_regression_full = train_regression_full / len(trainloader)
        train_pde_full = train_pde_full / len(trainloader)

        train_regression_list.append(train_regression_full)
        train_pde_list.append(train_pde_full)

        test_regression_full = 0
        test_pde_full = 0

        model.eval()
        with torch.no_grad():
            for x_test, y_test in validloader:
                x_test = x_test.to(device)
                y_test = y_test.to(device)

                optimizer.zero_grad()

                hour_for_prediction = random.randint(1, time_prediction)
                y_next_hour = y_test[:, hour_for_prediction + 1]

                time_tensor = hour_for_prediction * torch.ones(x_test.shape[0], device=y_test.device).unsqueeze(-1)
                prediction = model(x_test, time_tensor)

                loss_regression = regression_function_of_loss(prediction, y_next_hour)
                if coeff_of_pde != 0:
                    loss_pde = coeff_of_pde * pde_function_of_loss(prediction, y_test[:, hour_for_prediction - 1])
                    test_pde_full += loss_pde.item()
                else:
                    loss_pde = 0
                
                test_regression_full += loss_regression.item()

        test_regression_full = test_regression_full / len(validloader)
        test_pred_list.append(test_regression_full)

        test_pde_full = test_pde_full / len(validloader)
        test_pde_list.append(test_pde_full)

        scheduler.step()

        clear_output()

        if is_plotting_prediction:
            plot_results(prediction, y_next_hour, dataset_std, dataset_mean, idx_hour=hour_for_prediction)

        if epoch_number % 25 == 0:
            torch.save(model.state_dict(), f"best_models_weights/{name_experiment}_{epoch_number}.pth")
            
        save_logs(train_regression_list, test_pred_list, train_pde_list, test_pde_list, grad_list,
                  name_experiment=f'{name_experiment}')

        end_time = time.time()

        print(f"Epoch : {epoch_number}\n",
              f"Time : {round((end_time - time_start), 5)}\n",
              f"Train Pred loss : {round((float(train_regression_full)), 5)}\n",
              f"Eval Pred loss : {round((float(test_regression_full)), 5)}\n",
              f"Train PINN loss : {round((float(train_pde_full)), 5)}\n",
              f"Test PINN loss : {round((float(train_pde_full)), 5)}\n",
              f"Grad : {round((float(total_grad)), 5)}\n",
              f"grad_by_batch : {round((float(grad_by_batch)), 5)}\n",
              )

    return train_regression_list, test_pred_list
