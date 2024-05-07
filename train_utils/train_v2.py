import time
import torch
from IPython.display import clear_output

from pde.utils.plots_graphs import plot_results
from pde.utils.logs_utils import save_logs


def calculate_losses(prediction, x_t_minus_1, y_next_hour,
                     regression_function_of_loss, pde_function_of_loss, coeff=1e6):
    loss_pde = coeff * pde_function_of_loss(prediction, x_t_minus_1)
    loss_regression = regression_function_of_loss(prediction, y_next_hour)

    return loss_regression, loss_pde


def step(model, train_data, x_t_minus_1, y_t_plus_1, pde_function_of_loss, regression_function_of_loss, hour=1,
         coeff_of_pde=1e6):
    t_tensor = hour * torch.ones(train_data.shape[0], device=train_data.device).unsqueeze(-1)
    prediction = model(train_data, t_tensor)

    loss_regression, loss_pde = calculate_losses(prediction, x_t_minus_1, y_t_plus_1, regression_function_of_loss,
                                                 pde_function_of_loss, coeff=coeff_of_pde)
    return prediction, loss_regression, loss_pde


def iterate(model, x_from_loader, y_from_loader, pde_function_of_loss, regression_function_of_loss,
            time_prediction=24, coeff_of_pde=1e6):
    
    loss_pde = 0
    loss_regression = 0
    data_for_model = x_from_loader

    for hour in range(time_prediction):
        if hour == 0:
            # prediction for the first hour
            x_t_minus_1, y_t_plus_1 = x_from_loader[:, 0], y_from_loader[:, 2]

        elif hour == 1:
            # prediction for the second hour
            x_t_minus_1, y_t_plus_1 = x_from_loader[:, 1], y_from_loader[:, 3]

        else:
            x_t_minus_1, y_t_plus_1 = y_from_loader[:, hour - 1], y_from_loader[:, hour + 1]

        prediction, loss_regression_per_hour, loss_pde_per_hour = step(model, data_for_model,
                                                                       x_t_minus_1, y_t_plus_1,
                                                                       pde_function_of_loss,
                                                                       regression_function_of_loss,
                                                                       hour=hour, coeff_of_pde=coeff_of_pde)
        
        data_for_model = torch.cat((data_for_model[:, 1].unsqueeze(1), prediction.unsqueeze(1)), dim=1)

        loss_regression = loss_regression + loss_regression_per_hour
        loss_pde = loss_pde + loss_pde_per_hour
        
    return loss_regression, loss_pde


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
            
            loss_regression, loss_pde = iterate(model, x_train, y_train,
                                                pde_function_of_loss,
                                                regression_function_of_loss,
                                                time_prediction=time_prediction, coeff_of_pde=coeff_of_pde)

            train_regression_full += loss_regression.item()
            train_pde_full += loss_pde.item()

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

                loss_regression, loss_pde = iterate(model, x_test, y_test,
                                                    pde_function_of_loss,
                                                    regression_function_of_loss,
                                                    time_prediction=time_prediction, coeff_of_pde=coeff_of_pde)
                
                test_regression_full += loss_regression.item()
                test_pde_full += loss_pde.item()

        test_regression_full = test_regression_full / len(validloader)
        test_pred_list.append(test_regression_full)

        test_pde_full = test_pde_full / len(validloader)
        test_pde_list.append(test_pde_full)

        scheduler.step()

        clear_output()

        if is_plotting_prediction:
            with torch.no_grad():
                time_tensor = 1 * torch.ones(x_test.shape[0], device=x_test.device).unsqueeze(-1)
                prediction_to_plot_first_hour = model(x_test, time_tensor)
                test_data = torch.cat((x_test[:, 1].unsqueeze(1), prediction_to_plot_first_hour.unsqueeze(1)), dim=1)
                time_tensor = 2 * torch.ones(x_test.shape[0], device=x_test.device).unsqueeze(-1)
                prediction_to_plot_second_hour = model(test_data, time_tensor)
                test_data = torch.cat((prediction_to_plot_first_hour.unsqueeze(1), prediction_to_plot_second_hour.unsqueeze(1)), dim=1)
                prediction_previously = prediction_to_plot_second_hour
                for hour in range(2, time_prediction+1):
                    time_tensor = hour * torch.ones(x_test.shape[0], device=x_test.device).unsqueeze(-1)
                    prediction = model(test_data, time_tensor)
                    test_data = torch.cat((prediction_previously.unsqueeze(1), prediction.unsqueeze(1)), dim=1)
                    prediction_previously = prediction
                
            plot_results(prediction_to_plot_first_hour, y_test[:, 2], dataset_std, dataset_mean, idx_hour=1)
            plot_results(prediction_to_plot_first_hour, y_test[:, -1], dataset_std, dataset_mean, idx_hour=time_prediction)

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
