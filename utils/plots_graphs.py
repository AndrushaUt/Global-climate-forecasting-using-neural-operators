import numpy as np
import matplotlib.pyplot as plt


def plot_results(pred_to_plot, y_test, dataset_std, dataset_mean, idx_hour=0, name_of_model="FuXi"):
    idx_to_plot = np.random.randint(0, y_test.shape[0])

    prediction_u_wind = (pred_to_plot[idx_to_plot][0].squeeze().cpu().detach().numpy()) * dataset_std[0] + dataset_mean[0]
    true_u_wind = (y_test[idx_to_plot][0].squeeze().cpu().detach().numpy()) * dataset_std[0] + dataset_mean[0]

    prediction_v_wind = (pred_to_plot[idx_to_plot][1].squeeze().cpu().detach().numpy()) * dataset_std[1] + dataset_mean[1]
    true_v_wind = (y_test[idx_to_plot][1].squeeze().cpu().detach().numpy()) * dataset_std[1] + dataset_mean[1]

    plt.figure(constrained_layout=True, figsize=(16, 8))
    plt.subplot(231)
    plt.imshow(prediction_u_wind)
    plt.title(f"Prediction U wind by {name_of_model} hour = {idx_hour}")
    plt.colorbar()

    plt.subplot(232)
    plt.imshow(true_u_wind)
    plt.title(f"True U wind, hour = {idx_hour}")
    plt.colorbar()

    plt.subplot(233)
    plt.imshow(np.abs(prediction_u_wind - true_u_wind))
    plt.title(f"Absolute difference, hour = {idx_hour}")
    plt.colorbar()

    plt.subplot(234)
    plt.imshow(prediction_v_wind)
    plt.title(f"Prediction V wind by {name_of_model} hour = {idx_hour}")
    plt.colorbar()

    plt.subplot(235)
    plt.imshow(true_v_wind)
    plt.title(f"True V wind, hour = {idx_hour}")
    plt.colorbar()

    plt.subplot(236)
    plt.imshow(np.abs(prediction_v_wind - true_v_wind))
    plt.title(f"Absolute difference, hour = {idx_hour}")
    plt.colorbar()

    plt.show()
    