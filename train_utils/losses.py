import torch
from torch.nn.functional import conv2d
import numpy as np
import os

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def init_scales(device, data_scale="2_8125"):
    delta_dx = torch.Tensor(
        np.load(os.path.join(ROOT_DIR, f'weights/delta_np_dx_{data_scale}deg.npy'), mmap_mode='r')).to(
        device)
    delta_dy = torch.Tensor(
        np.load(os.path.join(ROOT_DIR, f'weights/delta_np_dy_{data_scale}deg.npy'), mmap_mode='r')).to(
        device)

    correction_dx = torch.Tensor(
        np.load(os.path.join(ROOT_DIR, f'weights/dx_correction_{data_scale}deg.npy'), mmap_mode='r')).to(device)
    correction_dy = torch.Tensor(
        np.load(os.path.join(ROOT_DIR, f'weights/dy_correction_{data_scale}deg.npy'), mmap_mode='r')).to(device)

    meridional_scale = torch.Tensor(
        np.load(os.path.join(ROOT_DIR, f'weights/meridional_scale_{data_scale}deg.npy'), mmap_mode='r')).to(
        device)
    parallel_scale = torch.Tensor(
        np.load(os.path.join(ROOT_DIR, f'weights/parallel_scale_{data_scale}deg.npy'), mmap_mode='r')).to(device)

    weights_y = torch.tensor([[-1], [1]], dtype=torch.float32, device=device) / (delta_dy)
    weights_y = weights_y.unsqueeze(0).unsqueeze(0)

    weights_x = torch.tensor([[-1, 1]], dtype=torch.float32, device=device) / (delta_dx)
    weights_x = weights_x.unsqueeze(0).unsqueeze(0)

    return weights_x, weights_y, parallel_scale, meridional_scale, correction_dx, correction_dy


def calculate_diff_by_convs(data, data_scale="2_8125"):
    (weights_x, weights_y, parallel_scale,
     meridional_scale, correction_dx, correction_dy) = init_scales(data.device, data_scale=data_scale)

    u_wind = data[:, 0, :, :].unsqueeze(1)
    v_wind = data[:, 1, :, :].unsqueeze(1)

    conv_diff_du_dy = conv2d(input=u_wind, weight=weights_y, padding=[1, 0])[:, :, 1:, :]
    conv_diff_du_dy = meridional_scale * conv_diff_du_dy + v_wind * correction_dy
    conv_diff_dv_dx = conv2d(input=v_wind, weight=weights_x, padding=[0, 1])[:, :, :, 1:]
    conv_diff_dv_dx = parallel_scale * conv_diff_dv_dx + u_wind * correction_dx

    return conv_diff_du_dy, conv_diff_dv_dx


def calculate_diff_by_diffs(data, data_scale="2_8125"):
    u_wind = data[:, 0, :, :].unsqueeze(1)
    v_wind = data[:, 1, :, :].unsqueeze(1)

    conv_diff_du_dy = torch.diff(u_wind, dim=2, prepend=u_wind[:, :, 0, :].unsqueeze(2))
    conv_diff_dv_dx = torch.diff(v_wind, dim=3, prepend=v_wind[:, :, :, 0].unsqueeze(-1))

    return conv_diff_du_dy, conv_diff_dv_dx


def calculate_continuity_loss_v2(prediction, data_t_minus_1):
    conv_diff_du_dy, conv_diff_dv_dx = calculate_diff_by_convs(prediction)

    pde_loss = conv_diff_du_dy + conv_diff_dv_dx
    pde_loss = pde_loss[:, :, 1:, 1:]

    return torch.nn.functional.smooth_l1_loss(pde_loss, torch.zeros_like(pde_loss), beta=1.0)


def calculate_pde_loss_v2(prediction, data_t_minus_1):
    conv_diff_du_dy, conv_diff_dv_dx = calculate_diff_by_diffs(prediction)
    zita_prediction = conv_diff_dv_dx - conv_diff_du_dy

    conv_diff_du_dy_t_minus_1, conv_diff_dv_dx_t_minus_1 = calculate_diff_by_diffs(data_t_minus_1)
    zita_t_minus_1 = conv_diff_dv_dx_t_minus_1 - conv_diff_du_dy_t_minus_1

    dzita_dt = (zita_prediction - zita_t_minus_1) / 2

    pinn_loss = dzita_dt + conv_diff_dv_dx * zita_prediction + conv_diff_du_dy * zita_prediction

    return torch.nn.functional.smooth_l1_loss(pinn_loss, torch.zeros_like(pinn_loss), beta=1.0)


def calculate_pde_loss(prediction, data_t_minus_1):
    conv_diff_du_dy, conv_diff_dv_dx = calculate_diff_by_convs(prediction)
    zita_prediction = conv_diff_dv_dx - conv_diff_du_dy

    conv_diff_du_dy_t_minus_1, conv_diff_dv_dx_t_minus_1 = calculate_diff_by_convs(data_t_minus_1)
    zita_t_minus_1 = conv_diff_dv_dx_t_minus_1 - conv_diff_du_dy_t_minus_1

    dzita_dt = (zita_prediction - zita_t_minus_1) / 2

    pinn_loss = dzita_dt + conv_diff_dv_dx * zita_prediction + conv_diff_du_dy * zita_prediction

    return weighted_mae_torch(pinn_loss, torch.zeros_like(pinn_loss))


def calculate_continuity_loss(prediction, data_t_minus_1):
    conv_diff_du_dy, conv_diff_dv_dx = calculate_diff_by_convs(prediction)

    pinn_loss = conv_diff_du_dy + conv_diff_dv_dx
    pinn_loss = pinn_loss[:, :, :63, :127]

    return weighted_mae_torch(pinn_loss, torch.zeros_like(pinn_loss))


def calculate_pde_and_continuity_loss(prediction, data_t_minus_1):
    conv_diff_du_dy, conv_diff_dv_dx = calculate_diff_by_convs(prediction)
    zita_prediction = conv_diff_dv_dx - conv_diff_du_dy

    conv_diff_du_dy_t_minus_1, conv_diff_dv_dx_t_minus_1 = calculate_diff_by_convs(data_t_minus_1)
    zita_t_minus_1 = conv_diff_dv_dx_t_minus_1 - conv_diff_du_dy_t_minus_1

    dzita_dt = (zita_prediction - zita_t_minus_1) / 2

    continuity_loss = conv_diff_du_dy + conv_diff_dv_dx
    continuity_loss = continuity_loss[:, :, :63, :127]

    continuity_loss = weighted_mae_torch(continuity_loss, torch.zeros_like(continuity_loss))

    pde_loss = dzita_dt + conv_diff_dv_dx * zita_prediction + conv_diff_du_dy * zita_prediction

    pde_loss = weighted_mae_torch(pde_loss, torch.zeros_like(pde_loss))

    return continuity_loss + pde_loss


# torch version for rmse comp
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180. / float(num_lat - 1)


def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(torch.pi / 180. * lat(j, num_lat)) / s


def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    # num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(torch.pi / 180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target) ** 2., dim=(-1, -2)))
    return result


def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.sum(torch.mean(result, dim=0))


def weighted_mae_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    # num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(torch.pi / 180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.mean(weight * torch.abs((pred - target)), dim=(-1, -2))
    return result


def weighted_mae_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_mae_torch_channels(pred, target)
    return torch.sum(torch.mean(result, dim=0))
