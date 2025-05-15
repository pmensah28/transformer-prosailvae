import numpy as np
import rasterio as rio
from sensorsio import sentinel2
import socket
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import inspect

def var_of_product(var_1, var_2, mean_1, mean_2):
    return (var_1 + mean_1.pow(2)) * (var_2 + mean_2.pow(2)) - (mean_1 * mean_2).pow(2)


def std_interpolate(x0, std0, x1, std1, x):
    assert (x <= x1).all() and (x >= x0).all()
    u0 = (x1-x) / (x1-x0)
    u1 = (x-x0) / (x1-x0)
    return np.sqrt((u0 * std0)**2 + (u1*std1)**2)

def interpolate(x0, y0, x1, y1, x):
    assert (x <= x1).all() and (x >= x0).all()
    u0 = (x1-x) / (x1-x0)
    u1 = (x-x0) / (x1-x0)
    return u0 * y0 + u1 * y1 

def simple_interpolate(y_after, y_before, dt_after, dt_before, is_std=False):
    res = np.zeros_like(y_after).astype(float)
    res[dt_before==0] = y_before[dt_before==0]
    res[dt_after==0] = y_after[dt_after==0]
    idx = np.logical_and(dt_after!=0, dt_before!=0)
    dt = np.abs(dt_after[idx]) + np.abs(dt_before[idx])
    v = np.abs(dt_after[idx]) / dt
    u = np.abs(dt_before[idx])  / dt
    # if is_std:
    #     res[idx] = std_interpolate(-dt_before[idx], y_before[idx], -dt_after[idx], y_after[idx], np.zeros_like(dt_before[idx]))
    # else:    
    res[idx] = interpolate(-dt_before[idx], y_before[idx], -dt_after[idx], y_after[idx], np.zeros_like(dt_before[idx]))
    return res

def get_bb_array_index(bb, image_bb, res=10):
    xmin = (bb[0] - image_bb[0]) / res
    ymin = ( - (bb[3] - image_bb[3])) / res
    xmax = xmin + (bb[2] - bb[0]) / res
    ymax = ymin + (bb[3] - bb[1]) / res
    return int(xmin), int(ymin), int(xmax), int(ymax)

def read_data_from_theia(left, bottom, right, top, src_epsg, path_to_theia_product, margin=100):
    dataset = sentinel2.Sentinel2(path_to_theia_product)
    left, bottom, right, top = rio.warp.transform_bounds(src_epsg, dataset.crs.to_epsg(), left, bottom, right, top)
    bb = rio.coords.BoundingBox(left - margin, bottom - margin, right + margin, top + margin)

    bands = [sentinel2.Sentinel2.B2,
             sentinel2.Sentinel2.B3,
             sentinel2.Sentinel2.B4,
             sentinel2.Sentinel2.B5,
             sentinel2.Sentinel2.B6,
             sentinel2.Sentinel2.B7,
             sentinel2.Sentinel2.B8,
             sentinel2.Sentinel2.B8A,
             sentinel2.Sentinel2.B11,
             sentinel2.Sentinel2.B12]

    xmin, ymin, xmax, ymax = get_bb_array_index(bb, dataset.bounds, res=10)

    if socket.gethostname()=='CELL200973':
        even_zen, odd_zen = dataset.read_zenith_angles_as_numpy()
        even_zen = even_zen[ymin:ymax, xmin:xmax]
        odd_zen = odd_zen[ymin:ymax, xmin:xmax]
        joint_zen = np.array(even_zen)
        joint_zen[np.isnan(even_zen)] = odd_zen[np.isnan(even_zen)]
        del even_zen
        del odd_zen

        even_az, odd_az = dataset.read_azimuth_angles_as_numpy()
        even_az = even_az[ymin:ymax, xmin:xmax]
        odd_az = odd_az[ymin:ymax, xmin:xmax]
        joint_az = np.array(even_az)
        joint_az[np.isnan(even_az)] = odd_az[np.isnan(even_az)]
        del even_az
        del odd_az

    else:
        even_zen, odd_zen, even_az, odd_az = dataset.read_incidence_angles_as_numpy()
        # Crop the arrays to the right dimensions
        even_zen = even_zen[ymin:ymax, xmin:xmax]
        odd_zen = odd_zen[ymin:ymax, xmin:xmax]
        even_az = even_az[ymin:ymax, xmin:xmax]
        odd_az = odd_az[ymin:ymax, xmin:xmax]
        
        joint_zen = np.array(even_zen)
        joint_zen[np.isnan(even_zen)] = odd_zen[np.isnan(even_zen)]
        del even_zen
        del odd_zen
        joint_az = np.array(even_az)
        joint_az[np.isnan(even_az)] = odd_az[np.isnan(even_az)]
        del even_az
        del odd_az
    sun_zen, sun_az = dataset.read_solar_angles_as_numpy()
    sun_zen = sun_zen[ymin:ymax, xmin:xmax]
    sun_az = sun_az[ymin:ymax, xmin:xmax]
    s2_a = np.stack((sun_zen, joint_zen, sun_az - joint_az), 0)
    # Convert to data attribute if available, otherwise use the array directly
    if hasattr(s2_a, 'data'):
        s2_a = s2_a.data
    
    # Based on sensorsio documentation, we need to handle different return values
    # Using a more flexible approach that catches all returned values
    try:
        # Check if read_atmos parameter is available
        read_as_numpy_params = inspect.signature(dataset.read_as_numpy).parameters
        
        # Call read_as_numpy and store all returned values
        if 'read_atmos' in read_as_numpy_params:
            result = dataset.read_as_numpy(
                bands, bounds=bb, crs=dataset.crs, band_type=dataset.SRE, read_atmos=True
            )
        else:
            result = dataset.read_as_numpy(
                bands, bounds=bb, crs=dataset.crs, band_type=dataset.SRE
            )
        
        # Extract components based on the number of returned values
        # In all versions, s2_r is always the first value and masks is the second
        s2_r = result[0]
        masks = result[1]
        
        # Handle different return signatures from different sensorsio versions
        if len(result) == 5:  # s2_r, masks, xcoords, ycoords, crs
            xcoords = result[2]
            ycoords = result[3]
            crs = result[4]
        elif len(result) == 6:  # s2_r, masks, atm, xcoords, ycoords, crs
            xcoords = result[3]
            ycoords = result[4]
            crs = result[5]
        elif len(result) == 7:  # Some newer versions might have additional returns
            xcoords = result[3]
            ycoords = result[4]
            crs = result[5]
            # Ignore the last value
        else:
            raise ValueError(f"Unexpected return format from read_as_numpy: got {len(result)} values")
            
    except Exception as e:
        print(f"Error reading data from {path_to_theia_product}: {str(e)}")
        raise
    
    # Process the validity mask
    if hasattr(masks, 'data'):
        validity_mask = np.sum(masks.data, axis=0, keepdims=True).astype(bool).astype(int).astype(float)
    else:
        validity_mask = np.sum(masks, axis=0, keepdims=True).astype(bool).astype(int).astype(float)
    
    # Extract data attribute if available
    if hasattr(s2_r, 'data'):
        s2_r = s2_r.data
        
    return s2_r, s2_a, validity_mask, xcoords, ycoords, crs

def plot_regression_with_error(y_pred, y_true, y_pred_std=None, y_true_std=None, 
                              hue=None, ax=None, xlabel='Predicted', ylabel='Measured',
                              add_metrics=True, add_1to1_line=True):
    """
    Plot regression with error bars and metrics
    
    Parameters:
    -----------
    y_pred : array-like
        Predicted values
    y_true : array-like
        True/measured values
    y_pred_std : array-like, optional
        Standard deviation of predicted values
    y_true_std : array-like, optional
        Standard deviation of true/measured values
    hue : array-like, optional
        Categorical variable for color coding
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis
    add_metrics : bool, optional
        Whether to add metrics (R², RMSE, bias) to the plot
    add_1to1_line : bool, optional
        Whether to add 1:1 line to the plot
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Convert inputs to numpy arrays
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    # Create a dataframe for the plot
    df = pd.DataFrame({'Predicted': y_pred, 'Measured': y_true})
    
    # Add hue if provided
    if hue is not None:
        df['Hue'] = hue
        scatter = sns.scatterplot(data=df, x='Predicted', y='Measured', hue='Hue', ax=ax)
    else:
        scatter = sns.scatterplot(data=df, x='Predicted', y='Measured', ax=ax)
    
    # Add error bars if provided
    if y_pred_std is not None and y_true_std is not None:
        # Convert to numpy arrays to ensure proper indexing
        y_pred_std = np.asarray(y_pred_std)
        y_true_std = np.asarray(y_true_std)
        
        for i in range(len(y_pred)):
            ax.errorbar(y_pred[i], y_true[i], 
                       xerr=y_pred_std[i], yerr=y_true_std[i],
                       fmt='none', ecolor='gray', alpha=0.5, zorder=0)
    
    # Add 1:1 line
    if add_1to1_line:
        min_val = min(y_pred.min(), y_true.min())
        max_val = max(y_pred.max(), y_true.max())
        margin = (max_val - min_val) * 0.1
        limits = [min_val - margin, max_val + margin]
        ax.plot(limits, limits, 'k--', alpha=0.5, label='1:1 line')
    
    # Calculate metrics
    if add_metrics:
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)
        r2 = r_value**2
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        bias = np.mean(y_pred - y_true)
        
        # Add regression line
        x_fit = np.linspace(min(y_pred), max(y_pred), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
        
        # Add metrics text
        metrics_text = (f'$R^2$ = {r2:.3f}\n'
                       f'RMSE = {rmse:.3f}\n'
                       f'Bias = {bias:.3f}\n'
                       f'N = {len(y_pred)}')
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left',
               bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    
    # Set labels and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if hue is not None or add_1to1_line:
        ax.legend(title=None if hue is None else "Land Cover")
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Make the plot square
    ax.set_box_aspect(1)
    
    return ax