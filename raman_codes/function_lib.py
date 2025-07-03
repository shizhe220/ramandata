import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import matplotlib
from scipy.optimize import curve_fit
from scipy.special import wofz
import os
import random

"""
def plot_2d_raman(coordinates, raman_amplitudes, wave_numbers):
    N_w = len(wave_numbers)
    data_grid = np.reshape(raman_amplitudes,(21,21,N_w))
    x_grid = np.reshape(coordinates[:, 1], (21, 21))
    y_grid = np.reshape(coordinates[:, 0], (21, 21))

    # Create a figure with two subplots


    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])

    ax_map = fig.add_subplot(gs[:, 0])
    ax_spectra = fig.add_subplot(gs[0, 1])
    # Adjust subplot spacing
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.15)



    # Plot the initial map on the first subplot
    scatter = ax_map.pcolormesh(x_grid, y_grid, data_grid[:, :, 0], cmap='viridis', vmax=250)

    # Add colorbar to the map subplot
    cbar = fig.colorbar(scatter, ax=ax_map, pad=0.02, aspect=20, orientation='vertical')
    cbar.set_label(r'Intensity (counts)', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    # Set labels and title for the map subplot
    ax_map.set_xlabel(r'X ($\mu m$)', fontsize=16)
    ax_map.set_ylabel(r'Y ($\mu m$)', fontsize=16)
    ax_map.set_title(r'Raman Map', fontsize=18, weight='bold')

    # Customize grid and axis
    ax_map.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_map.invert_yaxis()

    # Adjust the     position to fit in the combined figure
    ax_slider = plt.axes([0.25, 0.01, 0.5, 0.03], facecolor='lightgrey')
    slider = Slider(ax_slider, '', 0, N_w - 1, valinit=0, valstep=1)

    # Adjust text box positions to fit in the combined figure
    ax_text = plt.axes([0.45, 0.92, 0.1, 0.03])
    ax_text.set_frame_on(False)
    ax_text.axis('off')
    text = ax_text.text(0.5, 0.5, f'Wavenumber: {wave_numbers[0]:.2f} cm$^{-1}$', ha='center', va='center', fontsize=12)

    ax_text2 = plt.axes([0.12, 0.92, 0.1, 0.03])
    ax_text2.set_frame_on(False)
    ax_text2.axis('off')
    text2 = ax_text2.text(0.5, 0.5, f'Intensity: {data_grid[0, 0, 0]:.2f}', ha='center', va='center', fontsize=12)

    # Create a reset button
    ax_button = plt.axes([0.88, 0.02, 0.08, 0.04])  
    reset_button = Button(ax_button, 'Clear', color='lightgrey', hovercolor='0.975')
    slider.valtext.set_visible(False)

    clicked_indices = []
    map_markers = []
    color_list = list(mcolors.TABLEAU_COLORS)

    def update(val):
        wn = int(slider.val)
        scatter.set_array(data_grid[:, :, wn].ravel())
        cbar.update_normal(scatter)
        text.set_text(f'Wavenumber: {wave_numbers[wn]:.2f} cm$^{-1}$')
        fig.canvas.draw_idle()

    def on_hover(event):
        if event.inaxes != ax_map:
            return
        x_hover, y_hover = event.xdata, event.ydata
        distances = np.sqrt((coordinates[:, 1] - x_hover)**2 + (coordinates[:, 0] - y_hover)**2)
        closest_idx = np.argmin(distances)
        wn = int(slider.val)
        raman_shift_value = raman_amplitudes[closest_idx, wn]
        text2.set_text(f'Intensity: {raman_shift_value:.2f}')
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax_map:
            return
        if not event.dblclick:
            return
        x_click, y_click = event.xdata, event.ydata
        distances = np.sqrt((coordinates[:, 1] - x_click)**2 + (coordinates[:, 0] - y_click)**2)
        closest_idx = np.argmin(distances)
        if closest_idx in clicked_indices:
            print(f"Point at index {closest_idx} has already been clicked.")
            return
        if len(clicked_indices) >= len(color_list):
            print("Maximum number of points reached.")
            return
        clicked_indices.append(closest_idx)
        color = list(color_list)[len(clicked_indices)-1]
        # Plot cross marker on the map
        marker, = ax_map.plot(coordinates[closest_idx, 1], coordinates[closest_idx, 0], marker='x', color=color, markersize=10, markeredgewidth=2)
        map_markers.append(marker)
        # Plot Raman spectrum on the spectra subplot
        spectrum = raman_amplitudes[closest_idx, :]
        label = f'X={coordinates[closest_idx,1]:.2f}, Y={coordinates[closest_idx,0]:.2f}'
        ax_spectra.plot(wave_numbers, spectrum, color=color, label=label)
        # ax_spectra.legend(loc='upper right', fontsize=10)
        fig.canvas.draw_idle()
    def reset(event):
        global clicked_indices, map_markers
        clicked_indices = []
        # Remove markers from the map
        for marker in map_markers:
            marker.remove()
        map_markers = []
        # Clear the spectra plot
        ax_spectra.cla()
        # Reset the spectra plot labels and grid
        ax_spectra.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
        ax_spectra.set_ylabel('Intensity (counts)', fontsize=14)
        ax_spectra.set_title('Raman Spectra at Selected Points', fontsize=16)
        ax_spectra.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        fig.canvas.draw_idle()


    reset_button.on_clicked(reset)
    slider.on_changed(update)
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Set labels and title for the spectra subplot
    ax_spectra.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
    ax_spectra.set_ylabel('Intensity (counts)', fontsize=14)
    # ax_spectra.set_title('Raman Spectra at Selected Points', fontsize=16)
    ax_spectra.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.show()
"""

"""
def select_points(coordinates, raman_amplitudes, wave_numbers, file_name):
    N_w = len(wave_numbers)
    data_grid = np.reshape(raman_amplitudes,(21,21,N_w))
    x_grid = np.reshape(coordinates[:, 1], (21, 21))
    y_grid = np.reshape(coordinates[:, 0], (21, 21))

    # Create the figure and slider layout
    fig, ax_map = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)

    # Plot the initial 2D map
    scatter = ax_map.pcolormesh(x_grid, y_grid, data_grid[:, :, 0], cmap='viridis', vmax=250)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax_map, pad=0.02, aspect=20, orientation='vertical')
    cbar.set_label(r'Intensity (counts)', fontsize=14, labelpad=10)

    # Set labels and title
    ax_map.set_xlabel(r'X ($\mu m$)', fontsize=14)
    ax_map.set_ylabel(r'Y ($\mu m$)', fontsize=14)
    ax_map.set_title(r'Raman Map', fontsize=16, weight='bold')
    ax_map.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_map.invert_yaxis()

    # Add slider for selecting wavenumber
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgrey')
    slider = Slider(ax_slider, 'Frame', 0, N_w - 1, valinit=0, valstep=1)

    # List to store selected points
    selected_points = []

    # Update function for the slider
    def update(val):
        frame = int(slider.val)
        scatter.set_array(data_grid[:, :, frame].ravel())
        cbar.update_normal(scatter)
        fig.canvas.draw_idle()

    # On-click event to select points
    def on_click(event):
        if event.inaxes != ax_map:
            return
        x_click, y_click = event.xdata, event.ydata
        distances = np.sqrt((coordinates[:, 1] - x_click)**2 + (coordinates[:, 0] - y_click)**2)
        closest_idx = np.argmin(distances)
        selected_points.append((coordinates[closest_idx, 1], coordinates[closest_idx, 0]))
        ax_map.plot(coordinates[closest_idx, 1], coordinates[closest_idx, 0], 'x', color='red', markersize=10)
        fig.canvas.draw_idle()

    # Save selected points to a text file
    def save_points(event):
        with open(file_name + ".txt", "w") as f:
            for point in selected_points:
                f.write(f"{point[0]:.2f}, {point[1]:.2f}\n")
        print("Selected points saved to '", file_name+".txt'.")

    # Add a save button
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.04])
    save_button = Button(ax_button, 'Save', color='lightgrey', hovercolor='0.975')
    save_button.on_clicked(save_points)

    # Connect the slider and click event
    slider.on_changed(update)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show(block=True)
"""

def lorentzian(x, A, x0, gamma, B):
    """
    Lorentzian peak function.
    A     - peak amplitude
    x0    - center of the peak
    gamma - half-width at half-maximum (HWHM)
    B     - baseline offset
    """
    return A / (1 + ((x - x0)/gamma)**2) + B

def lorentzian2(x, A, x0, gamma, B, m):
    """
    Lorentzian peak function.
    A     - peak amplitude
    x0    - center of the peak
    gamma - half-width at half-maximum (HWHM)
    B     - baseline offset
    """
    return A / (1 + ((x - x0)/gamma)**2) + B + m*x

def double_lorentzian(x, 
                      A1, x01, gamma1, 
                      A2, x02, gamma2, 
                      B):
    """
    Double Lorentzian peak function with a shared baseline.

    Parameters
    ----------
    x : array-like
        Independent variable (e.g., wavenumber).
    A1 : float
        Amplitude of the first Lorentzian peak.
    x01 : float
        Center of the first Lorentzian peak.
    gamma1 : float
        Half-width at half-maximum (HWHM) of the first peak.
    A2 : float
        Amplitude of the second Lorentzian peak.
    x02 : float
        Center of the second Lorentzian peak.
    gamma2 : float
        Half-width at half-maximum (HWHM) of the second peak.
    B : float
        Constant baseline offset for the entire function.

    Returns
    -------
    array-like
        The summed intensity of the two Lorentzian peaks plus baseline.
    """
    # First Lorentzian
    lorentz1 = A1 / (1 + ((x - x01) / gamma1) ** 2)
    # Second Lorentzian
    lorentz2 = A2 / (1 + ((x - x02) / gamma2) ** 2)
    
    return lorentz1 + lorentz2 + B

def voigt(x, A, x0, sigma, gamma, B):
    """
    Computes the Voigt profile at each value of x.

    Parameters
    ----------
    x : array-like
        Independent variable (e.g., wavenumber).
    A : float
        Amplitude (scaling of the peak).
    x0 : float
        Center of the peak.
    sigma : float
        Standard deviation of the Gaussian part.
    gamma : float
        Half-width at half-maximum (HWHM) of the Lorentzian part.
    B : float
        Constant baseline offset.

    Returns
    -------
    array-like
        Voigt lineshape evaluated at each point in x.
    """
    # Convert x to a numpy array if it isn't already
    x = np.array(x, dtype=float)

    # Define the dimensionless complex variable
    z = (x - x0 + 1j*gamma) / (sigma * np.sqrt(2))

    # Voigt profile is the real part of wofz(z), scaled appropriately
    # wofz(z) is exp(-z^2) * erfc(-i z * sqrt(pi)) 
    # We normalize by (sigma*sqrt(2*pi)) so that A roughly corresponds to peak height.
    voigt_profile = A * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

    return voigt_profile + B

def wn_indices(neighbourhood, wave_numbers):
    """
    Function to obtain the wave number indices of the starting and end points of a neighbourhood range
    """
    ind_neigh = [np.argmin(np.abs(wave_numbers - neighbourhood[0])), np.argmin(np.abs(wave_numbers - neighbourhood[1]))]
    return ind_neigh


    """
    returns the lorentzian fit parameters for all the points in the data grid. neighbourhood defines the range
    parameters in order is defined in the lorentzian function
    """
    ind_neigh = wn_indices(neighbourhood, wave_numbers)#getting the indices of the neighbourhood
    sliced_arrays = data_grid[:,:,ind_neigh[0]:ind_neigh[1]]#slicing the data matrix 
    sliced_wn = wave_numbers[ind_neigh[0]:ind_neigh[1]]
    #initial guesses
    initial_guesses = [10, np.average(neighbourhood), 50, 10]
    n,m,l = np.shape(sliced_arrays)
    parameters = np.zeros((n,m,4))
    for i in range(n):
        for j in range(m):
            popt, pcov = curve_fit(lorentzian, sliced_wn, sliced_arrays[i,j,:], p0=initial_guesses)
            parameters[i,j] = popt
    return parameters

def parameter_matrix(fit_function, data_grid, wave_numbers, neighbourhood, initial_guesses):
    """
    returns the lorentzian fit parameters for all the points in the data grid. neighbourhood defines the range
    parameters in order is defined in the lorentzian function
    """
    ind_neigh = wn_indices(neighbourhood, wave_numbers)#getting the indices of the neighbourhood
    sliced_arrays = data_grid[:,:,ind_neigh[0]:ind_neigh[1]]#slicing the data matrix 
    sliced_wn = wave_numbers[ind_neigh[0]:ind_neigh[1]]
    #initial guesses
    n,m,l = np.shape(sliced_arrays)
    num_params = len(initial_guesses)
    parameters = np.zeros((n,m,num_params))
    for i in range(n):
        for j in range(m):
            try:
                # Perform the curve fitting
                popt, pcov = curve_fit(fit_function, sliced_wn, sliced_arrays[i, j, :], p0=initial_guesses, method = 'trf')
                parameters[i, j] = popt
            except RuntimeError:
                # Handle fitting failures gracefully (e.g., set to NaN or default values)
                parameters[i, j] = np.full(num_params, np.nan)
    
    return parameters

def plot_parameters(coordinates,  peaks):
    x_grid = np.reshape(coordinates[:, 1], (21, 21))
    y_grid = np.reshape(coordinates[:, 0], (21, 21))
    fig, ax_peak = plt.subplots(figsize=(12, 10))
    scatter_peak = ax_peak.pcolormesh(x_grid, y_grid, peaks, cmap = 'coolwarm')
    cbar = fig.colorbar(scatter_peak, ax=ax_peak, pad=0.02, aspect=20, orientation='vertical')
    cbar.set_label(r'Wavenumber', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    ax_peak.set_xlabel(r'X ($\mu m$)', fontsize=16)
    ax_peak.set_ylabel(r'Y ($\mu m$)', fontsize=16)
    ax_peak.set_title(r'Peak Position', fontsize=18, weight='bold')
    ax_peak.invert_yaxis()

    ax_text2 = plt.axes([0.066, 0.955, 0.1, 0.03])
    ax_text2.set_frame_on(False)
    ax_text2.axis('off')
    text2 = ax_text2.text(0.5, 0.5, f'Peak: {0}', ha='center', va='center', fontsize=12)

    def on_hover(event):
        if event.inaxes != ax_peak:
            return
        x_hover, y_hover = event.xdata, event.ydata
        distances = (x_grid-x_hover)**2 + (y_grid - y_hover)**2
        closest_idx =np.unravel_index(distances.argmin(), distances.shape)
        peak_value = peaks[closest_idx]
        text2.set_text(f'Peak: {peak_value:.2f}')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()

def plot_fit(fit_function, fit_parameters, wave_numbers, data_grid,neighbourhood, pos):
    ind_neigh = wn_indices(neighbourhood, wave_numbers)
    params = fit_parameters[pos[0],pos[1]]
    x_data = wave_numbers[ind_neigh[0]:ind_neigh[1]]
    y_fit = fit_function(x_data, *params)
    y_data = data_grid[pos[0],pos[1],ind_neigh[0]:ind_neigh[1]]

    plt.figure()
   
    plt.plot(x_data, y_data, c = 'C0',label = 'data')
    plt.plot(x_data, y_fit, c='red', label = 'fit')
    plt.ylabel('Intensity')
    plt.xlabel(r'Wavenumber cm$^{-1}$')
    plt.legend()
    plt.show()

def select_points(coordinates, raman_amplitudes, wave_numbers, file_name):
    """
    Allows users to:
      1) Slide through frames (wavenumbers) in a 3D data grid.
      2) Double-click on the 2D map to select points, and mark them with an 'x'.
      3) Save selected points to a text file.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, 2) array with [y, x] for each point. (21*21 = 441 total points)
    raman_amplitudes : np.ndarray
        (N, N_w) array with spectral data for each coordinate.
    wave_numbers : np.ndarray
        (N_w,) array of wavenumber values.
    file_name : str
        Name of the file (without extension) to save the selected coordinates.
    """

    N_w = len(wave_numbers)
    # Reshape raman_amplitudes to 21 x 21 x N_w for 2D map + spectral dimension
    data_grid = np.reshape(raman_amplitudes, (21, 21, N_w))

    # Reshape coordinates into 21 x 21 for the 2D map
    # Note: If your coordinates are in (y, x) format, use them accordingly.
    x_grid = np.reshape(coordinates[:, 1], (21, 21))
    y_grid = np.reshape(coordinates[:, 0], (21, 21))

    # Create the figure and axes
    fig, ax_map = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)

    # Plot the initial 2D map for the first frame
    scatter = ax_map.pcolormesh(
        x_grid, y_grid, data_grid[:, :, 0],
        cmap='viridis', vmax=250
    )

    # Add colorbar
    cbar = fig.colorbar(
        scatter, ax=ax_map, pad=0.02, aspect=20, orientation='vertical'
    )
    cbar.set_label(r'Intensity (counts)', fontsize=14, labelpad=10)

    # Set labels and title
    ax_map.set_xlabel(r'X ($\mu m$)', fontsize=14)
    ax_map.set_ylabel(r'Y ($\mu m$)', fontsize=14)
    ax_map.set_title(r'Raman Map', fontsize=16, weight='bold')
    ax_map.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_map.invert_yaxis()

    # Add slider for selecting wavenumber
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgrey')
    slider = Slider(ax_slider, 'Frame', 0, N_w - 1, valinit=0, valstep=1)

    # List to store selected points
    selected_points = []

    def update(val):
        """Update function for the slider."""
        frame = int(slider.val)
        # Flatten the 21x21 array for the pcolormesh
        scatter.set_array(data_grid[:, :, frame].ravel())
        cbar.update_normal(scatter)
        # Force redraw
        fig.canvas.draw_idle()

    def on_click(event):
        """On-click event to select points."""
        if event.inaxes != ax_map:
            return

        # Retrieve the clicked x,y in data coordinates
        x_click, y_click = event.xdata, event.ydata
        # Find the closest point in 'coordinates' to the click
        distances = np.sqrt(
            (coordinates[:, 1] - x_click) ** 2 +
            (coordinates[:, 0] - y_click) ** 2
        )
        closest_idx = np.argmin(distances)

        # Save the selected point in (x, y) format
        selected_points.append((coordinates[closest_idx, 1],
                                coordinates[closest_idx, 0]))

        # Plot a red 'x' at the selected point
        ax_map.plot(
            coordinates[closest_idx, 1],
            coordinates[closest_idx, 0],
            'x', color='red', markersize=10
        )
        fig.canvas.draw_idle()

    def save_points(event):
        """Save selected points to a text file."""
        print("\nSave button clicked!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Points to save: {selected_points}")

        # Save each point as "x, y" on a new line
        with open(file_name + ".txt", "w") as f:
            for point in selected_points:
                f.write(f"{point[0]:.2f}, {point[1]:.2f}\n")

        print(f"Selected points saved to '{file_name}.txt'")

    # Add a 'Save' button
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.04])
    save_button = Button(ax_button, 'Save', color='lightgrey', hovercolor='0.975')
    save_button.on_clicked(save_points)

    # Connect the slider and click event
    slider.on_changed(update)
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Show the interactive plot
    plt.show(block=True)

def plot_2d_raman(coordinates, raman_amplitudes, wave_numbers):
    """
    Displays a 2D Raman map with the ability to:
    1) Slide through frames (different wavenumbers),
    2) Double-click on points to add them to the spectrum subplot,
    3) Hover over the map to see updated intensity in a text box,
    4) Clear all clicked points and their spectra.
    
    Color cycling is used when you run out of colors in TABLEAU_COLORS.
    """

    N_w = len(wave_numbers)
    data_grid = np.reshape(raman_amplitudes, (21, 21, N_w))
    x_grid = np.reshape(coordinates[:, 1], (21, 21))
    y_grid = np.reshape(coordinates[:, 0], (21, 21))

    # Create the figure & gridspec layout
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])
    
    # Main map axis (spans both rows in the left column)
    ax_map = fig.add_subplot(gs[:, 0])
    # Spectra axis (top row in the right column)
    ax_spectra = fig.add_subplot(gs[0, 1])
    
    # Adjust spacing between subplots
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.15)

    # Plot the initial 2D map for the first wavenumber
    scatter = ax_map.pcolormesh(
        x_grid, y_grid, data_grid[:, :, 0],
        cmap='viridis', vmax=250
    )

    # Add colorbar to the map
    cbar = fig.colorbar(scatter, ax=ax_map, pad=0.02, aspect=20, orientation='vertical')
    cbar.set_label(r'Intensity (counts)', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    # Set labels and title for the map
    ax_map.set_xlabel(r'X ($\mu m$)', fontsize=16)
    ax_map.set_ylabel(r'Y ($\mu m$)', fontsize=16)
    ax_map.set_title(r'Raman Map', fontsize=18, weight='bold')
    ax_map.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_map.invert_yaxis()

    # Slider for wavenumber index
    ax_slider = plt.axes([0.25, 0.01, 0.5, 0.03], facecolor='lightgrey')
    slider = Slider(ax_slider, '', 0, N_w - 1, valinit=0, valstep=1)
    
    # Move the default valtext off-screen (we'll handle text ourselves)
    slider.valtext.set_visible(False)

    # Text box showing current wavenumber
    ax_text = plt.axes([0.45, 0.92, 0.1, 0.03])
    ax_text.set_frame_on(False)
    ax_text.axis('off')
    text = ax_text.text(
        0.5, 0.5,
        f'Wavenumber: {wave_numbers[0]:.2f} cm$^{-1}$',
        ha='center', va='center', fontsize=12
    )

    # Text box showing current intensity (on hover)
    ax_text2 = plt.axes([0.12, 0.92, 0.1, 0.03])
    ax_text2.set_frame_on(False)
    ax_text2.axis('off')
    text2 = ax_text2.text(
        0.5, 0.5,
        f'Intensity: {data_grid[0, 0, 0]:.2f}',
        ha='center', va='center', fontsize=12
    )

    # 'Clear' button
    ax_button = plt.axes([0.88, 0.02, 0.08, 0.04])
    reset_button = Button(ax_button, 'Clear', color='lightgrey', hovercolor='0.975')

    # Store clicked point indices & their map markers
    clicked_indices = []
    map_markers = []

    # Convert the color dictionary to a list for cycling
    color_list = list(mcolors.TABLEAU_COLORS)

    def update(val):
        """Update the map when the slider changes (wavenumber index)."""
        wn_index = int(slider.val)
        scatter.set_array(data_grid[:, :, wn_index].ravel())
        cbar.update_normal(scatter)
        text.set_text(f'Wavenumber: {wave_numbers[wn_index]:.2f} cm$^{-1}$')
        fig.canvas.draw_idle()

    def on_hover(event):
        """Update the intensity readout when hovering over the map."""
        if event.inaxes != ax_map:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        x_hover, y_hover = event.xdata, event.ydata
        distances = np.sqrt((coordinates[:, 1] - x_hover)**2 + (coordinates[:, 0] - y_hover)**2)
        closest_idx = np.argmin(distances)
        wn_index = int(slider.val)
        raman_shift_value = raman_amplitudes[closest_idx, wn_index]
        text2.set_text(f'Intensity: {raman_shift_value:.2f}')
        fig.canvas.draw_idle()

    def on_click(event):
        """On double-click, add a marker on the map and plot the spectrum."""
        if event.inaxes != ax_map:
            return
        if not event.dblclick:
            return
        
        x_click, y_click = event.xdata, event.ydata
        distances = np.sqrt((coordinates[:, 1] - x_click)**2 + (coordinates[:, 0] - y_click)**2)
        closest_idx = np.argmin(distances)
        
        # Check if we already clicked this exact index
        if closest_idx in clicked_indices:
            print(f"Point at index {closest_idx} has already been clicked.")
            return
        
        # Append new point
        clicked_indices.append(closest_idx)
        
        # Cycle the color by using modulo (no max point limit)
        color = color_list[len(clicked_indices) % len(color_list)]
        
        # Plot a cross marker on the map
        marker, = ax_map.plot(
            coordinates[closest_idx, 1],
            coordinates[closest_idx, 0],
            marker='x', color=color, markersize=10, markeredgewidth=2
        )
        map_markers.append(marker)
        
        # Plot the corresponding Raman spectrum on the spectra subplot
        spectrum = raman_amplitudes[closest_idx, :]
        label = f'X={coordinates[closest_idx,1]:.2f}, Y={coordinates[closest_idx,0]:.2f}'
        ax_spectra.plot(wave_numbers, spectrum, color=color, label=label)
        
        fig.canvas.draw_idle()

    def reset(event):
        """Clear all clicked points and the spectra subplot."""
        clicked_indices.clear()    # Empties the list in place
        
        # Remove each marker from the map
        for marker in map_markers:
            marker.remove()
        map_markers.clear()
        
        # Clear the spectra axis
        ax_spectra.cla()
        ax_spectra.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
        ax_spectra.set_ylabel('Intensity (counts)', fontsize=14)
        ax_spectra.set_title('Raman Spectra at Selected Points', fontsize=16)
        ax_spectra.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        fig.canvas.draw_idle()

    # Connect the callbacks
    reset_button.on_clicked(reset)
    slider.on_changed(update)
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Set up the spectra subplot
    ax_spectra.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
    ax_spectra.set_ylabel('Intensity (counts)', fontsize=14)
    ax_spectra.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.show()

# def param_list(file_name, neighbourhood)
def get_peak_params(fit_function, raman_amplitudes, coordinates,  wave_numbers, neighbourhood, initial_guesses, file_name):
    N_w = len(wave_numbers)
    data_grid = np.reshape(raman_amplitudes, (21, 21, N_w))
    x_grid = np.reshape(coordinates[:, 1], (21, 21))
    y_grid = np.reshape(coordinates[:, 0], (21, 21))

    p_matrix = parameter_matrix(fit_function, data_grid, wave_numbers, neighbourhood, initial_guesses)
    points_list = np.loadtxt(file_name, delimiter=",")
    grid_indices = []
    for point in points_list:
        x_point, y_point = point
        diff_x = (x_grid - x_point) ** 2
        diff_y = (y_grid - y_point) ** 2
        distances = diff_x + diff_y
        n, m = np.unravel_index(np.argmin(distances), x_grid.shape)
        grid_indices.append([n , m ])  #stores x and y indices of the selected points
    grid_indices = np.array(grid_indices)
    peak_params = []
    for c in grid_indices:
        c = tuple(c)
        peak_params.append(p_matrix[c[0],c[1]])
        
    return np.array(peak_params)
    

