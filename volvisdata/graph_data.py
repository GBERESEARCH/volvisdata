"""
Methods for graphing volatility data

"""
import copy
import os
import warnings
import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from volvisdata.vol_methods import VolMethods
# pylint: disable=invalid-name, consider-using-f-string

class GraphData():
    """
    Methods for graphing volatility data

    """
    @classmethod
    def line_graph(
        cls,
        params: dict,
        tables: dict) -> dict:
        """
        Returns data for plotting a linegraph of each option maturity plotted by strike
        and implied vol

        Parameters
        ----------
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The
            default is 'last'.

        Returns
        -------
        data_dict : Dict
            Dictionary of linegraph data

        """
        # Create a sorted list of the different number of option
        # expiries
        dates = sorted(list(set(tables['imp_vol_data']['Expiry'])))

        # Create a sorted list of the different number of option time
        # to maturity
        tenors = sorted(list(set(tables['imp_vol_data']['TTM'])))

        # Combine these in a dictionary
        tenor_date_dict = dict(zip(dates, tenors))

        opt_dict = {}
        opt_dict['tenors'] = []
        # For each expiry date
        for exp_date, tenor in tenor_date_dict.items():
            # Create a dictionary of strikes, vols & label 
            data = {}
            data['strikes'] = np.array(tables['imp_vol_data'][
                tables['imp_vol_data']['TTM']==tenor]['Strike'])
            data['vols'] = np.array(tables['imp_vol_data'][
                tables['imp_vol_data']['TTM']==tenor][str(
                    params['vols_dict'][str(params['voltype'])])] * 100)
            data['label'] = str(exp_date)+' Expiry'

            # Append this to the array of tenors
            opt_dict['tenors'].append(data)

        opt_dict = cls._create_opt_labels(
            params=params, 
            opt_dict=opt_dict, 
            output='line'
            )
        
        data_dict = {
            'params': params,
            'tables': tables,
            'opt_dict': opt_dict
        }

        return data_dict
        

    @classmethod
    def scatter_3d(
        cls,
        params: dict,
        tables: dict) -> dict:
        """
        Returns data for plotting a 3D scatter plot of each option implied vol against strike and maturity

        Parameters
        ----------
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The
            default is 'last'.
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.

        Returns
        -------
        data_dict : Dict
            Dictionary of 3D Scatter plot data.

        """
        # Create dictionary to store option data
        opt_dict = {}

        # Create figure and axis objects and format
        opt_dict = cls._create_opt_labels(
            params=params, 
            opt_dict=opt_dict, 
            output='mpl'
            )

        # Create copy of data
        tables['data_3D'] = copy.deepcopy(tables['imp_vol_data'])

        # Filter out any zero prices
        tables['data_3D'] = tables['data_3D'][tables['data_3D'][str(
            params['prices_dict'][str(params['voltype'])])] != 0]

        # Specify the 3 axis values
        opt_dict['strikes'] = np.array(tables['data_3D']['Strike'])
        opt_dict['ttms'] = np.array(tables['data_3D']['TTM'] * 365)
        opt_dict['vols'] = np.array(tables['data_3D'][str(params['vols_dict'][str(
            params['voltype'])])] * 100)

        data_dict = {
            'params': params,
            'tables': tables,
            'opt_dict': opt_dict
        }

        return data_dict


    @classmethod
    def surface_3d(
        cls,
        params: dict,
        tables: dict) -> dict:
        """
        Returns data for plotting a 3D surface plot of the implied vol surface against strike and maturity

        Parameters
        ----------
        surfacetype : Str
            The type of 3D surface to display from 'trisurf', 'mesh',
            'spline', 'interactive_mesh' and 'interactive_spline'.
            The default is 'mesh'.
        smoothing : Bool
            Whether to apply polynomial smoothing. The default is False.
        scatter : Bool
            Whether to plot scatter points on 3D mesh grid. The default
            is False.
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The
            default is 'last'.
        order : Int
            Polynomial order used in numpy polyfit function. The
            default is 3.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D
            graphs. The default
            is 100.
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.
        rbffunc : Str
            Radial basis function used in interpolation chosen from
            'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic',
            'quintic', 'thin_plate'. The default is 'thin_plate'
        colorscale : Str
            Colors used in plotly interactive graph. The default is
            'BlueRed'
        opacity : Float
            opacity of 3D interactive graph
        surf : Bool
            Plot surface in interactive graph
        notebook : Bool
            Whether interactive graph is run in Jupyter notebook or IDE.
            The default is False.
        save_image : Bool
            Whether to save a copy of the image as a png file. The default is
            False
        image_folder : Str
            Location to save any images. The default is 'images'
        image_dpi : Int
            Resolution to save images. The default is 50.

        Returns
        -------
        data_dict : Dict
            Dictionary of 3D surface plot data.

        """

        # Suppress mpl user warning about data containing nan values
        warnings.filterwarnings(
            "ignore", category=UserWarning, message='Z contains NaN values. '\
                'This may result in rendering artifacts.')

        opt_dict = {}

        # If smoothing is set to False
        if params['smoothing'] is False:

            # Create copy of implied vol data
            tables['data_3D'] = copy.deepcopy(tables['imp_vol_data'])

            # Filter out any zero prices
            tables['data_3D'] = tables['data_3D'][tables['data_3D'][str(
                params['prices_dict'][str(params['voltype'])])] != 0]

            # Set 'graph vol' to be the specified voltype
            tables['data_3D']['Graph Vol'] = tables['data_3D'][str(
                params['vols_dict'][str(params['voltype'])])]

        # Otherwise, if smoothing is set to True
        else:
            # Apply the smoothing function to the specified voltype
            params, tables = VolMethods.smooth(params=params, tables=tables)

            # Create copy of implied vol data
            tables['data_3D'] = copy.deepcopy(tables['imp_vol_data_smoothed'])

            # Filter out any zero prices
            tables['data_3D'] = tables['data_3D'][tables['data_3D'][str(
                params['prices_dict'][str(params['voltype'])])] != 0]

            # Set 'graph vol' to be the smoothed vol
            tables['data_3D']['Graph Vol'] = tables['data_3D']['Smoothed Vol']

        # Specify the 3 axis values
        params['x'] = tables['data_3D']['Strike']
        params['y'] = tables['data_3D']['TTM'] * 365
        params['z'] = tables['data_3D']['Graph Vol'] * 100
 
        if params['surfacetype'] == 'trisurf':
            opt_dict = cls._trisurf_graph(params=params, opt_dict=opt_dict)

        elif params['surfacetype'] == 'mesh':
            opt_dict = cls._mesh_graph(params=params, opt_dict=opt_dict)

        elif params['surfacetype'] == 'spline':
            opt_dict = cls._spline_graph(params=params, tables=tables, opt_dict=opt_dict)

        elif params['surfacetype'] in [
            'interactive_mesh', 
            'interactive_spline']:
            opt_dict = cls._interactive_graph(params=params, tables=tables, opt_dict=opt_dict)

        else:
            print("Enter a valid surfacetype from 'trisurf', 'mesh', "\
                "'spline', 'interactive_mesh', 'interactive_spline'")

        # Set warnings back to default
        warnings.filterwarnings("default", category=UserWarning)

        data_dict = {
            'params': params,
            'tables': tables,
            'opt_dict': opt_dict
        }

        return data_dict

        
    @classmethod
    def _trisurf_graph(
        cls,               
        params: dict, 
        opt_dict: dict) -> dict:

        # Create figure and axis objects and format
        opt_dict = cls._create_opt_labels(
            params=params, 
            opt_dict=opt_dict, 
            output='mpl'
            )
        
        opt_dict['strikes'] = np.array(params['x'])
        opt_dict['ttms'] = np.array(params['y'])
        opt_dict['vols'] = np.array(params['z'])

        return opt_dict
    

    @classmethod
    def _mesh_graph(
        cls, 
        params: dict, 
        opt_dict: dict) -> dict:

        # Create arrays across x and y-axes of equally spaced points
        # from min to max values
        x1, y1 = np.meshgrid(
            np.linspace(min(params['x']),
                        max(params['x']),
                        int(params['spacegrain'])),
            np.linspace(min(params['y']),
                        max(params['y']),
                        int(params['spacegrain'])))

        # Map the z-axis with the scipy griddata method, applying
        # cubic spline interpolation
        z1 = griddata(np.array([params['x'],params['y']]).T,
                      np.array(params['z']),
                      (x1,y1),
                      method='cubic')

        # Create figure and axis objects and format
        opt_dict = cls._create_opt_labels(
            params=params, 
            opt_dict=opt_dict, 
            output='mpl'
            )

        opt_dict['strikes_array'] = x1
        opt_dict['ttms_array'] = y1
        opt_dict['vol_surface'] = z1
        
        return opt_dict


    @classmethod
    def _spline_graph(
        cls,
        params: dict,
        tables: dict,
        opt_dict: dict) -> dict:

        # Create arrays across x and y-axes of equally spaced points
        # from min to max values
        x1 = np.linspace(min(params['x']),
                         max(params['x']),
                         int(params['spacegrain']))
        y1 = np.linspace(min(params['y']),
                         max(params['y']),
                         int(params['spacegrain']))
        x2, y2 = np.meshgrid(x1, y1, indexing='xy')

        # Initialize the z-axis as an array of zero values
        z2 = np.zeros((params['x'].size, params['z'].size))

        # Apply scipy interpolate radial basis function, choosing
        # the rbffunc parameter
        spline = sp.interpolate.Rbf(
            params['x'],
            params['y'],
            params['z'],
            function=params['rbffunc'],
            smooth=5,
            epsilon=5)

        # Populate z-axis array using this function
        z2 = spline(x2, y2)

        # Create figure and axis objects and format
        opt_dict = cls._create_opt_labels(
            params=params, 
            opt_dict=opt_dict, 
            output='mpl'
            )    

        opt_dict['strikes'] = np.array(params['x'])
        opt_dict['ttms'] = np.array(params['y'])
        opt_dict['vols'] = np.array(params['z'])
        opt_dict['strikes_linspace'] = x1
        opt_dict['ttms_linspace'] = y1
        opt_dict['strikes_linspace_array'] = x2
        opt_dict['ttms_linspace_array'] = y2
        opt_dict['vol_surface'] = z2
        
        return opt_dict


    @classmethod
    def _interactive_graph(
        cls,
        params: dict,
        tables: dict,
        opt_dict: dict) -> dict:

        params = cls._set_contours(params=params, tables=tables)

        # Specify the 3 axis values
        params['x'] = tables['data_3D']['TTM'] * 365
        params['y'] = tables['data_3D']['Strike']
        params['z'] = tables['data_3D']['Graph Vol'] * 100

        # Create arrays across x and y-axes of equally spaced
        # points from min to max values
        x1 = np.linspace(
            params['x'].min(),
            params['x'].max(),
            int(params['spacegrain'])
            )
        y1 = np.linspace(
            params['y'].min(),
            params['y'].max(),
            int(params['spacegrain'])
            )
        params['x2'], params['y2'] = np.meshgrid(x1, y1, indexing='xy')

        # If surfacetype is 'interactive_mesh', map the z-axis with
        # the scipy griddata method, applying cubic spline
        # interpolation
        if params['surfacetype'] == 'interactive_mesh':
            params['z2'] = griddata((params['x'], params['y']),
                                    params['z'],
                                    (params['x2'], params['y2']),
                                    method='cubic')

        # If surfacetype is 'interactive_spline', apply scipy
        # interpolate radial basis function, choosing the rbffunc
        # parameter
        if params['surfacetype'] == 'interactive_spline':
            params['z2'] = np.zeros((params['x'].size, params['z'].size))
            spline = sp.interpolate.Rbf(
                params['x'],
                params['y'],
                params['z'],
                function=params['rbffunc'],
                smooth=5,
                epsilon=5)
            params['z2'] = spline(params['x2'], params['y2'])

        opt_dict = cls._create_opt_labels(
            params=params, 
            opt_dict=opt_dict, 
            output='plotly')  

        opt_dict['strikes'] = np.array(tables['data_3D']['Strike'])
        opt_dict['ttms'] = np.array(tables['data_3D']['TTM'] * 365)
        opt_dict['vols'] = np.array(tables['data_3D']['Graph Vol'] * 100)
        opt_dict['ttms_linspace'] = x1
        opt_dict['strikes_linspace'] = y1
        opt_dict['ttms_linspace_array'] = params['x2']
        opt_dict['strikes_linspace_array'] = params['y2']
        opt_dict['vol_surface'] = params['z2']

        return opt_dict


    @staticmethod
    def _set_contours(
        params: dict,
        tables: dict) -> dict:

        # Set the range of x, y and z contours and interval
        params['contour_x_start'] = 0
        params['contour_x_stop'] = 2 * 360
        params['contour_x_size'] = params['contour_x_stop'] / 18
        params['contour_y_start'] = tables['data_3D']['Strike'].min()
        params['contour_y_stop'] = tables['data_3D']['Strike'].max()

        # Vary the strike interval based on spot level
        if ((tables['data_3D']['Strike'].max()
             - tables['data_3D']['Strike'].min()) > 2000):
            params['contour_y_size'] = 200
        elif ((tables['data_3D']['Strike'].max()
               - tables['data_3D']['Strike'].min()) > 1000):
            params['contour_y_size'] = 100
        elif ((tables['data_3D']['Strike'].max()
               - tables['data_3D']['Strike'].min()) > 250):
            params['contour_y_size'] = 50
        elif ((tables['data_3D']['Strike'].max()
               - tables['data_3D']['Strike'].min()) > 50):
            params['contour_y_size'] = 10
        else:
            params['contour_y_size'] = 5

        # Set z contours
        params['contour_z_start'] = 0
        params['contour_z_stop'] = 100
        params['contour_z_size'] = 5

        return params


    @staticmethod
    def _create_opt_labels(
        params: dict, 
        opt_dict: dict,
        output: str) -> dict:

        if output == 'mpl':
            # Add labels to option dictionary  
            opt_dict['x_label'] = 'Strike'
            opt_dict['y_label'] = 'Time to Expiration (Days)'
            opt_dict['z_label'] = 'Implied Volatility %'
            

        elif output == 'line':
            opt_dict['x_label'] = 'Strike'
            opt_dict['y_label'] = 'Implied Volatility %'
            opt_dict['legend_title'] = 'Option Expiry'

        else: 
            opt_dict['x_label'] = 'Time to Expiration (Days)'
            opt_dict['y_label'] = 'Strike'
            opt_dict['z_label'] = 'Implied Volatility %'

        opt_dict['title'] = (
            str(params['ticker_label'])
            +' Implied Volatility '
            +str(params['voltype'].title())
            +' Price '
            +str(params['start_date'])
            )    

        return opt_dict
