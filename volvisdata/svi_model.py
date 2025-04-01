"""
Methods for calibrating volatility surface using SVI.

"""
import numpy as np
from scipy.optimize import minimize

class SVIModel:
    """
    Stochastic Volatility Inspired model implementation for volatility surfaces

    The SVI parameterization is given by:
    w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))

    where:
    - w(k) is the total implied variance (σ² * T)
    - k is the log-moneyness (log(K/F))
    - a, b, ρ, m, and σ are the SVI parameters
    """

    @staticmethod
    def svi_function(k, a, b, rho, m, sigma):
        """
        SVI parametrization function

        Parameters
        ----------
        k : ndarray
            Log-moneyness (log(K/F))
        a : float
            Overall level parameter
        b : float
            Controls the angle between the left and right asymptotes
        rho : float
            Controls the skew/rotation (-1 <= rho <= 1)
        m : float
            Controls the horizontal translation
        sigma : float
            Controls the smoothness of the curve at the minimum

        Returns
        -------
        ndarray
            Total implied variance w(k)
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    @staticmethod
    def svi_calibrate(strikes, vols, ttm, forward_price, params):
        """
        Calibrate SVI parameters for a single maturity

        Parameters
        ----------
        strikes : ndarray
            Option strike prices
        vols : ndarray
            Implied volatilities corresponding to strikes
        ttm : float
            Time to maturity in years
        forward_price : float
            Forward price of the underlying
        params : dict
            Dictionary of parameters including SVI configuration parameters

        Returns
        -------
        tuple
            Calibrated SVI parameters (a, b, rho, m, sigma)
        """
        # Convert to log-moneyness
        k = np.log(strikes / forward_price)

        # Convert volatilities to total variance
        w = vols**2 * ttm

        # Set initial parameters from params dict
        if params['svi_compute_initial']:
            # Compute reasonable initial values based on data
            a_init = np.min(w)
            b_init = (np.max(w) - np.min(w)) / 2
            # Use defaults from params for other values
            rho_init = params['svi_rho_init']
            m_init = params['svi_m_init']
            sigma_init = params['svi_sigma_init']
        else:
            # Use values directly from params
            a_init = params['svi_a_init']
            b_init = params['svi_b_init']
            rho_init = params['svi_rho_init']
            m_init = params['svi_m_init']
            sigma_init = params['svi_sigma_init']

        initial_params = (a_init, b_init, rho_init, m_init, sigma_init)

        # Define the objective function to minimize (sum of squared errors)
        def objective(params):
            a, b, rho, m, sigma = params

            # Apply constraints
            if b < 0 or abs(rho) > 1 or sigma <= 0:
                return 1e10  # Large penalty for invalid parameters

            w_model = SVIModel.svi_function(k, a, b, rho, m, sigma)
            return np.sum((w - w_model)**2)

        # Perform the optimization using params
        result = minimize(
            objective,
            initial_params,
            bounds=params['svi_bounds'],
            method='L-BFGS-B',
            options={'maxiter': params['svi_max_iter'], 'ftol': params['svi_tol']}
        )

        return result.x

    @staticmethod
    def fit_svi_surface(data, params):
        """
        Fit SVI model to the entire volatility surface

        Parameters
        ----------
        data : DataFrame
            Option data with columns 'Strike', 'TTM', and implied vol columns
        params : dict
            Dictionary of parameters including spot price and rates

        Returns
        -------
        dict
            Dictionary of SVI parameters for each maturity and interpolation function
        """
        # Extract unique maturities
        ttms = sorted(list(set(data['TTM'])))

        # Dictionary to store SVI parameters for each maturity
        svi_params = {}

        # Fit SVI model for each maturity
        for ttm in ttms:
            # Filter data for this maturity
            ttm_data = data[data['TTM'] == ttm]

            # Get strikes and vols
            strikes = np.array(ttm_data['Strike'])
            vol_col = params['vols_dict'][params['voltype']]
            vols = np.array(ttm_data[vol_col])

            # Calculate forward price using parameters from params dictionary
            spot = params['spot'] if params['spot'] is not None else params['extracted_spot']
            forward_price = spot * np.exp((params['r'] - params['q']) * ttm)

            # Calibrate SVI parameters using params dictionary
            a, b, rho, m, sigma = SVIModel.svi_calibrate(strikes, vols, ttm, forward_price, params)

            # Store parameters
            svi_params[ttm] = {
                'a': a,
                'b': b,
                'rho': rho,
                'm': m,
                'sigma': sigma,
                'forward': forward_price
            }

        return svi_params

    @staticmethod
    def compute_svi_surface(strikes_grid, ttms_grid, svi_params):
        """
        Compute volatility surface using SVI parameters

        Parameters
        ----------
        strikes_grid : ndarray
            2D grid of strike prices
        ttms_grid : ndarray
            2D grid of time to maturities (in years)
        svi_params : dict
            Dictionary of SVI parameters for each maturity

        Returns
        -------
        ndarray
            2D grid of implied volatilities
        """
        # Get list of ttms for which we have SVI parameters
        svi_ttms = sorted(list(svi_params.keys()))

        # Initialize volatility surface grid
        vol_surface = np.zeros_like(strikes_grid)

        # Compute SVI implied volatilities
        for i in range(strikes_grid.shape[0]):
            for j in range(strikes_grid.shape[1]):
                strike = strikes_grid[i, j]
                ttm = ttms_grid[i, j]

                # Find the closest ttms with SVI parameters
                idx = np.searchsorted(svi_ttms, ttm)

                # Handle boundary cases
                if idx == 0:
                    ttm_params = svi_params[svi_ttms[0]]
                elif idx == len(svi_ttms):
                    ttm_params = svi_params[svi_ttms[-1]]
                else:
                    # Interpolate between the two closest ttms
                    ttm_lower = svi_ttms[idx-1]
                    ttm_upper = svi_ttms[idx]

                    params_lower = svi_params[ttm_lower]
                    params_upper = svi_params[ttm_upper]

                    # Linear interpolation weight
                    w = (ttm - ttm_lower) / (ttm_upper - ttm_lower)

                    # Interpolate SVI parameters
                    a = params_lower['a'] * (1-w) + params_upper['a'] * w
                    b = params_lower['b'] * (1-w) + params_upper['b'] * w
                    rho = params_lower['rho'] * (1-w) + params_upper['rho'] * w
                    m = params_lower['m'] * (1-w) + params_upper['m'] * w
                    sigma = params_lower['sigma'] * (1-w) + params_upper['sigma'] * w
                    forward = params_lower['forward'] * (1-w) + params_upper['forward'] * w

                    ttm_params = {
                        'a': a,
                        'b': b,
                        'rho': rho,
                        'm': m,
                        'sigma': sigma,
                        'forward': forward
                        }

                # Calculate log-moneyness
                k = np.log(strike / ttm_params['forward'])

                # Calculate total implied variance using SVI function
                w = SVIModel.svi_function(k, ttm_params['a'], ttm_params['b'],
                                         ttm_params['rho'], ttm_params['m'],
                                         ttm_params['sigma'])

                # Convert total variance to implied volatility
                if w > 0:
                    vol_surface[i, j] = np.sqrt(w / ttm)
                else:
                    vol_surface[i, j] = 0

        return vol_surface

    @staticmethod
    def compute_svi_surface_vectorized(strikes_grid, ttms_grid, svi_params):
        """
        Compute volatility surface using SVI parameters (vectorized implementation)

        Parameters
        ----------
        strikes_grid : ndarray
            2D grid of strike prices
        ttms_grid : ndarray
            2D grid of time to maturities (in years)
        svi_params : dict
            Dictionary of SVI parameters for each maturity

        Returns
        -------
        ndarray
            2D grid of implied volatilities (in decimal form)
        """
        # Get list of ttms for which we have SVI parameters
        svi_ttms = np.array(sorted(list(svi_params.keys())))

        # Flatten grids for vectorized computation
        strikes_flat = strikes_grid.flatten()
        ttms_flat = ttms_grid.flatten()
        vol_flat = np.zeros_like(strikes_flat)

        # Process each point in the grid
        for i, strike in enumerate(strikes_flat):
            ttm = ttms_flat[i]

            # Find the closest ttms with SVI parameters
            idx = np.searchsorted(svi_ttms, ttm)

            # Interpolate SVI parameters based on maturity
            if idx == 0:
                ttm_params = svi_params[svi_ttms[0]]
            elif idx == len(svi_ttms):
                ttm_params = svi_params[svi_ttms[-1]]
            else:
                # Interpolate between adjacent maturities
                ttm_lower = svi_ttms[idx-1]
                ttm_upper = svi_ttms[idx]

                params_lower = svi_params[ttm_lower]
                params_upper = svi_params[ttm_upper]

                # Linear interpolation weight
                w = (ttm - ttm_lower) / (ttm_upper - ttm_lower)

                # Interpolate each SVI parameter
                ttm_params = {
                    'a': params_lower['a'] * (1-w) + params_upper['a'] * w,
                    'b': params_lower['b'] * (1-w) + params_upper['b'] * w,
                    'rho': params_lower['rho'] * (1-w) + params_upper['rho'] * w,
                    'm': params_lower['m'] * (1-w) + params_upper['m'] * w,
                    'sigma': params_lower['sigma'] * (1-w) + params_upper['sigma'] * w,
                    'forward': params_lower['forward'] * (1-w) + params_upper['forward'] * w
                }

            # Calculate log-moneyness
            k = np.log(strike / ttm_params['forward'])

            # Apply SVI formula to get total implied variance
            w = SVIModel.svi_function(k, ttm_params['a'], ttm_params['b'],
                                     ttm_params['rho'], ttm_params['m'],
                                     ttm_params['sigma'])

            # Convert total variance to annualized volatility
            vol_flat[i] = np.sqrt(max(0, w) / ttm)

        # Reshape back to original grid dimensions
        vol_surface = vol_flat.reshape(strikes_grid.shape)

        return vol_surface
    