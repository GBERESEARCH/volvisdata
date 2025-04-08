"""
Methods for calibrating volatility surface using SVI.

"""
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
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

    @classmethod
    def svi_calibrate(cls, strikes, vols, ttm, forward_price, params):
        """
        Calibrate SVI parameters for a single maturity with ridge regularization
        
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
        
        # Define the objective function with ridge regularization
        def objective(params_vec):
            a, b, rho, m, sigma = params_vec
            
            # Apply constraints
            if b <= 0 or abs(rho) >= 1 or sigma <= 0:
                return 1e10
            
            # Calculate model values
            w_model = cls.svi_function(k, a, b, rho, m, sigma)
            
            # Fit error
            fit_error = np.sum((w - w_model)**2)
            
            # Ridge regularization term
            reg_weight = params.get('svi_reg_weight', 0.0)
            reg_penalty = reg_weight * (b**2 + sigma**2 + rho**2 * 0.5)
            
            return fit_error + reg_penalty
        
        # Perform the optimization
        result = minimize(
            objective,
            initial_params,
            bounds=params['svi_bounds'],
            method='L-BFGS-B',
            options={'maxiter': params['svi_max_iter'], 'ftol': params['svi_tol']}
        )
        
        return result.x


    @classmethod
    def fit_svi_surface(cls, data, params):
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

        if params['svi_joint_calibration']:
            return cls.fit_svi_surface_joint(data, params)

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
            a, b, rho, m, sigma = cls.svi_calibrate(strikes, vols, ttm, forward_price, params)

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
    

    @classmethod
    def fit_svi_surface_joint(cls, data, params):
        """
        Fit SVI model to all tenors simultaneously with term structure regularization
        
        Parameters
        ----------
        data : DataFrame
            Option data with columns 'Strike', 'TTM', and implied vol columns
        params : dict
            Dictionary of parameters including spot price and rates
        
        Returns
        -------
        dict
            Dictionary of SVI parameters for each maturity
        """
        import numpy as np
        from scipy.optimize import minimize
        
        # Extract unique maturities
        ttms = sorted(list(set(data['TTM'])))
        
        # Extract strikes and vols for all maturities
        vol_col = params['vols_dict'][params['voltype']]
        
        # Initialize parameters for all maturities (5 params per maturity)
        initial_params = []
        ttm_data_dict = {}
        forward_prices = {}
        log_moneyness_dict = {}
        total_variance_dict = {}
        
        # Prepare data and initial parameters for each maturity
        for ttm in ttms:
            # Filter data for this maturity
            ttm_data = data[data['TTM'] == ttm]
            ttm_data_dict[ttm] = ttm_data
            
            # Get strikes and vols
            strikes = np.array(ttm_data['Strike'])
            vols = np.array(ttm_data[vol_col])
            
            # Calculate forward price using parameters
            spot = params['spot'] if params['spot'] is not None else params['extracted_spot']
            forward_price = spot * np.exp((params['r'] - params['q']) * ttm)
            forward_prices[ttm] = forward_price
            
            # Convert to log-moneyness and total variance
            log_moneyness = np.log(strikes / forward_price)
            log_moneyness_dict[ttm] = log_moneyness
            
            total_variance = vols**2 * ttm
            total_variance_dict[ttm] = total_variance
            
            # Set initial parameters
            if params['svi_compute_initial']:
                # Compute reasonable initial values from data
                a_init = np.min(total_variance)
                b_init = (np.max(total_variance) - np.min(total_variance)) / 2
                rho_init = params['svi_rho_init']
                m_init = params['svi_m_init']
                sigma_init = params['svi_sigma_init']
            else:
                a_init = params['svi_a_init']
                b_init = params['svi_b_init']
                rho_init = params['svi_rho_init']
                m_init = params['svi_m_init']
                sigma_init = params['svi_sigma_init']
            
            # Add these initial parameters to our list
            initial_params.extend([a_init, b_init, rho_init, m_init, sigma_init])
        
        # Define joint objective function with term structure regularization
        def joint_objective(all_params):
            # Reshape parameters into matrix (rows = tenors, cols = parameters)
            param_matrix = np.array(all_params).reshape(len(ttms), 5)
            
            # Calculate fit error
            fit_error = 0
            
            for i, ttm in enumerate(ttms):
                # Get data for this tenor
                k = log_moneyness_dict[ttm]
                w = total_variance_dict[ttm]
                
                # Get parameters for this tenor
                a, b, rho, m, sigma = param_matrix[i]
                
                # Apply constraints (additional cost for invalid parameters)
                if b <= 0 or abs(rho) >= 1 or sigma <= 0:
                    return 1e10
                
                # Calculate model values
                w_model = cls.svi_function(k, a, b, rho, m, sigma)
                
                # Add fit error for this tenor
                tenor_fit_error = np.sum((w - w_model)**2)
                fit_error += tenor_fit_error
            
            # Regularization across tenors - penalize parameter differences
            term_reg_weight = params.get('svi_term_reg_weight', 0.5)
            term_structure_penalty = 0
            
            if len(ttms) > 1:
                for i in range(len(ttms) - 1):
                    # Calculate weighted parameter differences between adjacent tenors
                    # Weight by inverse of tenor difference to penalize close tenors more
                    tenor_diff = ttms[i+1] - ttms[i]
                    weight = 1.0 / max(tenor_diff, 0.01)  # Avoid division by zero
                    
                    param_diff = param_matrix[i+1] - param_matrix[i]
                    
                    # Square and sum the differences (L2 norm)
                    term_structure_penalty += weight * np.sum(param_diff**2)
                
                # Scale penalty by regularization weight
                term_structure_penalty *= term_reg_weight
            
            # Within-tenor regularization (parameter magnitude penalty)
            reg_weight = params.get('svi_reg_weight', 0.01)
            parameter_penalty = 0
            
            for i in range(len(ttms)):
                a, b, rho, m, sigma = param_matrix[i]
                penalty = b**2 + sigma**2 + rho**2 * 0.5
                parameter_penalty += penalty
            
            parameter_penalty *= reg_weight
            
            # Total objective with both fit errors and regularization
            return fit_error + term_structure_penalty + parameter_penalty
        
        # Create full bounds list for all parameters
        bounds = []
        for _ in ttms:
            bounds.extend(params['svi_bounds'])
        
        # Perform the optimization
        result = minimize(
            joint_objective,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': params['svi_max_iter'], 'ftol': params['svi_tol']}
        )
        
        # Extract optimized parameters
        optimized_params = result.x.reshape(len(ttms), 5)
        
        # Create result dictionary
        svi_params = {}
        for i, ttm in enumerate(ttms):
            a, b, rho, m, sigma = optimized_params[i]
            svi_params[ttm] = {
                'a': a,
                'b': b,
                'rho': rho,
                'm': m,
                'sigma': sigma,
                'forward': forward_prices[ttm]
            }
        
        return svi_params


    @classmethod
    def compute_svi_surface(cls, strikes_grid, ttms_grid, svi_params, params):
        """
        Compute volatility surface using SVI parameters with vectorized operations.
        
        Parameters
        ----------
        strikes_grid : ndarray
            2D grid of strike prices
        ttms_grid : ndarray
            2D grid of time to maturities (in years)
        svi_params : dict
            Dictionary of SVI parameters for each maturity
        params : dict
            Dictionary of configuration parameters
        
        Returns
        -------
        ndarray
            2D grid of implied volatilities
        """
    
        # Get list of ttms for which we have SVI parameters
        svi_ttms = np.array(sorted(list(svi_params.keys())))
        
        # Flatten grids for vectorized computation
        strikes_flat = strikes_grid.flatten()
        ttms_flat = ttms_grid.flatten()
        vol_flat = np.zeros_like(strikes_flat)
        
        # Handle case with single tenor point
        if len(svi_ttms) <= 1:
            ttm_params = svi_params[svi_ttms[0]]
            
            for i, (strike, ttm) in enumerate(zip(strikes_flat, ttms_flat)):
                # Skip if maturity is too small to avoid numerical issues
                if ttm < params['epsilon']:  # Use epsilon from params
                    vol_flat[i] = 0
                    continue
                
                # Calculate log-moneyness
                k = np.log(strike / ttm_params['forward'])
                
                # Calculate total variance using SVI function
                w = cls.svi_function(k, ttm_params['a'], ttm_params['b'],
                                        ttm_params['rho'], ttm_params['m'],
                                        ttm_params['sigma'])
                
                # Convert to implied volatility
                vol_flat[i] = np.sqrt(max(0, w) / ttm)
            
            return vol_flat.reshape(strikes_grid.shape)
        
        # Extract parameter arrays for interpolation
        ttm_array = np.array(svi_ttms)
        a_array = np.array([svi_params[t]['a'] for t in svi_ttms])
        b_array = np.array([svi_params[t]['b'] for t in svi_ttms])
        rho_array = np.array([svi_params[t]['rho'] for t in svi_ttms])
        m_array = np.array([svi_params[t]['m'] for t in svi_ttms])
        sigma_array = np.array([svi_params[t]['sigma'] for t in svi_ttms])
        forward_array = np.array([svi_params[t]['forward'] for t in svi_ttms])
        
        # Select interpolation method
        interpolation_method = params.get('svi_interpolation_method', 'linear')
        
        # Create appropriate interpolators based on method and available points
        if interpolation_method == 'pchip' and len(svi_ttms) > 2:
            # Use PCHIP (monotonic) interpolation
            a_interp = PchipInterpolator(ttm_array, a_array)
            b_interp = PchipInterpolator(ttm_array, b_array)
            rho_interp = PchipInterpolator(ttm_array, rho_array)
            m_interp = PchipInterpolator(ttm_array, m_array)
            sigma_interp = PchipInterpolator(ttm_array, sigma_array)
            forward_interp = PchipInterpolator(ttm_array, forward_array)
        else:
            # Ensure interpolation method is valid and limit by available points
            valid_method = interpolation_method
            if len(svi_ttms) < 3 and valid_method == 'cubic':
                valid_method = 'quadratic'
            if len(svi_ttms) < 2 and valid_method == 'quadratic':
                valid_method = 'linear'
            
            # Default fill value (using bounds[0][0] as a reasonable numeric default)
            default_fill = params['svi_bounds'][0][0] if params['svi_bounds'][0][0] is not None else 0.0
            
            # Create interpolators with valid string method
            a_interp = interp1d(ttm_array, a_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
            b_interp = interp1d(ttm_array, b_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
            rho_interp = interp1d(ttm_array, rho_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
            m_interp = interp1d(ttm_array, m_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
            sigma_interp = interp1d(ttm_array, sigma_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
            forward_interp = interp1d(ttm_array, forward_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
        
        # Process each point in the grid
        for i, (strike, ttm) in enumerate(zip(strikes_flat, ttms_flat)):
            # Skip if maturity is too small
            if ttm < params['epsilon']:  # Use epsilon from params
                vol_flat[i] = 0
                continue
            
            # Interpolate parameters for this maturity
            try:
                a = float(a_interp(ttm))
                b = float(b_interp(ttm))
                rho = float(rho_interp(ttm))
                m = float(m_interp(ttm))
                sigma = float(sigma_interp(ttm))
                forward = float(forward_interp(ttm))
            except Exception:
                # Fallback to nearest point if interpolation fails
                idx = int(np.abs(ttm_array - ttm).argmin())
                a = float(a_array[idx])
                b = float(b_array[idx])
                rho = float(rho_array[idx])
                m = float(m_array[idx])
                sigma = float(sigma_array[idx])
                forward = float(forward_array[idx])
            
            # Enforce parameter constraints from bounds
            bounds = params['svi_bounds']
            
            # Apply bounds with proper handling of None values
            min_b = float(bounds[1][0]) if bounds[1][0] is not None else params['svi_b_init']
            max_b = float(bounds[1][1]) if bounds[1][1] is not None else params['svi_b_init'] * 100
            b = max(min_b, min(max_b, b))
            
            min_rho = float(bounds[2][0]) if bounds[2][0] is not None else -bounds[2][1]
            max_rho = float(bounds[2][1]) if bounds[2][1] is not None else -bounds[2][0]
            rho = max(min_rho, min(max_rho, rho))
            
            min_sigma = float(bounds[4][0]) if bounds[4][0] is not None else params['svi_sigma_init'] * 0.1
            max_sigma = float(bounds[4][1]) if bounds[4][1] is not None else params['svi_sigma_init'] * 100
            sigma = max(min_sigma, min(max_sigma, sigma))
            
            # Calculate log-moneyness
            k = np.log(strike / forward)
            
            # Apply SVI formula to get total implied variance
            w = cls.svi_function(k, a, b, rho, m, sigma)
            
            # Convert total variance to annualized volatility
            vol_flat[i] = np.sqrt(max(0, w) / ttm)
        
        # Reshape back to original grid dimensions
        vol_surface = vol_flat.reshape(strikes_grid.shape)
        
        return vol_surface
        