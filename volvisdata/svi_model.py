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
    def svi_function(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
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
    def svi_calibrate(cls, strikes: np.ndarray, vols: np.ndarray, ttm: float, 
                     forward_price: float, params: dict) -> tuple[float, float, float, float, float]:
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
        config = params['svi_config_params']
        
        k = np.log(strikes / forward_price)
        w = vols**2 * ttm
        
        if config['compute_initial']:
            a_init = np.min(w)
            b_init = (np.max(w) - np.min(w)) / 2
            rho_init = config['rho_init']
            m_init = config['m_init']
            sigma_init = config['sigma_init']
        else:
            a_init = config['a_init']
            b_init = config['b_init']
            rho_init = config['rho_init']
            m_init = config['m_init']
            sigma_init = config['sigma_init']
        
        initial_params = (a_init, b_init, rho_init, m_init, sigma_init)
        
        ttm_days = ttm * 365
        if ttm_days < config['short_dated_threshold_days']:
            bounds = [
                config['bounds'][0],
                (config['bounds'][1][0], config['short_dated_b_max']),
                (config['short_dated_rho_min'], config['short_dated_rho_max']),
                config['bounds'][3],
                config['bounds'][4],
            ]
        else:
            bounds = config['bounds']
        
        def objective(params_vec: tuple) -> float:
            a, b, rho, m, sigma = params_vec
            
            if b <= 0 or abs(rho) >= 1 or sigma <= 0:
                return 1e10
            
            w_model = cls.svi_function(k, a, b, rho, m, sigma)
            fit_error = np.sum((w - w_model)**2)
            
            reg_weight = config['reg_weight']
            reg_penalty = reg_weight * (b**2 + sigma**2 + rho**2 * 0.5)
            
            return fit_error + reg_penalty
        
        result = minimize(
            objective,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': config['max_iter'], 'ftol': config['tol']}
        )
        
        return result.x


    @classmethod
    def fit_svi_surface(cls, data, params: dict) -> dict:
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
        config = params['svi_config_params']

        if config['joint_calibration']:
            return cls.fit_svi_surface_joint(data, params)

        ttms = sorted(list(set(data['TTM'])))
        
        verbose = config['verbose']
        verbose_level = config['verbose_level']
        
        if verbose:
            print(f"\nCalibrating SVI surface for {len(ttms)} maturities (slice-by-slice)...")
        
        svi_params = {}

        for i, ttm in enumerate(ttms):
            ttm_data = data[data['TTM'] == ttm]

            strikes = np.array(ttm_data['Strike'])
            vol_col = params['vols_dict'][params['voltype']]
            vols = np.array(ttm_data[vol_col])

            spot = params['spot'] if params['spot'] is not None else params['extracted_spot']
            forward_price = spot * np.exp((params['r'] - params['q']) * ttm)

            a, b, rho, m, sigma = cls.svi_calibrate(strikes, vols, ttm, forward_price, params)
            
            k = np.log(strikes / forward_price)
            w_fitted = cls.svi_function(k, a, b, rho, m, sigma)
            vols_fitted = np.sqrt(w_fitted / ttm)
            rmse_vol = np.sqrt(np.mean((vols - vols_fitted)**2))

            svi_params[ttm] = {
                'a': a,
                'b': b,
                'rho': rho,
                'm': m,
                'sigma': sigma,
                'forward': forward_price
            }
            
            if verbose and verbose_level >= 2:
                print(f"  TTM {ttm:.4f} ({int(ttm*365):3d}d): "
                      f"RMSE={rmse_vol:.4f}, b={b:.3f}, rho={rho:.3f}")
        
        if verbose:
            rmses = []
            for ttm in ttms:
                ttm_data = data[data['TTM'] == ttm]
                strikes = np.array(ttm_data['Strike'])
                vol_col = params['vols_dict'][params['voltype']]
                vols = np.array(ttm_data[vol_col])
                spot = params['spot'] if params['spot'] is not None else params['extracted_spot']
                forward_price = spot * np.exp((params['r'] - params['q']) * ttm)
                
                k = np.log(strikes / forward_price)
                p = svi_params[ttm]
                w_fitted = cls.svi_function(k, p['a'], p['b'], p['rho'], p['m'], p['sigma'])
                vols_fitted = np.sqrt(w_fitted / ttm)
                rmse = np.sqrt(np.mean((vols - vols_fitted)**2))
                rmses.append(rmse)
            
            print(f"\nCalibration complete:")
            print(f"  Average RMSE: {np.mean(rmses):.4f}")
            print(f"  Max RMSE: {np.max(rmses):.4f}")

        return svi_params
    

    @classmethod
    def fit_svi_surface_joint(cls, data, params: dict) -> dict:
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
        config = params['svi_config_params']
        
        ttms = sorted(list(set(data['TTM'])))
        
        verbose = config['verbose']
        verbose_level = config['verbose_level']
        
        if verbose:
            print(f"\nCalibrating SVI surface for {len(ttms)} maturities (joint optimization)...")
        
        spot = params['spot'] if params['spot'] is not None else params['extracted_spot']
        forward_prices = {}
        log_moneyness_dict = {}
        total_variance_dict = {}
        
        for ttm in ttms:
            forward_prices[ttm] = spot * np.exp((params['r'] - params['q']) * ttm)
            
            ttm_data = data[data['TTM'] == ttm]
            strikes = np.array(ttm_data['Strike'])
            vol_col = params['vols_dict'][params['voltype']]
            vols = np.array(ttm_data[vol_col])
            
            log_moneyness_dict[ttm] = np.log(strikes / forward_prices[ttm])
            total_variance_dict[ttm] = vols**2 * ttm
        
        initial_params = []
        for ttm in ttms:
            w = total_variance_dict[ttm]
            
            if config['compute_initial']:
                a_init = np.min(w)
                b_init = (np.max(w) - np.min(w)) / 2
            else:
                a_init = config['a_init']
                b_init = config['b_init']
            
            initial_params.extend([
                a_init,
                b_init,
                config['rho_init'],
                config['m_init'],
                config['sigma_init']
            ])
        
        initial_params = np.array(initial_params)
        
        def joint_objective(params_flat: np.ndarray) -> float:
            param_matrix = params_flat.reshape(len(ttms), 5)
            
            fit_error = 0
            for i, ttm in enumerate(ttms):
                a, b, rho, m, sigma = param_matrix[i]
                
                k = log_moneyness_dict[ttm]
                w = total_variance_dict[ttm]
                
                if b <= 0 or abs(rho) >= 1 or sigma <= 0:
                    return 1e10
                
                w_model = cls.svi_function(k, a, b, rho, m, sigma)
                tenor_fit_error = np.sum((w - w_model)**2)
                fit_error += tenor_fit_error
            
            term_reg_weight = config['term_reg_weight']
            term_structure_penalty = 0
            
            if len(ttms) > 1:
                for i in range(len(ttms) - 1):
                    tenor_diff = ttms[i+1] - ttms[i]
                    weight = 1.0 / max(tenor_diff, 0.01)
                    
                    param_diff = param_matrix[i+1] - param_matrix[i]
                    term_structure_penalty += weight * np.sum(param_diff**2)
                
                term_structure_penalty *= term_reg_weight
            
            reg_weight = config['reg_weight']
            parameter_penalty = 0
            
            for i in range(len(ttms)):
                a, b, rho, m, sigma = param_matrix[i]
                penalty = b**2 + sigma**2 + rho**2 * 0.5
                parameter_penalty += penalty
            
            parameter_penalty *= reg_weight
            
            return fit_error + term_structure_penalty + parameter_penalty
        
        bounds = []
        for ttm in ttms:
            ttm_days = ttm * 365
            if ttm_days < config['short_dated_threshold_days']:
                bounds.extend([
                    config['bounds'][0],
                    (config['bounds'][1][0], config['short_dated_b_max']),
                    (config['short_dated_rho_min'], config['short_dated_rho_max']),
                    config['bounds'][3],
                    config['bounds'][4],
                ])
            else:
                bounds.extend(config['bounds'])
        
        if verbose and verbose_level >= 2:
            print("  Starting joint optimization...")
        
        result = minimize(
            joint_objective,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': config['max_iter'], 'ftol': config['tol']}
        )
        
        optimized_params = result.x.reshape(len(ttms), 5)
        
        svi_params = {}
        rmses = []
        
        for i, ttm in enumerate(ttms):
            a, b, rho, m, sigma = optimized_params[i]
            
            k = log_moneyness_dict[ttm]
            w = total_variance_dict[ttm]
            
            w_fitted = cls.svi_function(k, a, b, rho, m, sigma)
            vols_fitted = np.sqrt(w_fitted / ttm)
            vols_market = np.sqrt(w / ttm)
            
            rmse_vol = np.sqrt(np.mean((vols_market - vols_fitted)**2))
            rmses.append(rmse_vol)
            
            svi_params[ttm] = {
                'a': a,
                'b': b,
                'rho': rho,
                'm': m,
                'sigma': sigma,
                'forward': forward_prices[ttm]
            }
            
            if verbose and verbose_level >= 2:
                print(f"  TTM {ttm:.4f} ({int(ttm*365):3d}d): "
                      f"RMSE={rmse_vol:.4f}, b={b:.3f}, rho={rho:.3f}")
        
        if verbose:
            print(f"\nJoint calibration complete:")
            print(f"  Average RMSE: {np.mean(rmses):.4f}")
            print(f"  Max RMSE: {np.max(rmses):.4f}")
            print(f"  Converged: {result.success}")
        
        return svi_params


    @classmethod
    def compute_svi_surface(cls, strikes_grid: np.ndarray, ttms_grid: np.ndarray, 
                           svi_params: dict, params: dict) -> np.ndarray:
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
        config = params['svi_config_params']
        
        svi_ttms = np.array(sorted(list(svi_params.keys())))
        
        strikes_flat = strikes_grid.flatten()
        ttms_flat = ttms_grid.flatten()
        vol_flat = np.zeros_like(strikes_flat)
        
        if len(svi_ttms) <= 1:
            ttm_params = svi_params[svi_ttms[0]]
            
            for i, (strike, ttm) in enumerate(zip(strikes_flat, ttms_flat)):
                if ttm < params['epsilon']:
                    vol_flat[i] = 0
                    continue
                
                k = np.log(strike / ttm_params['forward'])
                
                w = cls.svi_function(k, ttm_params['a'], ttm_params['b'],
                                        ttm_params['rho'], ttm_params['m'],
                                        ttm_params['sigma'])
                
                vol_flat[i] = np.sqrt(max(0, w) / ttm)
            
            return vol_flat.reshape(strikes_grid.shape)
        
        ttm_array = np.array(svi_ttms)
        a_array = np.array([svi_params[t]['a'] for t in svi_ttms])
        b_array = np.array([svi_params[t]['b'] for t in svi_ttms])
        rho_array = np.array([svi_params[t]['rho'] for t in svi_ttms])
        m_array = np.array([svi_params[t]['m'] for t in svi_ttms])
        sigma_array = np.array([svi_params[t]['sigma'] for t in svi_ttms])
        forward_array = np.array([svi_params[t]['forward'] for t in svi_ttms])
        
        interpolation_method = config['interpolation_method']
        
        if interpolation_method == 'pchip' and len(svi_ttms) > 2:
            a_interp = PchipInterpolator(ttm_array, a_array)
            b_interp = PchipInterpolator(ttm_array, b_array)
            rho_interp = PchipInterpolator(ttm_array, rho_array)
            m_interp = PchipInterpolator(ttm_array, m_array)
            sigma_interp = PchipInterpolator(ttm_array, sigma_array)
            forward_interp = PchipInterpolator(ttm_array, forward_array)
        else:
            valid_method = interpolation_method
            if len(svi_ttms) < 3 and valid_method == 'cubic':
                valid_method = 'quadratic'
            if len(svi_ttms) < 2 and valid_method == 'quadratic':
                valid_method = 'linear'
            
            default_fill = config['bounds'][0][0] if config['bounds'][0][0] is not None else 0.0
            
            a_interp = interp1d(ttm_array, a_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
            b_interp = interp1d(ttm_array, b_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
            rho_interp = interp1d(ttm_array, rho_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
            m_interp = interp1d(ttm_array, m_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
            sigma_interp = interp1d(ttm_array, sigma_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
            forward_interp = interp1d(ttm_array, forward_array, kind=valid_method, bounds_error=False, fill_value=default_fill)
        
        for i, (strike, ttm) in enumerate(zip(strikes_flat, ttms_flat)):
            if ttm < params['epsilon']:
                vol_flat[i] = 0
                continue
            
            try:
                a = float(a_interp(ttm))
                b = float(b_interp(ttm))
                rho = float(rho_interp(ttm))
                m = float(m_interp(ttm))
                sigma = float(sigma_interp(ttm))
                forward = float(forward_interp(ttm))
            except Exception:
                idx = int(np.abs(ttm_array - ttm).argmin())
                a = float(a_array[idx])
                b = float(b_array[idx])
                rho = float(rho_array[idx])
                m = float(m_array[idx])
                sigma = float(sigma_array[idx])
                forward = float(forward_array[idx])
            
            bounds = config['bounds']
            
            min_b = float(bounds[1][0]) if bounds[1][0] is not None else config['b_init']
            max_b = float(bounds[1][1]) if bounds[1][1] is not None else config['b_init'] * 100
            b = max(min_b, min(max_b, b))
            
            min_rho = float(bounds[2][0]) if bounds[2][0] is not None else -bounds[2][1]
            max_rho = float(bounds[2][1]) if bounds[2][1] is not None else -bounds[2][0]
            rho = max(min_rho, min(max_rho, rho))
            
            min_sigma = float(bounds[4][0]) if bounds[4][0] is not None else config['sigma_init'] * 0.1
            max_sigma = float(bounds[4][1]) if bounds[4][1] is not None else config['sigma_init'] * 100
            sigma = max(min_sigma, min(max_sigma, sigma))
            
            k = np.log(strike / forward)
            w = cls.svi_function(k, a, b, rho, m, sigma)
            vol_flat[i] = np.sqrt(max(0, w) / ttm)
        
        vol_surface = vol_flat.reshape(strikes_grid.shape)
        
        return vol_surface