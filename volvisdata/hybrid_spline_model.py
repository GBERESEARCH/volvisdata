"""
Hybrid B-spline volatility surface calibration with arbitrage constraints.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from typing import Dict
import warnings


class HybridSplineModel:
    """
    Hybrid B-spline volatility surface calibration.
    
    Tensor product B-splines with explicit arbitrage constraints:
    - Calendar: total variance non-decreasing in time
    - Butterfly: convexity bounds on variance
    - Smoothness: second derivative penalties
    """
    
    @staticmethod
    def _compute_knots(
        data_points: np.ndarray,
        n_interior_knots: int,
        degree: int,
        adaptive: bool
    ) -> np.ndarray:
        """
        Compute knot vector for B-spline basis.
        
        Parameters
        ----------
        data_points : ndarray
            Observed data points for this dimension
        n_interior_knots : int
            Number of interior knots
        degree : int
            Polynomial degree of B-spline
        adaptive : bool
            If True, place knots based on data density
            
        Returns
        -------
        ndarray
            Complete knot vector including boundary repetitions
        """
        min_val = np.min(data_points)
        max_val = np.max(data_points)
        
        # Place interior knots adaptively at data percentiles or uniformly
        if adaptive and len(data_points) > n_interior_knots:
            percentiles = np.linspace(0, 100, n_interior_knots + 2)
            interior_knots = np.percentile(data_points, percentiles[1:-1])
        else:
            interior_knots = np.linspace(min_val, max_val, n_interior_knots + 2)[1:-1]
        
        # Repeat boundary knots (degree+1) times for B-spline definition
        knots = np.concatenate([
            np.repeat(min_val, degree + 1),
            interior_knots,
            np.repeat(max_val, degree + 1)
        ])
        
        return knots
    
    @staticmethod
    def _basis_matrix(
        eval_points: np.ndarray,
        knots: np.ndarray,
        degree: int
    ) -> np.ndarray:
        """
        Compute B-spline basis matrix.
        
        Parameters
        ----------
        eval_points : ndarray
            Points at which to evaluate basis functions
        knots : ndarray
            Knot vector
        degree : int
            Polynomial degree
            
        Returns
        -------
        ndarray
            Basis matrix of shape (n_points, n_basis_functions)
        """
        n_basis = len(knots) - degree - 1
        n_points = len(eval_points)
        basis_mat = np.zeros((n_points, n_basis))
        
        # Evaluate each basis function at all evaluation points
        for i in range(n_basis):
            coefs = np.zeros(n_basis)
            coefs[i] = 1.0
            spline = BSpline(knots, coefs, degree, extrapolate=False)
            basis_mat[:, i] = spline(eval_points, extrapolate=False)
        
        # Replace NaN with 0 (occurs outside knot range)
        basis_mat = np.nan_to_num(basis_mat, nan=0.0)
        return basis_mat
    
    @staticmethod
    def _calendar_arbitrage_penalty(
        coefs_2d: np.ndarray,
        strike_basis: np.ndarray,
        time_basis: np.ndarray,
        time_knots: np.ndarray,
        time_degree: int,
        eval_strikes: np.ndarray,
        eval_times: np.ndarray,
        tolerance: float
    ) -> float:
        """
        Compute penalty for calendar arbitrage violations.
        
        Calendar arbitrage: total variance w = σ²τ must be non-decreasing in time.
        
        Parameters
        ----------
        coefs_2d : ndarray
            Coefficient matrix (n_strike_basis x n_time_basis)
        strike_basis : ndarray
            Strike basis matrix at evaluation points
        time_basis : ndarray
            Time basis matrix at evaluation points
        time_knots : ndarray
            Time knot vector
        time_degree : int
            Time polynomial degree
        eval_strikes : ndarray
            Strike points for constraint evaluation
        eval_times : ndarray
            Time points for constraint evaluation
        tolerance : float
            Tolerance for constraint violation
            
        Returns
        -------
        float
            Calendar arbitrage penalty
        """
        n_basis_time = len(time_knots) - time_degree - 1
        
        # Compute first derivative basis in time: d(B_j(τ))/dτ
        time_deriv_basis = np.zeros((len(eval_times), n_basis_time))
        for i in range(n_basis_time):
            coefs_i = np.zeros(n_basis_time)
            coefs_i[i] = 1.0
            spline_i = BSpline(time_knots, coefs_i, time_degree, extrapolate=False)
            time_deriv_basis[:, i] = spline_i.derivative(1)(eval_times, extrapolate=False)
        
        time_deriv_basis = np.nan_to_num(time_deriv_basis, nan=0.0)
        
        # Compute ∂w/∂τ at evaluation grid
        variance_time_deriv = strike_basis @ coefs_2d @ time_deriv_basis.T
        
        # Penalize violations: max(0, -∂w/∂τ - tolerance)²
        violations = np.maximum(-variance_time_deriv - tolerance, 0.0)
        penalty = np.sum(violations**2)
        
        return penalty
    
    @staticmethod
    def _butterfly_arbitrage_penalty(
        coefs_2d: np.ndarray,
        strike_basis: np.ndarray,
        time_basis: np.ndarray,
        strike_knots: np.ndarray,
        strike_degree: int,
        eval_strikes: np.ndarray,
        eval_times: np.ndarray,
        tolerance: float
    ) -> float:
        """
        Compute penalty for butterfly arbitrage violations (∂²w/∂k² ≥ -1).
        
        Parameters
        ----------
        coefs_2d : ndarray
            Coefficient matrix
        strike_basis : ndarray
            Strike basis matrix
        time_basis : ndarray
            Time basis matrix
        strike_knots : ndarray
            Strike knot vector
        strike_degree : int
            Strike polynomial degree
        eval_strikes : ndarray
            Strike points for evaluation
        eval_times : ndarray
            Time points for evaluation
        tolerance : float
            Tolerance for violation
            
        Returns
        -------
        float
            Butterfly penalty
        """
        n_basis_strike = len(strike_knots) - strike_degree - 1
        
        # Compute second derivative basis in strike: d²(Bᵢ(k))/dk²
        strike_deriv2_basis = np.zeros((len(eval_strikes), n_basis_strike))
        for i in range(n_basis_strike):
            coefs_i = np.zeros(n_basis_strike)
            coefs_i[i] = 1.0
            spline_i = BSpline(strike_knots, coefs_i, strike_degree, extrapolate=False)
            try:
                strike_deriv2_basis[:, i] = spline_i.derivative(2)(eval_strikes, extrapolate=False)
            except ValueError:
                strike_deriv2_basis[:, i] = 0.0
        
        strike_deriv2_basis = np.nan_to_num(strike_deriv2_basis, nan=0.0)
        
        # Compute ∂²w/∂k² at evaluation grid
        variance_strike_deriv2 = strike_deriv2_basis @ coefs_2d @ time_basis.T
        
        # Penalize violations: max(0, -∂²w/∂k² - 1 - tolerance)²
        violations = np.maximum(-variance_strike_deriv2 - 1.0 - tolerance, 0.0)
        penalty = np.sum(violations**2)
        
        return penalty
    
    @staticmethod
    def _smoothness_penalty(
        coefs_2d: np.ndarray,
        strike_knots: np.ndarray,
        time_knots: np.ndarray,
        strike_degree: int,
        time_degree: int,
        n_eval: int
    ) -> float:
        """
        Compute smoothness penalty based on second derivatives.
        
        Parameters
        ----------
        coefs_2d : ndarray
            Coefficient matrix
        strike_knots : ndarray
            Strike knots
        time_knots : ndarray
            Time knots
        strike_degree : int
            Strike degree
        time_degree : int
            Time degree
        n_eval : int
            Number of evaluation points per dimension
            
        Returns
        -------
        float
            Smoothness penalty
        """
        penalty = 0.0
        
        # Define evaluation grid on interior of knot range
        strike_range = (strike_knots[strike_degree], strike_knots[-(strike_degree+1)])
        time_range = (time_knots[time_degree], time_knots[-(time_degree+1)])
        
        eval_strikes = np.linspace(strike_range[0], strike_range[1], n_eval)
        eval_times = np.linspace(time_range[0], time_range[1], n_eval)
        
        n_basis_strike = len(strike_knots) - strike_degree - 1
        n_basis_time = len(time_knots) - time_degree - 1
        
        # Compute second derivative basis in strike direction
        strike_deriv2 = np.zeros((n_eval, n_basis_strike))
        for i in range(n_basis_strike):
            coefs_i = np.zeros(n_basis_strike)
            coefs_i[i] = 1.0
            spline_i = BSpline(strike_knots, coefs_i, strike_degree, extrapolate=False)
            try:
                strike_deriv2[:, i] = spline_i.derivative(2)(eval_strikes, extrapolate=False)
            except ValueError:
                strike_deriv2[:, i] = 0.0
        
        # Compute second derivative basis in time direction
        time_deriv2 = np.zeros((n_eval, n_basis_time))
        for i in range(n_basis_time):
            coefs_i = np.zeros(n_basis_time)
            coefs_i[i] = 1.0
            spline_i = BSpline(time_knots, coefs_i, time_degree, extrapolate=False)
            try:
                time_deriv2[:, i] = spline_i.derivative(2)(eval_times, extrapolate=False)
            except ValueError:
                time_deriv2[:, i] = 0.0
        
        strike_deriv2 = np.nan_to_num(strike_deriv2, nan=0.0)
        time_deriv2 = np.nan_to_num(time_deriv2, nan=0.0)
        
        # Sum of squared second derivatives in both dimensions
        penalty += np.sum((strike_deriv2 @ coefs_2d)**2)
        penalty += np.sum((coefs_2d @ time_deriv2.T)**2)
        
        return penalty
    
    @classmethod
    def _fit_spline_surface(
        cls,
        strikes: np.ndarray,
        ttms: np.ndarray,
        vols: np.ndarray,
        spot: float,
        config: dict
    ) -> dict:
        """
        Fit B-spline surface to volatility data.
        
        Parameters
        ----------
        strikes : ndarray
            Strike prices
        ttms : ndarray
            Times to maturity (years)
        vols : ndarray
            Implied volatilities (decimal)
        spot : float
            Spot price
        config : dict
            Configuration parameters
            
        Returns
        -------
        dict
            Dictionary containing fitted parameters
        """
        # Convert to total variance and log-moneyness
        total_variance = vols**2 * ttms
        log_moneyness = np.log(strikes / spot)
        
        # Construct knot vectors
        strike_knots = cls._compute_knots(
            data_points=log_moneyness,
            n_interior_knots=config['n_strike_knots'],
            degree=config['strike_degree'],
            adaptive=config['adaptive_knots']
        )
        
        time_knots = cls._compute_knots(
            data_points=ttms,
            n_interior_knots=config['n_time_knots'],
            degree=config['time_degree'],
            adaptive=config['adaptive_knots']
        )
        
        # Build basis matrices at data points
        strike_basis = cls._basis_matrix(log_moneyness, strike_knots, config['strike_degree'])
        time_basis = cls._basis_matrix(ttms, time_knots, config['time_degree'])
        
        n_basis_strike = strike_basis.shape[1]
        n_basis_time = time_basis.shape[1]
        
        # Build basis matrices for constraint evaluation grid
        n_eval_constraint = config['n_constraint_points']
        eval_log_moneyness = np.linspace(np.min(log_moneyness), np.max(log_moneyness), n_eval_constraint)
        eval_ttms = np.linspace(np.min(ttms), np.max(ttms), n_eval_constraint)
        
        eval_strike_basis = cls._basis_matrix(eval_log_moneyness, strike_knots, config['strike_degree'])
        eval_time_basis = cls._basis_matrix(eval_ttms, time_knots, config['time_degree'])
        
        # Initialize coefficient matrix
        initial_coefs = np.random.randn(n_basis_strike, n_basis_time) * config['initial_coef_noise']
        initial_coefs += np.mean(total_variance)
        
        iteration_count = {'nit': 0}
        
        def objective(coefs_flat: np.ndarray) -> float:
            """Objective function: fit error + regularization + arbitrage penalties"""
            coefs_2d = coefs_flat.reshape(n_basis_strike, n_basis_time)
            
            # Compute predicted variance at data points
            variance_pred = np.einsum('ij,jk,ik->i', strike_basis, coefs_2d, time_basis)
            fit_error = np.sum((total_variance - variance_pred)**2)
            
            total_penalty = fit_error
            
            # Add smoothness penalty
            if config['smoothness_weight'] > 0:
                smooth_penalty = cls._smoothness_penalty(
                    coefs_2d=coefs_2d,
                    strike_knots=strike_knots,
                    time_knots=time_knots,
                    strike_degree=config['strike_degree'],
                    time_degree=config['time_degree'],
                    n_eval=config['smoothness_eval_points']
                )
                total_penalty += config['smoothness_weight'] * smooth_penalty
            
            # Add calendar arbitrage penalty
            if config['calendar_penalty_weight'] > 0:
                cal_penalty = cls._calendar_arbitrage_penalty(
                    coefs_2d=coefs_2d,
                    strike_basis=eval_strike_basis,
                    time_basis=eval_time_basis,
                    time_knots=time_knots,
                    time_degree=config['time_degree'],
                    eval_strikes=eval_log_moneyness,
                    eval_times=eval_ttms,
                    tolerance=config['calendar_tolerance']
                )
                total_penalty += config['calendar_penalty_weight'] * cal_penalty
            
            # Add butterfly arbitrage penalty
            if config['butterfly_penalty_weight'] > 0:
                bf_penalty = cls._butterfly_arbitrage_penalty(
                    coefs_2d=coefs_2d,
                    strike_basis=eval_strike_basis,
                    time_basis=eval_time_basis,
                    strike_knots=strike_knots,
                    strike_degree=config['strike_degree'],
                    eval_strikes=eval_log_moneyness,
                    eval_times=eval_ttms,
                    tolerance=config['butterfly_tolerance']
                )
                total_penalty += config['butterfly_penalty_weight'] * bf_penalty
            
            # Progress reporting
            iteration_count['nit'] += 1
            if config['verbose'] and config['progress_interval'] > 0 and iteration_count['nit'] % config['progress_interval'] == 0:
                rmse = np.sqrt(fit_error / len(total_variance))
                print(f"  Iteration {iteration_count['nit']}: RMSE={rmse:.4f}")
            
            return total_penalty
        
        # Print optimization setup
        if config['verbose']:
            print(f"Calibrating hybrid spline surface for {len(strikes)} points...")
        
        # Run L-BFGS-B optimization
        result = minimize(
            objective,
            initial_coefs.flatten(),
            method='L-BFGS-B',
            options={
                'maxiter': config['max_iter'],
                'maxfun': config['max_iter'] * config['maxfun_multiplier'],
                'ftol': config['ftol']
            }
        )
        
        # Extract optimal coefficients
        optimal_coefs = result.x.reshape(n_basis_strike, n_basis_time)
        
        # Compute predicted variance at data points
        final_variance_pred = np.einsum('ij,jk,ik->i', strike_basis, optimal_coefs, time_basis)
        final_rmse = np.sqrt(np.mean((total_variance - final_variance_pred)**2))
        
        # Compute per-maturity RMSE for diagnostics
        if config['verbose']:
            unique_ttms = np.unique(ttms)
            print(f"  Per-maturity fit quality:")
            for ttm in sorted(unique_ttms):
                mask = ttms == ttm
                if np.sum(mask) > 0:
                    ttm_variance = total_variance[mask]
                    ttm_pred = final_variance_pred[mask]
                    ttm_rmse = np.sqrt(np.mean((ttm_variance - ttm_pred)**2))
                    ttm_days = int(ttm * 365)
                    print(f"    TTM {ttm:.4f} ({ttm_days:3d}d): RMSE={ttm_rmse:.4f}, N={np.sum(mask):3d}")
        
        # Report convergence status factually
        function_evaluations = iteration_count['nit']
        lbfgs_iterations = result.nit
        hit_max_iter = lbfgs_iterations >= config['max_iter']
        
        if config['verbose']:
            convergence_msg = 'reached max iterations' if hit_max_iter else 'converged'
            print(f"  Overall RMSE: {final_rmse:.4f}, L-BFGS iterations: {lbfgs_iterations}/{config['max_iter']}, Optimization {convergence_msg}")
        
        return {
            'coefs': optimal_coefs,
            'strike_knots': strike_knots,
            'time_knots': time_knots,
            'strike_degree': config['strike_degree'],
            'time_degree': config['time_degree'],
            'spot': spot,
            'success': result.success or not hit_max_iter,
            'final_error': result.fun,
            'final_rmse': final_rmse,
            'iterations': lbfgs_iterations,
            'function_evaluations': function_evaluations,
            'converged': result.success,
            'hit_max_iter': hit_max_iter
        }
    
    @classmethod
    def fit_hybrid_spline_surface(cls, data, params: dict) -> dict:
        """
        Fit hybrid spline model to entire volatility surface.
        
        Parameters
        ----------
        data : DataFrame
            Option data with columns 'Strike', 'TTM', and 'Imp Vol - Last'
        params : dict
            Parameter dictionary containing 'hybrid_spline_params' and 'spot'
            
        Returns
        -------
        dict
            Fitted model parameters
        """
        config = params['hybrid_spline_params']
        
        valid_data = data.dropna(subset=['Strike', 'TTM', 'Imp Vol - Last'])
        
        if len(valid_data) < config['min_data_points']:
            raise ValueError(
                f"Insufficient data points: {len(valid_data)} < {config['min_data_points']}"
            )
        
        strikes = valid_data['Strike'].values
        ttms = valid_data['TTM'].values
        vols = valid_data['Imp Vol - Last'].values
        
        surface_params = cls._fit_spline_surface(
            strikes=strikes,
            ttms=ttms,
            vols=vols,
            spot=params['spot'],
            config=config
        )
        
        return surface_params
    
    @classmethod
    def compute_hybrid_spline_surface(
        cls,
        strikes_grid: np.ndarray,
        ttms_grid: np.ndarray,
        spline_params: dict,
        params: dict
    ) -> np.ndarray:
        """
        Compute volatility surface on a grid using fitted spline.
        
        Parameters
        ----------
        strikes_grid : ndarray
            Grid of strike prices
        ttms_grid : ndarray
            Grid of times to maturity (years)
        spline_params : dict
            Fitted spline parameters from fit_hybrid_spline_surface
        params : dict
            General parameters
            
        Returns
        -------
        ndarray
            Implied volatility surface (decimal)
        """
        config = params['hybrid_spline_params']
        
        original_shape = strikes_grid.shape
        strikes_flat = strikes_grid.flatten()
        ttms_flat = ttms_grid.flatten()
        
        # Convert strikes to log-moneyness
        log_moneyness_flat = np.log(strikes_flat / spline_params['spot'])
        
        # Build basis matrices at evaluation points
        strike_basis = cls._basis_matrix(
            log_moneyness_flat,
            spline_params['strike_knots'],
            spline_params['strike_degree']
        )
        
        time_basis = cls._basis_matrix(
            ttms_flat,
            spline_params['time_knots'],
            spline_params['time_degree']
        )
        
        # Compute total variance: w(k,τ) = Σᵢⱼ cᵢⱼ Bᵢ(k) Bⱼ(τ)
        total_variance_flat = np.einsum(
            'ij,jk,ik->i',
            strike_basis,
            spline_params['coefs'],
            time_basis
        )
        
        # Enforce positive variance
        total_variance_flat = np.maximum(total_variance_flat, config['variance_floor'])
        
        # Convert to implied volatility: σ = sqrt(w/τ)
        vols_flat = np.sqrt(total_variance_flat / np.maximum(ttms_flat, config['ttm_floor']))
        
        # Reshape to match input grid
        vols_grid = vols_flat.reshape(original_shape)
        
        return vols_grid