"""
Volatility Surface Calibration Parameters

Configuration parameters for SVI model calibration and surface visualization.
"""
import numpy as np

# Dictionary of default parameters
vol_params_dict = {
    # ============================================================================
    # DATA EXTRACTION PARAMETERS
    # ============================================================================
    'ticker': '^SPX',
    'ticker_label': None,
    'start_date': None,
    'wait': 2,
    'minopts': 4,
    'mindays': None,
    'lastmins': None,
    'volume': None,
    'openint': None,
    'monthlies': True,
    'divisor': None,
    'divisor_SPX': 25,
    'spot': None,
    'strike_limits': (0.5, 2.0),
    'put_strikes': None,
    'call_strikes': None,
    
    # ============================================================================
    # PRICING PARAMETERS
    # ============================================================================
    'r': 0.005,
    'q': 0,
    'epsilon': 0.001,
    'method': 'gauss',
    'discount_type': 'smooth',
    'yield_curve': None,
    
    # ============================================================================
    # VISUALIZATION PARAMETERS
    # ============================================================================
    'graphtype': 'line',
    'surfacetype': 'mesh', # Options: 'mesh', 'smoothed', 'svi', 'ssvi'
    'smoothing': False,
    #'smooth_type_svi': True,
    'smooth_type': 'svijw', # Options: 'svi', 'svijw', 'rbf'
    'scatter': True,
    'voltype': 'last',
    'smoothopt': 6,
    'notebook': False,
    'data_output': False,
    'show_graph': True,
    'order': 3,
    'spacegrain': 100,
    'azim': -50,
    'elev': 20,
    'fig_size': (15, 12),
    'rbffunc': 'thin_plate',
    'colorscale': 'Jet',
    'opacity': 0.8,
    'surf': True,
    'save_image': False,
    'image_folder': 'images',
    'image_filename': 'impvol',
    'image_dpi': 50,
    'skew_months': 12,
    'skew_direction': 'downside',
    
    # # ============================================================================
    # # SVI MODEL PARAMETERS - INITIALIZATION
    # # ============================================================================
    # 'svi_compute_initial': True,
    # 'svi_a_init': 0.04,
    # 'svi_b_init': 0.04,
    # 'svi_rho_init': 0.0,
    # 'svi_m_init': 0.0,
    # 'svi_sigma_init': 0.1,
    # 'svi_b_init_divisor': 2.0,
    
    # # ============================================================================
    # # SVI MODEL PARAMETERS - BOUNDS
    # # ============================================================================
    # 'svi_bounds': [
    #     (0.0, None),       # a: variance level
    #     (0.0001, 3.0),     # b: asymptote slope
    #     (-0.95, 0.95),     # rho: skew
    #     (-2.0, 2.0),       # m: horizontal translation
    #     (0.0001, 6.0)      # sigma: ATM curvature
    # ],
    
    # # ============================================================================
    # # SVI MODEL PARAMETERS - OPTIMIZATION
    # # ============================================================================
    # 'svi_max_iter': 1000,
    # 'svi_tol': 1e-6,
    
    # # ============================================================================
    # # SVI MODEL PARAMETERS - WEIGHTING
    # # ============================================================================
    # 'svi_atm_weight_short': 4.0,
    # 'svi_atm_weight_medium': 6.0,
    # 'svi_atm_weight_long': 8.0,
    # 'svi_short_tenor_threshold': 0.25,
    # 'svi_medium_tenor_threshold': 0.5,
    
    # 'svi_enhanced_wing_weight': False,
    # 'svi_wing_boost_factor': 2.0,
    # 'svi_wing_threshold': 0.1,
    
    # # ============================================================================
    # # SVI MODEL PARAMETERS - REGULARIZATION
    # # ============================================================================
    # 'svi_reg_weight': 0.001,
    # 'svi_reg_multiplier_short': 0.3,
    # 'svi_reg_multiplier_medium': 0.5,
    # 'svi_reg_multiplier_long': 1.0,
    # 'svi_b_penalty': 0.1,
    # 'svi_sigma_penalty': 0.1,
    # 'svi_rho_penalty': 0.05,
    # 'svi_term_reg_weight': 0.5,
    
    # # ============================================================================
    # # SVI MODEL PARAMETERS - ARBITRAGE CONSTRAINTS
    # # ============================================================================
    # 'svi_enforce_arbitrage_free': True,
    # 'svi_butterfly_penalty_weight': 100.0,
    # 'svi_calendar_penalty_weight': 1000.0,
    # 'svi_butterfly_tolerance': 1.5e-3,
    # 'svi_calendar_tolerance': 1.5e-3,
    
    # # ============================================================================
    # # SVI MODEL PARAMETERS - TERM STRUCTURE
    # # ============================================================================
    # 'svi_term_smoothing': True,
    # 'svi_term_smoothing_weight': 0.1, # Was 0.02,
    # 'svi_term_smoothing_params': ['b', 'rho', 'm', 'sigma'],
    
    # # ============================================================================
    # # SVI MODEL PARAMETERS - INTERPOLATION
    # # ============================================================================
    # 'svi_interpolation_method': 'pchip',
    
    # # ============================================================================
    # # SVI MODEL PARAMETERS - VALIDATION
    # # ============================================================================
    # 'svi_validate_surface': True,
    # 'svi_validation_strikes': 50,
    # 'svi_validation_moneyness_range': (-0.5, 0.5),
    
    # # ============================================================================
    # # SVI MODEL PARAMETERS - OUTPUT
    # # ============================================================================
    # 'svi_verbose': True,
    # 'svi_verbose_level': 2,
    
    # # ============================================================================
    # # LEGACY PARAMETERS
    # # ============================================================================
    # 'svi_joint_calibration': False,
    # 'svi_term_reg_weight': 0.0,
    # 'svi_a_bump_init': 0.0,
    # 'svi_w_bump_init': 0.1,
    # 'svi_bump_reg_weight': 0.0,
    
    # # ============================================================================
    # # SSVI PARAMETERS
    # # ============================================================================

    # # Output control
    # 'ssvi_verbose': True,
    # 'ssvi_print_max_violations': 5,

    # # Theta curve bounds (θ(t) = a + b*t^c)
    # 'ssvi_theta_a_min': 1e-6,
    # 'ssvi_theta_a_max': 0.1,
    # 'ssvi_theta_b_min': 1e-6,
    # 'ssvi_theta_b_max': 1.0,
    # 'ssvi_theta_c_min': 0.3,
    # 'ssvi_theta_c_max': 2.0,
    # 'ssvi_theta_min': 1e-6,

    # # Smile parameter bounds
    # 'ssvi_rho_inf_min': -0.95,
    # 'ssvi_rho_inf_max': 0.95,
    # 'ssvi_rho_power_min': 0.05,
    # 'ssvi_rho_power_max': 2.0,
    # 'ssvi_phi_0_min': 0.01,
    # 'ssvi_phi_0_max': 3.0,
    # 'ssvi_phi_power_min': 0.1,
    # 'ssvi_phi_power_max': 0.8,

    # # Regularization
    # 'ssvi_reg_weight': 0.001,
    # 'ssvi_reg_rho_inf_penalty': 0.1,
    # 'ssvi_reg_phi_0_center': 0.5,
    # 'ssvi_reg_phi_0_penalty': 0.1,
    # 'ssvi_reg_rho_power_center': 0.5,
    # 'ssvi_reg_rho_power_penalty': 0.05,
    # 'ssvi_reg_phi_power_center': 0.4,
    # 'ssvi_reg_phi_power_penalty': 0.05,

    # # Arbitrage enforcement
    # 'ssvi_enforce_arbitrage_free': True,
    # 'ssvi_butterfly_penalty_weight': 100.0,
    # 'ssvi_calendar_penalty_weight': 1000.0,
    # 'ssvi_butterfly_tolerance': 1.5e-3,
    # 'ssvi_calendar_tolerance': 1.5e-3,

    # # Validation
    # 'ssvi_validate_surface': True,
    # 'ssvi_validation_strikes': 50,
    # 'ssvi_validation_moneyness_min': -0.5,
    # 'ssvi_validation_moneyness_max': 0.5,

    # # Differential evolution optimizer
    # 'ssvi_de_maxiter': 300,
    # 'ssvi_de_popsize': 15,
    # 'ssvi_de_seed': 42,
    # 'ssvi_de_atol': 1e-6,
    # 'ssvi_de_tol': 1e-6,
    # 'ssvi_de_workers': 1,

    # # L-BFGS-B optimizer
    # 'ssvi_lbfgsb_maxiter': 1000,
    # 'ssvi_lbfgsb_ftol': 1e-6,

    # # ATM volatility estimation
    # 'ssvi_atm_window': 3,

    # ===== SVI PARAMETERS =====
    'svi_config_params': {
        'compute_initial': True,
        'a_init': 0.04,
        'b_init': 0.04,
        'rho_init': 0.0,
        'm_init': 0.0,
        'sigma_init': 0.1,
        'bounds': [
            (0.0, None),
            (0.0001, 0.5),
            (-0.9, 0.9),
            (-0.5, 0.5),
            (0.0001, 0.5)
        ],
        'short_dated_threshold_days': 60,
        'short_dated_b_max': 1.2,
        'short_dated_rho_min': -0.95,
        'short_dated_rho_max': 0.95,
        'max_iter': 3000,
        'tol': 1e-6,
        'reg_weight': 0.01,
        'term_reg_weight': 0.5,
        'interpolation_method': 'pchip',
        'joint_calibration': True,
        'verbose': True,
        'verbose_level': 2,
    },

    # ===== SVIJW (SVI Jump-Wings) PARAMETERS =====
    'svijw_config_params': {
        # Parameter initialization
        'atm_variance_factor': 1.0,
        'initial_params': [0.1, 2.5, 1.5],  # [psi_t, pt, ct]
        'min_variance_factor': 0.8,
        
        # Parameter bounds: [vt, psi_t, pt, ct, evt]
        'bounds': [
            (0.0001, 2.0),
            (-1.0, 1.0),
            (0.01, 3.99),
            (0.01, 3.99),
            (0.0001, 2.0)
        ],
        
        # Short-dated maturity overrides
        'short_dated': {
            'threshold_days': 30,
            'psi_t_bounds': (0.0, 0.5),
            'evt_floor_factor': 0.95,
            'beta_squared_min': 1.005
        },
        
        # Optimization
        'method': 'SLSQP',
        'max_iter': 1000,
        'ftol': 1e-8,
        'eps': 1e-10,
        'beta_squared_min': 1.01,
        
        # Calibration strategy
        'use_two_stage': False,
        
        # Term structure regularization
        'term_structure_weight': 0.05,  # Increased from 0.0005 for smoother surfaces
        'smoothness_weights': np.array([1.0, 0.5, 1.0, 1.0, 0.5]),
        
        # Two-stage calibration (only if use_two_stage=True)
        'interpolation_method': 'pchip',
        'outlier_threshold': 2.0,
        'outlier_handling': 'exclude',
        
        # Output control
        'verbose': True,
        'print_frequency': 100,
        'calculate_rmse': True,
        'calculate_mae': True
    },

    # ============================================================================
    # HYBRID SPLINE MODEL PARAMETERS
    # ============================================================================

    'hybrid_spline_params': {
        # Knot configuration
        'n_strike_knots': 5,                   # Number of interior strike knots
        'n_time_knots': 4,                     # Number of interior time knots
        'adaptive_knots': True,                # Use data-density-based knot placement
        
        # Polynomial degrees
        'strike_degree': 3,                    # Cubic B-splines in strike dimension
        'time_degree': 3,                      # Cubic B-splines in time dimension
        
        # Arbitrage constraint parameters
        'calendar_penalty_weight': 10000.0,     # Weight for calendar arbitrage penalty 1000
        'butterfly_penalty_weight': 1000.0,     # Weight for butterfly arbitrage penalty 100
        'calendar_tolerance': 1e-4,            # Tolerance for calendar constraint
        'butterfly_tolerance': 1e-3,           # Tolerance for butterfly constraint
        'n_constraint_points': 30,             # Grid size for constraint evaluation
        
        # Regularization
        'smoothness_weight': 0.1,             # Weight for smoothness penalty 0.01
        'smoothness_eval_points': 20,          # Grid size for smoothness penalty evaluation
        
        # Optimization
        'max_iter': 2000,                      # Maximum L-BFGS-B iterations
        'maxfun_multiplier': 50,               # Function evaluation limit = max_iter * maxfun_multiplier
        'ftol': 1e-6,                          # Function tolerance for convergence
        'initial_coef_noise': 0.01,            # Random perturbation magnitude for coefficient initialization
        
        # Numerical stability
        'variance_floor': 1e-8,                # Minimum total variance for numerical stability
        'ttm_floor': 1e-8,                     # Minimum time to maturity for numerical stability
        
        # Data requirements
        'min_data_points': 20,                 # Minimum data points required for calibration
        
        # Output control
        'verbose': True,                       # Print calibration diagnostics
        'progress_interval': 1000,             # Print RMSE every N function evaluations (0 = no output)
    },

    # ============================================================================
    # FIELD MAPPING DICTIONARIES
    # ============================================================================
    'vols_dict': {
        'bid': 'Imp Vol - Bid',
        'mid': 'Imp Vol - Mid',
        'ask': 'Imp Vol - Ask',
        'last': 'Imp Vol - Last'
    },
    
    'prices_dict': {
        'bid': 'Bid',
        'mid': 'Mid',
        'ask': 'Ask',
        'last': 'Last Price'
    },
    
    'row_dict': {
        'Bid': 'Imp Vol - Bid',
        'Mid': 'Imp Vol - Mid',
        'Ask': 'Imp Vol - Ask',
        'Last Price': 'Imp Vol - Last'
    },
    
    'method_dict': {
        'nr': 'implied_vol_newton_raphson',
        'gauss':'implied_vol_inverse_gauss',
        'jaeckel':'implied_vol_jaeckel',
        'bisection': 'implied_vol_bisection',
        'naive': 'implied_vol_naive'
    },
    
    'ir_tenor_dict': {
        '1 Mo': 30,
        '2 Mo': 60,
        '3 Mo': 90,
        '6 Mo': 180,
        '1 Yr': 365,
        '2 Yr': 730,
        '3 Yr': 1095,
        '5 Yr': 1826,
        '7 Yr': 2556,
        '10 Yr': 3652,
        '20 Yr': 7305,
        '30 Yr': 10952
    },
    
    # ============================================================================
    # MATPLOTLIB PARAMETERS
    # ============================================================================
    'mpl_line_params': {
        'axes.edgecolor': 'black',
        'axes.titlepad': 20,
        'axes.xmargin': 0.05,
        'axes.ymargin': 0.05,
        'axes.linewidth': 2,
        'axes.facecolor': (0.8, 0.8, 0.9, 0.5),
        'xtick.major.pad': 10,
        'ytick.major.pad': 10,
        'lines.linewidth': 3.0,
        'grid.color': 'black',
        'grid.linestyle': ':'
    },
    
    'mpl_3D_params': {
        'axes.facecolor': 'w',
        'axes.labelcolor': 'k',
        'axes.edgecolor': 'w',
        'lines.linewidth': 0.5,
        'xtick.labelbottom': True,
        'ytick.labelleft': True
    },
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 "
    "Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 '
    'Safari/537.36'
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 "
    "Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 "
    "Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
]