"""
Methods for producing skew reports

"""
from decimal import Decimal
import datetime as dt
from dateutil.relativedelta import relativedelta
from volvisdata.vol_methods import VolMethods
# pylint: disable=invalid-name, consider-using-f-string

class SkewReport():
    """
    Produce skew report
    """
    @classmethod
    def create_vol_dict(
        cls,
        params: dict,
        surface_models: dict) -> dict:
        """
        Create dictionary of implied vols by tenor and strike to use in skew
        report

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.
        surface_models : Dict
            Dictionary of vol surfaces.

        Returns
        -------
        vol_dict : Dict
            Dictionary of implied vols.

        """
        vol_dict = {}
        start_date = dt.datetime.strptime(params['start_date'], '%Y-%m-%d')
        for month in range(1, params['skew_months']+1):
            for strike in [80, 90, 100, 110, 120]:
                maturity = dt.datetime.strftime(
                    start_date + relativedelta(months=month), '%Y-%m-%d')
                vol_dict[(month, strike)] = VolMethods.get_vol(
                    maturity=maturity, strike=strike, params=params,
                    surface_models=surface_models)

        return vol_dict


    @classmethod
    def print_skew_report(
        cls,
        vol_dict: dict,
        params: dict) -> None:
        """
        Print a report showing implied vols for 80%, 90% and ATM strikes and
        selected tenor length

        Parameters
        ----------
        vol_dict : Dict
            Dictionary of implied vols.
        params : Dict
            Dictionary of key parameters.

        Returns
        -------
        Prints the report to the console.

        """
        # Set decimal format
        dp2 = Decimal(10) ** -2  # (equivalent to Decimal '0.01')

        if params['skew_direction'] == 'full':
            cls._full_skew(vol_dict=vol_dict, params=params, dp2=dp2)
        else:
            cls._header(params=params)

            if params['skew_direction'] == 'up':
                cls._upside_skew(vol_dict=vol_dict, params=params, dp2=dp2)

            else:
                cls._downside_skew(vol_dict=vol_dict, params=params, dp2=dp2)


    @staticmethod
    def _header(params: dict) -> None:

        print('='*78)
        print(': {:^74} :'.format('Skew Summary'))
        print('-'*78)

        # Contract traded on left and period covered on right
        # print(f": Underlying Ticker : {params['ticker_label']:<19}{'Close of Business Date'} : {params['start_date']} :")
        print(': Underlying Ticker : {:<19}{} : {} :'.format(
            params['ticker_label'],
            'Close of Business Date',
            params['start_date']))
        print('-'*78)

        # Strike and skew headers
        print(': {:^12} :{:^34} : {:^23} :'.format(
            'Maturity',
            'Strike',
            'Skew'))
        print('-'*78)

        if params['skew_direction'] == 'up':

            print(': {:>15}{:>7}   : {:>7}   : {:>7}   : {:>10}'\
                  ' : {:>10} :'.format(
                ': ',
                'ATM',
                '110%',
                '120%',
                '+10% Skew',
                '+20% Skew'))

        if params['skew_direction'] == 'down':
            print(': {:>15}{:>7}   : {:>7}   : {:>7}   : {:>10}'\
                  ' : {:>10} :'.format(
                ': ',
                '80%',
                '90%',
                'ATM',
                '-10% Skew',
                '-20% Skew'))


    @staticmethod
    def _downside_skew(
        vol_dict: dict,
        params: dict,
        dp2: Decimal) -> None:

        # Monthly skew summary for selected number of months
        for month in range(1, params['skew_months'] + 1):
            if month < 10:
                month_label = ' '+str(month)
            else:
                month_label = str(month)
            print(': {} Month Vol : {:>7}   : {:>7}   : {:>7}   : {:>7}'\
                  '    : {:>7}    :'.format(
                month_label,
                Decimal(vol_dict[(month, 80)]).quantize(dp2),
                Decimal(vol_dict[(month, 90)]).quantize(dp2),
                Decimal(vol_dict[(month, 100)]).quantize(dp2),
                Decimal((vol_dict[(month, 90)]
                         - vol_dict[(month, 100)]) / 10).quantize(dp2),
                Decimal((vol_dict[(month, 80)]
                         - vol_dict[(month, 100)]) / 20).quantize(dp2)))

        print('-'*78)
        print('='*78)


    @staticmethod
    def _upside_skew(
        vol_dict: dict,
        params: dict,
        dp2: Decimal) -> None:

        # Monthly skew summary for selected number of months
        for month in range(1, params['skew_months'] + 1):
            if month < 10:
                month_label = ' '+str(month)
            else:
                month_label = str(month)
            print(': {} Month Vol : {:>7}   : {:>7}   : {:>7}   : {:>7}'\
                  '    : {:>7}    :'.format(
                month_label,
                Decimal(vol_dict[(month, 100)]).quantize(dp2),
                Decimal(vol_dict[(month, 110)]).quantize(dp2),
                Decimal(vol_dict[(month, 120)]).quantize(dp2),
                Decimal((vol_dict[(month, 110)]
                         - vol_dict[(month, 100)]) / 10).quantize(dp2),
                Decimal((vol_dict[(month, 120)]
                         - vol_dict[(month, 100)]) / 20).quantize(dp2)))

        print('-'*78)
        print('='*78)


    @staticmethod
    def _full_skew(
        vol_dict: dict,
        params: dict,
        dp2: Decimal) -> None:

        print('='*115)
        print(': {:^111} :'.format('Skew Summary'))
        print('-'*115)

        # Contract traded on left and period covered on right
        print(': Underlying Ticker : {:<56}{} : {} :'.format(
            params['ticker_label'],
            'Close of Business Date',
            params['start_date']))
        print('-'*115)

        # Strike and skew headers
        print(': {:^13} : {:^47} : {:^45} :'.format(
            'Maturity',
            'Strike',
            'Skew'))
        print('-'*115)

        # Header rows
        print(': {:>16}{:>6}  : {:>6}  : {:>6}  : {:>6}  : {:>6}  : {:>9}'\
              ' : {:>9} : {:>9} : {:>9} :'.format(
            ': ',
            '80%',
            '90%',
            'ATM',
            '110%',
            '120%',
            '-20% Skew',
            '-10% Skew',
            '+10% Skew',
            '+20% Skew'))

        # Set decimal format
        dp2 = Decimal(10) ** -2  # (equivalent to Decimal '0.01')

        # Monthly skew summary for selected number of months
        for month in range(1, params['skew_months'] + 1):
            if month < 10:
                month_label = ' '+str(month)
            else:
                month_label = str(month)
            print(': {} Month Vol  : {:>6}  : {:>6}  : {:>6}  : {:>6}  : '\
                  '{:>6}  : {:>7}   : {:>7}   : {:>7}   : {:>7}   :'.format(
                month_label,
                Decimal(vol_dict[(month, 80)]).quantize(dp2),
                Decimal(vol_dict[(month, 90)]).quantize(dp2),
                Decimal(vol_dict[(month, 100)]).quantize(dp2),
                Decimal(vol_dict[(month, 110)]).quantize(dp2),
                Decimal(vol_dict[(month, 120)]).quantize(dp2),
                Decimal((vol_dict[(month, 80)]
                         - vol_dict[(month, 100)]) / 20).quantize(dp2),
                Decimal((vol_dict[(month, 90)]
                         - vol_dict[(month, 100)]) / 10).quantize(dp2),
                Decimal((vol_dict[(month, 110)]
                         - vol_dict[(month, 100)]) / 10).quantize(dp2),
                Decimal((vol_dict[(month, 120)]
                         - vol_dict[(month, 100)]) / 20).quantize(dp2)))

        print('-'*115)
        print('='*115)
