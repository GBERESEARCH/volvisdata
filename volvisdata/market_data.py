"""
Market data import and transformation functions

"""
import copy
from datetime import date, datetime, timedelta
from io import StringIO
import time
import warnings
#import datetime as dt
from bs4 import BeautifulSoup
from dateutil import parser
import pandas as pd
# from pandas.tseries.holiday import get_calendar, HolidayCalendarFactory, GoodFriday, AbstractHolidayCalendar
from pandas.tseries.holiday import (
    USFederalHolidayCalendar, 
    HolidayCalendarFactory, 
    GoodFriday,
    AbstractHolidayCalendar
)
import yfinance as yf
from volvisdata.market_data_prep import DataPrep, UrlOpener
warnings.filterwarnings("ignore", category=DeprecationWarning)
# pylint: disable=invalid-name

class Data():
    """
    Market data import and transformation functions

    """
    @classmethod
    def create_option_data(
        cls,
        params: dict,
        tables: dict) -> tuple[dict, dict]:
        """
        Extract the URL for each of the listed option on Yahoo Finance
        for the given ticker. Extract option data from each URL.

        Filter / transform the data and calculate implied volatilities for
        specified put and call strikes.

        Parameters
        ----------
        start_date : Str
            Date from when to include prices (some of the options
            won't have traded for days / weeks and therefore will
            have stale prices).
        ticker : Str
            The ticker identifier used by Yahoo for the chosen stock. The
            default is '^SPX'.
        ticker_label : Str
            The ticker label used in charts.
        wait : Int
            Number of seconds to wait between each url query
        lastmins : Int, Optional
            Restrict to trades within number of minutes since last
            trade time recorded
        mindays : Int, Optional
            Restrict to options greater than certain option expiry
        minopts : Int, Optional
            Restrict to minimum number of options to include that
            option expiry
        volume : Int, Optional
            Restrict to minimum Volume
        openint : Int, Optional
            Restrict to minimum Open Interest
        monthlies : Bool
            Restrict expiries to only 3rd Friday of the month. Default
            is False.
        spot : Float
            Underlying reference level.
        put_strikes : List
            Range of put strikes to calculate implied volatility for.
        call_strikes : List
            Range of call strikes to calculate implied volatility for.
        strike_limits : Tuple
            min and max strikes to use expressed as a decimal
            percentage. The default is (0.5, 2.0).
        divisor : Int
            Distance between each strike in dollars. The default is 25 for SPX
            and 10 otherwise.
        r : Float
            Interest Rate. The default is 0.005.
        q : Float
            Dividend Yield. The default is 0.
        epsilon : Float
            Degree of precision to return implied vol. The default
            is 0.001.
        method : Str
            Implied Vol method; 'nr', 'bisection' or 'naive'. The
            default is 'nr'.

        Returns
        -------
        DataFrame
            DataFrame of Option data.

        """
        # Extract URLs and option data
        # params, tables = cls.extractoptions(params=params, tables=tables)
        params, tables = cls.get_option_data(params=params, tables=tables)
        print("Options data extracted")

        # Filter / transform data
        params, tables = DataPrep.transform(params=params, tables=tables)
        print("Data transformed")

        # Calculate implied volatilities and combine
        params, tables = DataPrep.combine(params=params, tables=tables)
        print("Data combined")

        return params, tables


    @staticmethod
    def process_df_option_data(params: dict, tables: dict)-> tuple[dict, dict]:
        """


        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.
        tables : dict
            Dictionary of key tables.

        Returns
        -------
        params : Dict
            Dictionary of key parameters.
        tables : dict
            Dictionary of key tables.

        """
        # Direct path for calibrated data
        df = params['precomputed_data'].copy()

        params['extracted_spot'] = df['Spot Price'].iloc[0]
        # params['start_date'] = df['Reference Date'].iloc[0]

        tables['full_data'] = df[[
            'Contract Symbol',
            'Last Price',
            'Bid',
            'Ask',
            'Last Trade Date',
            'Expiry',
            'Strike',
            'Option Type',
            'Open Interest',
            'Volume',
            'Implied Volatility',
            'Discount Rate'
        ]]

        # Process data through standard pipeline starting with transform
        params, tables = DataPrep.transform(params=params, tables=tables)
        params, tables = DataPrep.combine(params=params, tables=tables)

        return params, tables


    @staticmethod
    def get_option_data(params: dict, tables: dict) -> tuple[dict, dict]:
        """


        Parameters
        ----------
        ticker : TYPE
            DESCRIPTION.
        params : Dict
            Dictionary of key parameters.
        tables : dict
            Dictionary of key tables.

        Returns
        -------
        params : Dict
            Dictionary of key parameters.
        tables : dict
            Dictionary of key tables.

        """
        params['option_dict'] = {}
        params['opt_except_list'] = []
        tables['full_data'] = pd.DataFrame()
        asset = yf.Ticker(params['ticker'])

        try:
            extracted_spot = asset.info['currentPrice']
        except KeyError:
            try:
                extracted_spot = (asset.info['bid'] + asset.info['ask'])/2
                if (abs(extracted_spot - asset.info['previousClose'])
                    / asset.info['previousClose']) > 0.2:
                    extracted_spot = asset.info['previousClose']
            except KeyError:
                try:
                    extracted_spot = asset.info['navPrice']
                except KeyError:
                    extracted_spot = asset.info['previousClose']
        params['extracted_spot'] = extracted_spot

        opt_list = asset.options
        params['date_list'] = opt_list
        for expiry in opt_list:
            params['option_dict'][expiry] = []
            chain = asset.option_chain(expiry)
            try:
                calls = chain.calls

                # Create a column designating these as calls
                calls['Option Type'] = 'call'
                if len(calls) > 0:
                    if isinstance(calls['strike'][0], (int, float, complex)):
                        params['option_dict'][expiry].append(calls)
                        try:
                            puts = chain.puts

                            # Create a column designating these as puts
                            puts['Option Type'] = 'put'

                            if len(puts) > 0:
                                if isinstance(puts['strike'][0], (int, float, complex)):
                                    params['option_dict'][expiry].append(puts)
                                    # Concatenate these two DataFrames
                                    options = pd.concat([calls, puts])

                                    # Add an 'Expiry' column with the expiry date
                                    options['Expiry'] = pd.to_datetime(expiry).date()

                                    # Add this DataFrame to 'full_data'
                                    tables['full_data'] = pd.concat(
                                        [tables['full_data'], options])

                        except (IndexError, KeyError, ValueError):
                            # Add an 'Expiry' column with the expiry date
                            calls['Expiry'] = pd.to_datetime(expiry).date()

                            # Add this DataFrame to 'full_data'
                            tables['full_data'] = pd.concat(
                                [tables['full_data'], calls])

            except (IndexError, KeyError, ValueError):
                try:
                    # The second entry is 'puts'
                    puts = chain.puts

                    # Create a column designating these as puts
                    puts['Option Type'] = 'put'

                    if len(puts) > 0:
                        if isinstance(puts['strike'][0], (int, float, complex)):
                            params['option_dict'][expiry][1] = puts
                            # Add an 'Expiry' column with the expiry date
                            puts['Expiry'] = pd.to_datetime(expiry).date()

                            # Add this DataFrame to 'full_data'
                            tables['full_data'] = pd.concat(
                                [tables['full_data'], puts])

                except (IndexError, KeyError, ValueError):
                    params['opt_except_list'].append(expiry)

        tables['full_data'] = tables['full_data'].rename(columns={
            'lastPrice': 'Last Price',
            'bid': 'Bid',
            'ask': 'Ask',
            'lastTradeDate': 'Last Trade Date',
            'strike': 'Strike',
            'openInterest': 'Open Interest',
            'volume': 'Volume',
            'impliedVolatility': 'Implied Volatility'
            })

        return params, tables

    @classmethod
    def extractoptions(
        cls,
        params: dict,
        tables: dict) -> tuple[dict, dict]:
        """
        Extract the URL for each of the listed option on Yahoo Finance
        for the given ticker. Extract option data from each URL.


        Parameters
        ----------
        ticker : Str
            The ticker identifier used by Yahoo for the chosen stock. The
            default is '^SPX'.
        wait : Int
            Number of seconds to wait between each url query

        Returns
        -------
        DataFrame
            All option data from each of the supplied urls.

        """

        # Extract dictionary of option dates and urls
        params = cls._extracturls(params=params)
        print("URL's extracted")

        params['raw_web_data'] = cls._extract_web_data(params=params)

        params = cls._read_web_data(params=params)

        # Create an empty DataFrame
        tables['full_data'] = pd.DataFrame()

        # Make a list of all the dates of the DataFrames just stored
        # in the default dictionary
        params['date_list'] = list(params['option_dict'].keys())

        params, tables = cls._process_options(params=params, tables=tables)

        return params, tables


    @staticmethod
    def _extracturls(params: dict) -> dict:
        """
        Extract the URL for each of the listed option on Yahoo Finance
        for the given ticker.

        Parameters
        ----------
        ticker : Str
            Yahoo ticker (Reuters RIC) for the stock.

        Returns
        -------
        Dict
            Dictionary of dates and URLs.

        """

        # Define the stock root webpage
        url = 'https://finance.yahoo.com/quote/'+params['ticker']\
            +'/options?p='+params['ticker']

        # Create a UrlOpener object to extract data from the url
        urlopener = UrlOpener()
        response = urlopener.open(url)

        # Collect the text from this object
        params['html_doc'] = response.text

        # Use Beautiful Soup to parse this
        soup = BeautifulSoup(params['html_doc'], features="lxml")

        # Create a list of all the option dates
        #option_dates = [a.get_text() for a in soup.find_all('option')]
        option_dates = [a.get_text() for a in soup.findAll('div', {'role': 'option'})]

        # Convert this list from string to datetimes
        #dates_list = [dt.datetime.strptime(date, "%B %d, %Y").date() for date
        #              in option_dates]
        dates_list = []
        for option_date in option_dates:
            option_date.rstrip()
            try:
                dates_list.append(parser.parse(option_date).date())
            except (IndexError, KeyError, ValueError):
                pass

        # Convert back to strings in the required format
        str_dates = [date_obj.strftime('%Y-%m-%d') for date_obj in dates_list]

        # Create a list of all the unix dates used in the url for each
        # of these dates
        #option_pages = [a.attrs['value'] for a in soup.find_all('option')]
        raw_option_pages = [a.attrs['data-value'] for a in soup.findAll('div', {'role': 'option'})]
        option_pages = [num for num in raw_option_pages if ((len(num) == 10) and (num.isdigit()))]

        # Combine the dates and unixdates in a dictionary
        optodict = dict(zip(str_dates, option_pages))

        # Create an empty dictionary
        params['url_dict'] = {}

        # For each date and unixdate in the first dictionary
        for date_val, page in optodict.items():

            # Create an entry with the date as key and the url plus
            # unix date as value
            params['url_dict'][date_val] = str(
                'https://finance.yahoo.com/quote/'
                +params['ticker']+'/options?date='+page)

        return params


    @staticmethod
    def _extract_web_data(params: dict) -> dict:

        # Create an empty dictionary
        raw_web_data = {}

        # each url needs to have an option expiry date associated with
        # it in the url dict
        for input_date, url in params['url_dict'].items():

            # UrlOpener function downloads the data
            urlopener = UrlOpener()
            weburl = urlopener.open(url)
            try:
                raw_web_data[input_date] = weburl.text

                # wait between each query so as not to overload server
                time.sleep(params['wait'])

            # If there is a problem, report the date and apply extended wait
            except ValueError:
                print("Problem with "+input_date+" data")
                time.sleep(5)

        return raw_web_data


    @staticmethod
    def _read_web_data(params: dict) -> dict:

        # Create an empty dictionary
        params['option_dict'] = {}
        params['url_except_dict'] = {}

        for input_date, url in params['url_dict'].items():
            # if data exists
            try:
                # read html data into a DataFrame
                option_frame = pd.read_html(
                    StringIO(params['raw_web_data'][input_date]))

                # Add this DataFrame to the default dictionary, named
                # with the expiry date it refers to
                params['option_dict'][input_date] = option_frame

            # otherwise collect dictionary of exceptions
            except ValueError:
                params['url_except_dict'][input_date] = url

        return params


    @staticmethod
    def _process_options(
        params: dict,
        tables: dict) -> tuple[dict, dict]:

        # Create list to store exceptions
        params['opt_except_list'] = []

        # For each of these dates
        for input_date in params['date_list']:

            try:
                # The first entry is 'calls'
                calls = params['option_dict'][input_date][0]

                # Create a column designating these as calls
                calls['Option Type'] = 'call'

                if str(calls['Strike'][0]).isdigit():
                    try:
                        # The second entry is 'puts'
                        puts = params['option_dict'][input_date][1]

                        # Create a column designating these as puts
                        puts['Option Type'] = 'put'

                        if str(puts['Strike'][0]).isdigit():
                            # Concatenate these two DataFrames
                            options = pd.concat([calls, puts])

                            # Add an 'Expiry' column with the expiry date
                            options['Expiry'] = pd.to_datetime(input_date).date()

                            # Add this DataFrame to 'full_data'
                            tables['full_data'] = pd.concat(
                                [tables['full_data'], options])

                    except IndexError:

                        # Add an 'Expiry' column with the expiry date
                        calls['Expiry'] = pd.to_datetime(input_date).date()

                        # Add this DataFrame to 'full_data'
                        tables['full_data'] = pd.concat(
                            [tables['full_data'], calls])

            except IndexError:

                try:
                    # The second entry is 'puts'
                    puts = params['option_dict'][input_date][1]

                    # Create a column designating these as puts
                    puts['Option Type'] = 'put'

                    if str(puts['Strike'][0]).isdigit():
                        # Add an 'Expiry' column with the expiry date
                        puts['Expiry'] = pd.to_datetime(input_date).date()

                        # Add this DataFrame to 'full_data'
                        tables['full_data'] = pd.concat(
                            [tables['full_data'], puts])

                except IndexError:
                    params['opt_except_list'].append(input_date)

        return params, tables


    @staticmethod
    def trading_calendar(params: dict) -> dict:
        """
        Generate list of trading holidays

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.

        Returns
        -------
        params : Dict
            Dictionary of key parameters.

        """
        # Create calendar instance
        # cal = get_calendar('USFederalHolidayCalendar')
        # cal_mod = copy.deepcopy(cal)

        # start = date.today()
        # end = start + timedelta(days=2500)

        # # Remove Columbus Day rule and Veteran's Day rule
        # cal_mod.rules = cal_mod.rules[0:6] + cal_mod.rules[8:]

        # # Create new calendar generator
        # tradingCal = HolidayCalendarFactory(
        #     'TradingCalendar', cal_mod, GoodFriday)

        # tcal = tradingCal()

        # holiday_array = tcal.holidays(start=start, end=end).to_pydatetime()

        # params['trade_holidays'] = []
        # for hol in holiday_array:
        #     params['trade_holidays'].append(hol.date())

        # return params

         # Create a modified calendar class that excludes Columbus Day and Veteran's Day
        class CustomTradingCalendar(USFederalHolidayCalendar):
            rules = [rule for i, rule in enumerate(USFederalHolidayCalendar.rules) if i != 6 and i != 7]  # Skip indices 6 and 7
        
        # Create a calendar class for GoodFriday
        class GoodFridayCalendar(AbstractHolidayCalendar):
            rules = [GoodFriday]
        
        # Create the combined calendar class
        TradingCalendar = HolidayCalendarFactory(
            'TradingCalendar', 
            CustomTradingCalendar,  
            GoodFridayCalendar    
        )
        
        # Instantiate the calendar
        tcal = TradingCalendar()
        
        # Convert date objects to datetime for the holidays method
        start_dt = datetime.combine(date.today(), datetime.min.time())
        end_dt = datetime.combine(date.today() + timedelta(days=2500), datetime.min.time())
        
        # Get holidays
        holiday_array = tcal.holidays(start=start_dt, end=end_dt).to_pydatetime()
        
        # Store holiday dates in params
        params['trade_holidays'] = [hol.date() for hol in holiday_array]
        
        return params
    