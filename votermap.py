import pandas as pd
import numpy as np
import os
import re
import time
import requests
import warnings

import geopandas as gpd
from datashader.utils import lnglat_to_meters

import matplotlib.pyplot as plt
import pylab
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')

class VoterMapAPI:
    def __init__(self, config, state, district_type):
        self.config = config
        self.state = state.title()
        self.district_type = 'House' if district_type.title() == 'Assembly' else district_type.title()

        # determine application and datasource name
        self.app = self.get_app_name()
        self.datasource = self.get_datasource_name()

    @property
    def app_list(self):
        # access the list of available applications
        url = os.path.join(self.config['api']['base_url'],
                           "applications?id={id}&apikey={apikey}".format(**self.config['api']))

        results = requests.get(url, verify=False)
        return results.json()['applications']

    @property
    def all_columns(self):
        url = os.path.join(self.config['api']['base_url'],
                           "application/datasource/columns/{}/{}".format(self.app, self.datasource))

        results = requests.get(url, params=self.config['api'], verify=False)
        return [col['id'] for col in results.json()['columns']]

    @property
    def district_key(self):
        try:
            key = self.search_columns('{}_District'.format(self.district_type))[-1] # most recent
        except IndexError:
            key = self.search_columns('Legislative_District')[-1] # most recent
        return key


    def get_app_name(self):
        return list(filter(lambda x: x['name'] == self.state, self.app_list))[0]['appname']

    def get_datasource_name(self):
        url = os.path.join(self.config['api']['base_url'],
                           "application/datasources/{}".format(self.app))

        results = requests.get(url, params=self.config['api'], verify=False)
        return results.json()['datasources'][0]['name']

    def search_columns(self, search_term, startswith=False):
        pattern = r''+search_term
        if startswith:
            return [col for col in self.all_columns if re.match(pattern,col) is not None]

        else:
            return [col for col in self.all_columns if re.search(pattern,col) is not None]

    def get_column_values(self, column):
        url = os.path.join(self.config['api']['base_url'],
                           "application/datasource/column/values/{}/{}/{}".format(self.app, self.datasource, column))

        results = requests.get(url, params=self.config['api'], verify=False)
        return results.json()['values']

    def view_stats_url(self, view_column):
        # combine/format inputs
        view_params = dict(self.config['api'], **{"view":"view.{}=1".format(view_column),
                                    "app":self.app,
                                    "datasource":self.datasource})

        # baseline url for stats, view-only
        url = os.path.join(self.config['api']['base_url'],
                           "application/datasource/statistics",
                           "{app}/{datasource}?id={id}&apikey={apikey}&{view}".format(**view_params))
        return url

    def add_filters_to_url(self, view_url, filter_dict):

        # add each filter (column=value) to the url
        filter_url = view_url
        for filt in filter_dict.items():
            col,val = filt
            filter_url += '&filter.{}={}'.format(col,val)

        return filter_url

    def query(self, view_column, filter_dict=None, verbose=True,
                    return_dataframe=False, ignore_unknown=False):

        # build URL / view a column and get stats
        stats_url = self.view_stats_url(view_column)

        # OPTIONALLY add extra filters
        if filter_dict is not None:
            stats_url = self.add_filters_to_url(stats_url, filter_dict)

        if verbose:
            print(stats_url)
        results = requests.get(stats_url, verify=False).json()

        # the API will not return results where the count is zero. for consistency we need to keep this information.
        for val in self.get_column_values(view_column):
            if val not in results['statistics'][view_column].keys():
                results['statistics'][view_column][val] = 0

        return QueryResults(results, self.config, self.state)



class QueryResults:

    def __init__(self, results, config, state):
        self.config = config
        self.results = results
        self.state = state
        self._df = None

    def to_dataframe(self):

        if self._df is not None:
            return self._df

        # name of index
        index_col = list(self.results['statistics'].keys())[0]

        df = pd.DataFrame(list(self.results['statistics'][index_col].items()),
                          columns=[index_col, 'count'])

        df.set_index(index_col, inplace=True)
        df.sort_index(inplace=True)
        df.name = index_col
        self._df = df

        return df

    def show_counts(self, force_linear_axes=False, ignore_unknowns=False):

        df = self.to_dataframe()
        if ignore_unknowns:
            df = df.drop(['Unknown', '__unknown'], axis=0)

        fig, ax = plt.subplots(1,1, figsize=(10,6))

        ax.set_title(df.index.name)
        ax.barh(df.index, df['count'], color='#636687')

        if (np.max(df['count']+1)/np.min(df['count']+1) > 100) and not force_linear_axes:
            ax.set_xscale('log')
        return fig, ax



def cache_demographics(query_func):

    def demo_wrapper(*args, **kwargs):

        # make sure config file is passed
        if 'config' not in kwargs.keys():
            raise ValueError("You need to provide a config file!")

        # filename to where the data are / will be saved
        csv_basename = get_data_location(*args, **kwargs)
        csv_path = os.path.join(kwargs['config']['data_path'], 'queries', csv_basename)

        # if cache'ing and if file exists, read it in and return dataframe.
        # if it doesn't exist, run the query, save the results, and return the dataframe.
        if os.path.isfile(csv_path) and kwargs.pop('cache',True):
            print("Data exist locally! Reading...")
            df = pd.read_csv(csv_path)
            df.set_index('District', inplace=True)
            df.name = args[0]
            return df

        df = query_func(*args, **kwargs)
        df.name = args[0]
        df.to_csv(csv_path)

        return df

    return demo_wrapper


def get_data_location(view_column, state=None, api=None, filter_dict=None,**kwargs):
    if state is None:
        if api is None:
            raise ValueError("Provide an API, or specify the state directly.")
        state = api.state

    state = state.title()

    if filter_dict is None:
        filter_string = ''
    else:
        filter_string =  '_filters.' + '_'.join(filter_dict.keys())

    fn = '{}_{}{}.csv'.format(state, view_column, filter_string)

    return fn


@cache_demographics
def get_district_demographics(view_column, api=None, filter_dict=None,
                              num_districts=99, district_list=None,
                              verbose=False, max_retries=5, **kwargs):

    """
    kwargs:
    * config
    * api
    *
    """

    # check the inputs & deal with some defaults
    if api is None:
        raise ValueError("You need to provide an API if you want to run a query!")

    # identify the district column you need to use
    try:
        district_key = api.search_columns('{}_District'.format(api.district_type))[-1] # most recent
    except IndexError:
        district_key = api.search_columns('Legislative_District')[-1] # most recent

    if district_list is None:
        #district_list = list(range(1,num_districts+1))
        district_list = api.get_column_values(district_key)

    print("Running query on {} districts...".format(len(district_list)))
    district_dfs = []
    for district in district_list:
        if verbose:
            print(' .. filtering results for District {}'.format(district))

        # add district info to the filter set
        if filter_dict is None:
            #filter_dict = {district_key:str(district).zfill(2)}
            filter_dict = {district_key:district}

        else:
            #filter_dict = dict(filter_dict, **{district_key:str(district).zfill(2)})
            filter_dict = dict(filter_dict, **{district_key:district})

        # get stats for this district
        #results = api.query(view_column, filter_dict=filter_dict)
        for _ in range(max_retries):
            try:
                results = api.query(view_column, filter_dict=filter_dict)
                break
            except:
                print("Request timed out. Waiting 60 seconds before trying again .. ")
                time.sleep(60)

        # convert to dataframe
        df = results.to_dataframe().T

        # add district as index // append to list
        df['District'] = int(district)
        df.set_index('District', inplace=True)

        # if '__unknown' is there, it is just a duplicate of 'Unknown'
        # so we can drop it without worrying about losing information
        if '__unknown' in df.columns:
            df.drop(columns=['__unknown'], inplace=True)

        district_dfs.append(df)

    df = pd.concat(district_dfs, ignore_index=False)

    return df


def query_demographic_list(query_list, api=None, config=None):
    dfs = []
    for query_name in query_list:
        print('*** STARTING {} ***'.format(query_name.upper()))
        df = get_district_demographics(query_name, verbose=True,
                                        api=api, config=config)

        # name the multi-level columns
        multilevel_columns = [(df.name,col) for col in df.columns]

        # update column names and append
        df.columns = pd.MultiIndex.from_tuples(multilevel_columns)
        dfs.append(df)

    # concatenate output
    return pd.concat(dfs, axis=1)


def grab_dataset(mldf, query_name):
    """mldf == multi-level dataframe"""
    df = mldf[mldf.columns[mldf.columns.get_level_values(0)==query_name]]

    # put it back to one level
    df.columns = [tup[1] for tup in df.columns]
    df.name = query_name
    return df

def check_unknown_fraction(mldf, query_name):
    """mldf == multi-level dataframe"""
    df = grab_dataset(mldf, query_name)

    # check if this is even relevant
    if 'Unknown' in df.columns:
        if '__unknown' in df.columns:
            df.drop(columns=['__unknown'], inplace=True)

        f_unknown = df['Unknown'].div(df.sum(axis=1))
        f_med,f_min,f_max = f_unknown.median(), f_unknown.min(), f_unknown.max()
    else:
        f_med,f_min,f_max = 0.0, 0.0, 0.0

    return {'min':f_min, 'max':f_max, 'median':f_med}

def check_response_rates(df, query_name=None, exclude_unknowns=False, api=None, **kwargs):
    """
    df can be multi-level or not
    """

    # are we working with a combined dataframe or an individual one?
    if query_name is not None:
        # grab dataset
        df = grab_dataset(df, query_name)

    if exclude_unknowns:
        df = drop_unknowns(df)

    # total counts per district
    counts = drop_unknowns(api.query(api.district_key, verbose=False).to_dataframe())
    counts.set_index(df.index, inplace=True)
    f = df.sum(axis=1).div(counts.sum(axis=1))

    f_med,f_min,f_max = f.median(), f.min(), f.max()

    return {'min':f_min, 'max':f_max, 'median':f_med}


def compare_response_rates(mldf, query_names, exclude_unknowns=False, api=None, **kwargs):
    """mldf == multi-level dataframe"""
    all_rates = []
    for query_name in query_names:
        df = grab_dataset(mldf, query_name)
        rates = check_response_rates(df, exclude_unknowns=exclude_unknowns, api=api)
        all_rates.append(pd.DataFrame(data=[rates], index=[query_name]))

    return pd.concat(all_rates, axis=0)



def drop_unknowns(df):
    drop_columns = [s for s in ['__unknown','Unknown'] if s in df.columns]
    drop_indices = [s for s in ['__unknown','Unknown'] if s in df.index]
    new_df = df.drop(drop_columns, axis=1).drop(drop_indices, axis=0)
    new_df.name = df.name
    return new_df

def show_district_counts(df, ignore_unknowns=True, force_linear_axes=False,
                         max_ratio=100, figsize=(8,4), color='#636687', **kwargs):

    # optionally drop unknown categories
    if ignore_unknowns:
        #df = df.drop(['__unknown','Unknown'],axis=1)
        df = drop_unknowns(df)

    # sum across districts
    summed_df = df.sum(axis=0)

    fig, ax = plt.subplots(1,1, figsize=figsize)

    ax.set_title(summed_df.index.name)
    ax.barh(summed_df.index, summed_df.values, color=color)

    if (np.max(summed_df.values+1)/np.min(summed_df.values+1) > max_ratio) and not force_linear_axes:
        ax.set_xscale('log')
    return fig, ax


def shade_districts_by_demo(df, column, normalized=True, config=None, state=None, district_type=None,
                            ignore_unknowns=True, figsize=(10,6), cmap='Purples', **kwargs):

    # optionally drop unknown categories
    if ignore_unknowns:
        #df = df.drop(['__unknown','Unknown'],axis=1)
        df = drop_unknowns(df)

    if normalized:
        # use the FRACTION of each district that falls in this column
        description = 'Fraction per district'
        vals = df[column]/df.sum(axis=1)
    else:
        # use values directly
        vals = df[column]
        description = 'Counts per district'

    # scale colors so that 0-->1 runs from min() to max()
    minval, maxval = np.min(vals),np.max(vals)
    scaled_values = (vals - minval)/(maxval - minval)

    # shape files and coordinate limits
    district_shapes = gpd.read_file(os.path.join(config['border_path'],
                                                 config['{}_districts'.format(
                                                     district_type.lower())][state]['shape_file'])
                                   )
    district_shapes = district_shapes.to_crs("EPSG:3857")

    lon_range = tuple(config['{}_districts'.format(district_type.lower())][state]['lon_range'])
    lat_range = tuple(config['{}_districts'.format(district_type.lower())][state]['lat_range'])

    x_range, y_range = [list(r) for r in lnglat_to_meters(lon_range, lat_range)]


    # set up figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle('{}: {}'.format(df.name, column))

    spec = gridspec.GridSpec(nrows=7, ncols=9)
    cmap = pylab.cm.get_cmap(cmap)

    # left side: district shapes, color coded by the column we care about.
    ax_shapes = fig.add_subplot(spec[:, :5])
    district_shapes.boundary.plot(linewidth=0.25, edgecolor='white', facecolor=cmap(scaled_values),
                                  alpha=1, ax=ax_shapes)
    ax_shapes.set_xlim(x_range)
    ax_shapes.set_ylim(y_range)
    ax_shapes.set_axis_off()


    # right top:
    ax_districts = fig.add_subplot(spec[1:3, 5:])
    ax_districts.scatter(df.index, vals, c=scaled_values, cmap=cmap, s=40)
    ax_districts.set_xlabel('Districts')
    ax_districts.set_ylabel('{}:\n{}'.format(description, column))

    # right bottom:
    ax_hist = fig.add_subplot(spec[4:6, 5:])
    ax_hist.hist(vals, color='#5A565F')
    ax_hist.set_xlabel('{}: {}'.format(description, column))
    ax_hist.set_ylabel('Number')

    return fig


def plot_demo_vs_margins(df_demo, margins, config=None, normalized=True, ignore_unknowns=True,
                         colorby=None, colorlabel=None, color='#745C92', ncols=3, cmap='RdBu',
                         figwidth=10, ms=30, margin_absmax=80, **kwargs):
    # one panel per demographic category
    # but only one demographic at a time (e.g. agebins OR gender, not both)

    # optionally drop unknown categories
    if ignore_unknowns:
        #df_demo = df_demo.drop(['__unknown','Unknown'],axis=1)
        df_demo = drop_unknowns(df_demo)

    demo_name = df_demo.name
    demo_bins = list(df_demo.columns)

    # check margins. are these fractions or percentage points?
    if np.max(margins) <= 1.0:
        margin_vals = 100* margins
    else:
        margin_vals = margins


    # election results or LDI?
    if 'LDI' in margins.name.upper():
        margin_type = 'LDI'
    else:
        margin_type = 'Election Results'

    # set up grid
    nrows = int(np.ceil(len(demo_bins) / ncols))
    spec = gridspec.GridSpec(nrows=nrows, ncols=ncols)

    # set up fig
    fig = plt.figure(figsize=(figwidth, figwidth*(nrows/ncols)))
    fig.suptitle('{} vs {}'.format(margin_type, demo_name))

    for i in range(len(demo_bins)):
        row = int(i/ncols)
        col = i % ncols
        dname = demo_bins[i]

        if normalized:
            demo_vals = df_demo[dname]/df_demo.sum(axis=1)
            description = 'Fraction per district'
        else:
            demo_vals = df_demo[dname]
            description = 'Number per district'

        ax = fig.add_subplot(spec[row, col])
        ax.set_xlabel('{}: {}'.format(description, dname))
        ax.set_ylabel('Democratic margin [%]')
        if colorby is None:
            ax.scatter(demo_vals, margin_vals, s=ms, color=color)
        else:
            p = ax.scatter(demo_vals, margin_vals, s=ms, c=colorby, cmap=cmap,
                           vmin=-margin_absmax, vmax=margin_absmax)


    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    if colorby is not None:
        fig.subplots_adjust(bottom=0.2)
        colorleg = fig.add_axes([0.3,0.03,0.4,0.02])

        if colorlabel is None:
            colorlabel = colorby.name

        fig.colorbar(p, cax=colorleg, orientation='horizontal', label=colorlabel)

    return fig


def sort_dollars(df):
    if df.name == 'EstHomeValueBin':
        # strip off the $ and K (i.e. $10K --> 10)
        int_dollars = [int(s.split()[0][1:-1]) for s in df.columns]
        return df[df.columns[np.argsort(int_dollars)]]

    elif df.name == 'EstimatedIncome':
        int_dollars = [int(s.split()[0].split('-')[0][1:].replace(',','')) for s in df.columns]
        return df[df.columns[np.argsort(int_dollars)]]
    else:
        return df


def show_district_heatmap(dfs, state=None, district_type=None, config=None, normalized=True, ignore_unknowns=True,
                          colorby=None, colorlabel=None, cmap='cividis', figsize=(10,4), **kwargs):

    # accepts regular plotting kwargs too!!

    # simplify types here (dfs can be a list of dataframes or a single dataframe)
    if type(dfs) is not list:
        dfs = [dfs]

    # optionally drop unknown categories
    if ignore_unknowns:
        dfs = [drop_unknowns(df) for df in dfs]

    # sort such that columns containing dollar values appear in increasing order
    dfs = [sort_dollars(df) for df in dfs]

    # optionally normalize (values per district are fractions, rather than counts)
    if normalized:
        dfs = [df.div(df.sum(axis=1), axis=0) for df in dfs]

    # set up figure
    fig, ax = plt.subplots(1,1, figsize=figsize)
    fig.suptitle('{} {} District Demographics'.format(state.title(), district_type.title()))
    sns.heatmap(pd.concat(dfs, axis=1), ax=ax, cmap=cmap, **kwargs)

    return fig

def show_aggregated_correlations(dfs, plot_type='triangle',state=None, district_type=None, config=None,
                                 margins=None, colorby=None, colorlabel=None,
                                 cmap='PuOr_r', figsize=(6,6), **kwargs):


    #type == 'triangle' --> everything vs everything, 'line' --> everything vs one thing
    # accepts regular plotting kwargs too!!

    # simplify types here (dfs can be a list of dataframes or a single dataframe)
    if type(dfs) is not list:
        dfs = [dfs]

    # REQUIRED: drop unknown categories
    dfs = [drop_unknowns(df) for df in dfs]

    # sort such that columns containing dollar values appear in increasing order
    dfs = [sort_dollars(df) for df in dfs]

    # REQUIRED: normalize values! (ie measure correlation using values per district are fractions, rather than counts)
    dfs = [df.div(df.sum(axis=1), axis=0) for df in dfs]

    # add margins in there if they exist
    if margins is not None:
        dfs.append(margins)

    # measure correlation!
    corr = pd.concat(dfs, axis=1).corr()

    # set up figure
    fig, ax = plt.subplots(1,1, figsize=figsize)
    fig.suptitle('{} {} District Demographics'.format(state.title(), district_type.title()))

    # triangle plot?
    if plot_type == 'triangle':
        # show triangular correlation map
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, square=True, ax=ax, cmap=cmap)

    if plot_type in ['line']:
        # make sure margins were provided
        if margins is None:
            raise ValueError("Bar and Line correlation plots require margins != None")

        # extract correlations of everything-VS-margins
        single_corr = corr.loc[margins.name, corr.columns != margins.name]

        if plot_type == 'line':
            ax.plot(single_corr,'-o', color='k')
            ax.tick_params(axis='x', labelrotation=90)
            ax.set_ylabel('Correlation with Democratic Margin')


    return fig
