import pandas as pd
import numpy as np
import os
import re

import random
import scipy.stats as st

import geopandas as gpd
from datashader.utils import lnglat_to_meters

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

viz_style = {
    'font.family': 'sans-serif',
    'font.size':11,
    'axes.titlesize':'large',
    'axes.labelsize':'medium',
    'xtick.labelsize':'small',
    'ytick.labelsize':'small',
    'text.color':'#5B5654',
    'axes.labelcolor':'#5B5654',
    'xtick.color':'#5B5654',
    'ytick.color':'#5B5654',
    'axes.edgecolor':'#5B5654',
    'xtick.top':False,
    'ytick.right':False,
    'axes.spines.top':False,
    'axes.spines.right':False,
    'axes.grid':False,
    'boxplot.showfliers':False,
    'boxplot.patchartist':True
}

plt.style.use(viz_style)

class ElectionResults:
    def __init__(self, config, state, district_type, kos_skiprows=0, excel_header=[0,1]):
        self.config = config
        self.state = state
        self.district_type = 'House' if district_type.title() == 'Assembly' else district_type.title()

        self.data = self.init_elections_data(excel_header, kos_skiprows)
        self.statewide_margins, self.D_votes, self.R_votes, self.n_votes = self.calc_sort_statewide_margins()
        self.margin_trends, self.D_votes_trends, self.R_votes_trends = self.calc_kos_election_trends()


    def recalculate_margins(self):
        self.statewide_margins, self.D_votes, self.R_votes, self.n_votes = self.calc_sort_statewide_margins()
        self.margin_trends, self.D_votes_trends, self.R_votes_trends = self.calc_kos_election_trends()


    def init_elections_data(self, excel_header, kos_skiprows):
        # read state-wide results -- i.e., Daily Kos Election results by LD
        dat = pd.read_excel(os.path.join(self.config['data_path'],
                                         self.config['state_wide']['kos_filename']),
                            sheet_name=self.config['{}_districts'.format(
                                self.district_type.lower())][self.state]['sheet_name'],
                            header=excel_header, skiprows=kos_skiprows
                           )

        # drop empty columns (used to create readable space in the spreadsheet)
        dat.drop([mi for mi in dat.columns if mi[1][-2:] == ".1"], axis=1, inplace=True)

        # also drop the last row because it's just totals
        last_row_index = dat.index[-1]
        dat.drop(index=last_row_index, axis=0, inplace=True)

        # set index to reflect the district
        index_name = tuple(self.config['{}_districts'.format(self.district_type.lower())][self.state]['kos_index'])
        dat.set_index(index_name, inplace=True)
        dat.index.rename('District', inplace=True)

        db = {'state_wide': dat}
        return db


    def add_new_statewide_data(self, add_data):
        self.data['state_wide'] = pd.concat([self.data['state_wide'], add_data], axis=1)
        self.recalculate_margins()

    def add_to_database(self, add_data=None, label=None, index=None):

        # if add_data is a file, then read in new data
        if type(add_data) == str:
            add_data = pd.read_excel(os.path.join(self.config['data_path'],
                                                  self.config['{}_districts'.format(
                                                      self.district_type.lower())][self.state][add_data])
                                    )

        # set index to reflect the district
        # if string, set it directly and auto rename to 'District'
        # if tuple, the first is the column to set as the index
        #           and the second is whatever you want to rename it to
        #           (e.g. for consistency with other datasets)
        if index is not None:
            if isinstance(index, str):
                add_data.set_index(index, inplace=True)
                add_data.index.rename('District', inplace=True)
            else:
                add_data.set_index(index[0], inplace=True)
                add_data.index.rename(index[1], inplace=True)

        self.data[label] = add_data

    def calc_sort_statewide_margins(self):

        # get set of elections in the spreadsheet (ignore duplicates and the Assembly summary columns)
        # second check is to eliminate date ranges, which exist in the general headers of some states
        elections = [s for s in set(self.data['state_wide'].columns.get_level_values(0))
                     if re.search(r'\d{4}', s) and re.search(r'\d{4}-\d{4}', s) is None]

        # store this information
        self.elections = elections

        # calculate margins for each election & keep track of their years
        pattern = r'(?P<yy>\d{4}) (?P<elect>\w+[\s\w\W]+)'

        statewide_margins = {}
        dem_votes = {}
        rep_votes = {}
        n_votes = {}
        election_yrs, election_types = [],[]
        for election in elections:
            # identify the democratic and republican candidates in this election
            d_name, r_name = [mi for mi in self.data['state_wide'].columns if election in mi and mi[1][-1]=='%']
            tot_col = [mi for mi in self.data['state_wide'].columns if election in mi and 'Total' in mi][0]

            # save democratic and republican votes (percentages)
            dem_votes[election] = 100*self.data['state_wide'][d_name]
            rep_votes[election] = 100*self.data['state_wide'][r_name]
            n_votes[election] = self.data['state_wide'][tot_col]

            #print(d_name)
            #print(tot_col)
            #print('\n\n\n')

            #print(dem_votes[election])
            #print('\n\n')
            #print(n_votes[election])

            # calculate the margin by which the democratic candidate won or lost
            # and convert units from fraction to percentage-points
            margin = 100*self.data['state_wide'][d_name].sub(self.data['state_wide'][r_name])

            statewide_margins[election] = margin

            # find the year and type of election
            grps = re.search(pattern, election)
            yr, election_type = grps.groups()
            election_yrs.append(int(yr))
            election_types.append(election_type)

        # convert to dataframes, set index labels appropriately
        df_marg = pd.DataFrame(data=statewide_margins)
        df_marg.index.rename('District', inplace=True)
        df_d = pd.DataFrame(data=dem_votes)
        df_d.index.rename('District', inplace=True)
        df_r = pd.DataFrame(data=rep_votes)
        df_r.index.rename('District', inplace=True)
        df_n = pd.DataFrame(data=n_votes)
        df_n.index.rename('District', inplace=True)

        # sort chronologically
        cols_chron = np.array([x for _,x in sorted(zip(election_yrs, df_marg.columns))])
        df_marg_sorted = df_marg.loc[:,cols_chron]
        df_d_sorted = df_d.loc[:,cols_chron]
        df_r_sorted = df_r.loc[:,cols_chron]
        df_n_sorted = df_n.loc[:,cols_chron]

        # keep track of the sorted order
        self.election_types_chron = np.array([x for _,x in sorted(zip(election_yrs,election_types))])
        self.election_years_chron = np.array(sorted(election_yrs))
        self.elections_chron = np.array([' '.join(map(str, tup)) for tup in zip(self.election_years_chron, self.election_types_chron)])

        return df_marg_sorted, df_d_sorted, df_r_sorted, df_n_sorted

    def calc_kos_election_trends(self):
        """
        calculate differences in democratic margins and democratic votes
        for elections that happened more than once
        """
        # identify elections that happened more than once
        election_pairs = self.repeated_elections

        # initialize new dataframes
        margin_trends = pd.DataFrame(index=self.statewide_margins.index)
        D_votes_trends = pd.DataFrame(index=self.statewide_margins.index)
        R_votes_trends = pd.DataFrame(index=self.statewide_margins.index)

        for pair in election_pairs:
            # fill it in
            margin_trends[tuple(pair)] = self.statewide_margins[pair].diff(axis=1).iloc[:, 1]
            D_votes_trends[tuple(pair)] = self.D_votes[pair].diff(axis=1).iloc[:, 1]
            R_votes_trends[tuple(pair)] = self.R_votes[pair].diff(axis=1).iloc[:, 1]

        return margin_trends, D_votes_trends, R_votes_trends

    def running_average_statewide(self, window_years=2, end_year=2022,
        use_data='dem_votes', verbose=False):

        # skip the first year because there's nothing before it to average ...
        # and also add 2020
        #years = list(np.unique(self.election_years_chron)[1:])+[2020]
        years = list(range(self.election_years_chron[1], end_year+1))

        # what data are we using?
        if use_data == 'dem_votes':
            data = self.D_votes
        elif use_data == 'rep_votes':
            data = self.R_votes
        else:
            data = self.statewide_margins

        # initialize new dataframe for running averages
        results = pd.DataFrame(index=data.index)

        for year in years:
            # find all elections that fall into this time window
            window_elections = [et for (ey, et) in zip(
                                        self.election_years_chron,
                                        self.elections_chron) if
                                    year - ey <= window_years and
                                    year - ey > 0]

            # take an average
            results[str(year)] = data[window_elections].mean(axis=1)
            if verbose:
                print('year: {}'.format(year))
                print('.. averaging these elections: {}'.format(window_elections))
                print('')
        return results

    def plot_running_averages(self, figheight=6):

        ldi_color = '#666273'
        windows = [2, 4, 6]
        window_colors = ['#40B0C9','#406BC9','#9D40C9']

        if 'district_margins' not in self.datasets:
            raise ValueError('This function requires a dataset called "district_margins" \
             containing all the district-level margins')

        if 'LDI' not in self.datasets:
            raise ValueError('This function requires a dataset called "LDI" \
             containing LDI estimates for every year available')

        elections = np.sort(self.data['district_margins'].columns)
        years = [int(s.split()[0]) for s in elections]
        n_elections = len(years)

        fig, axes = plt.subplots(2, n_elections, figsize=(min(figheight*n_elections+1, 11),figheight))

        for ind, year, election in zip(range(n_elections), years, elections):
            # labels etc
            axes[0, ind].set_title(str(year))
            axes[0, ind].set_xlabel('data')
            axes[0, ind].set_ylabel('model margins')
            axes[1, ind].set_xlabel('data')
            axes[1, ind].set_ylabel('data - model')
            axes[1, ind].set_ylim(-40,40)

            # 1:1 line for reference
            axes[0, ind].plot(range(-100,100), range(-100,100),'--', color='silver')
            axes[1, ind].axhline(0, ls='--', color='silver')

            # actual results
            margins_true = self.data['district_margins']['{} Dem Margin'.format(year)]

            # LDI
            axes[0, ind].plot(margins_true[np.abs(margins_true) < 100],
                           self.data['LDI'][str(year)][np.abs(margins_true) < 100],
                           'o', label='LDI', color=ldi_color, ms=5)
            axes[0, ind].plot(margins_true[np.abs(margins_true) == 100],
                           self.data['LDI'][str(year)][np.abs(margins_true) == 100],
                           'o', label='LDI', alpha=0.25, color=ldi_color, ms=5)
            axes[1, ind].plot(margins_true[np.abs(margins_true) < 100],
                           (margins_true - self.data['LDI'][str(year)])[np.abs(margins_true) < 100],
                           'o', label='LDI', color=ldi_color, ms=5)
            r2_ldi = r2_score(margins_true[np.abs(margins_true) < 100],
                              self.data['LDI'][str(year)][np.abs(margins_true) < 100])
            print('Checking models for {} ...'.format(year))
            print('  R2 score for LDI: {}'.format(r2_ldi))

            # running averages
            use_ms = 5
            for window, window_color in zip(windows, window_colors):
                # obtain prediction for votes -> margins using this type of averaging
                dvotes_pred = self.running_average_statewide(window_years=window,
                                                             use_data='dem_votes')
                margins_pred = demvotes2margin(dvotes_pred[str(year)])

                # plot where elections occured
                axes[0, ind].plot(margins_true[np.abs(margins_true) < 100],
                               margins_pred[np.abs(margins_true) < 100],
                               'o', ms=use_ms, color=window_color,
                               label='{} year window'.format(window))

                axes[1, ind].plot(margins_true[np.abs(margins_true) < 100],
                               (margins_true - margins_pred)[np.abs(margins_true) < 100],
                               'o', ms=use_ms, color=window_color,
                               label='{} year window'.format(window))

                r2 = r2_score(margins_true[np.abs(margins_true) < 100], margins_pred[np.abs(margins_true) < 100])
                if r2 > r2_ldi:
                    improv = '**'
                else:
                    improv = ''
                print('  R2 score for {}-year window running average: {}{}'.format(window, r2, improv))

                use_ms = use_ms - 1.2

            print('')

        fig.subplots_adjust(wspace=0.5)

        fig.subplots_adjust(bottom=0.15)
        leg = fig.add_axes([0.1,0.01,0.8,0.05])
        leg.plot([0.07], [0.5], 'o', ms=6, color=ldi_color)
        leg.text(0.1, 0.5, 'original LDI', color='k', verticalalignment='center', horizontalalignment='left')

        x0 = 0.25
        w = 0.25
        for i,window in enumerate([2,4,6]):
            leg.plot(x0+w*i, 0.5, 'o', ms=6, color=window_colors[i])
            leg.text(x0+w*i+0.02, 0.5, '{}-year running avg.'.format(window),
                        color='k', verticalalignment='center', horizontalalignment='left')
        leg.set_xlim(0,1)
        leg.set_ylim(0,1)
        leg.set_axis_off()
        return fig



    def plot_trend_correction(self, figheight=6, window=4):

        ldi_color = '#666273'
        avg_color =  '#52B7A0' # '#3F57B7'
        corr_color = '#B73F93'

        if 'district_margins' not in self.datasets:
            raise ValueError('This function requires a dataset called "district_margins" \
             containing all the district-level margins')

        if 'LDI' not in self.datasets:
            raise ValueError('This function requires a dataset called "LDI" \
             containing LDI estimates for every year available')

        elections = np.sort(self.data['district_margins'].columns)
        years = [int(s.split()[0]) for s in elections]
        n_elections = len(years)

        fig, axes = plt.subplots(2, n_elections, figsize=(min(figheight*n_elections+1, 11),figheight))

        for ind, year, election in zip(range(n_elections), years, elections):

            # labels etc
            axes[0, ind].set_title(str(year))
            axes[0, ind].set_xlabel('data')
            axes[0, ind].set_ylabel('model')
            axes[1, ind].set_xlabel('data')
            axes[1, ind].set_ylabel('data - model')
            axes[0, ind].set_xlim(-100,100)
            axes[1, ind].set_xlim(-100,100)
            axes[1, ind].set_ylim(-40,40)

            # 1:1 line for reference
            axes[0, ind].plot(range(-100,100), range(-100,100),'--', color='silver')
            axes[1, ind].axhline(0,ls='--', color='silver')

            # actual results
            margins_true = self.data['district_margins']['{} Dem Margin'.format(year)]

            # LDI
            axes[0, ind].plot(margins_true[np.abs(margins_true) < 100],
                           self.data['LDI'][str(year)][np.abs(margins_true) < 100],
                           'o', color=ldi_color, label='LDI')
            axes[0, ind].plot(margins_true[np.abs(margins_true) == 100],
                           self.data['LDI'][str(year)][np.abs(margins_true) == 100],
                           'o', label='LDI', alpha=0.25, color=ldi_color)
            axes[1, ind].plot(margins_true[np.abs(margins_true) < 100],
                           (margins_true - self.data['LDI'][str(year)])[np.abs(margins_true) < 100],
                           'o', color=ldi_color, label='LDI')
            r2_ldi = r2_score(margins_true[np.abs(margins_true) < 100],
                              self.data['LDI'][str(year)][np.abs(margins_true) < 100])
            print('Checking models for {} ...'.format(year))
            print('  R2 score for LDI: {}'.format(r2_ldi))

            # obtain prediction for votes -> margins using a 4 year window
            dvotes_avgpred = self.running_average_statewide(window_years=window)
            margins_avgpred = demvotes2margin(dvotes_avgpred[str(year)])

            # plot where elections occured
            axes[0, ind].plot(margins_true[np.abs(margins_true) < 100],
                           margins_avgpred[np.abs(margins_true) < 100],
                           'o', color=avg_color, ms=4, label='4 year window')
            axes[1, ind].plot(margins_true[np.abs(margins_true) < 100],
                           (margins_true - margins_avgpred)[np.abs(margins_true) < 100],
                           'o', color=avg_color, ms=4, label='4 year window')
            r2_runningavg = r2_score(margins_true[np.abs(margins_true) < 100],
                                     margins_avgpred[np.abs(margins_true) < 100])
            if r2_runningavg > r2_ldi:
                improv1 = '**'
            else:
                improv1 = ''
            print('  R2 score for {}-year window running average: {}{}'.format(window, r2_runningavg, improv1))

            # check for trends where BOTH bracket years were before this one
            trend_cols = self._find_previous_trends(year)

            if len(trend_cols) > 0:
                # take the average of the appropriate set of election trends
                corr = self.D_votes_trends[trend_cols].mean(axis=1)

                # update model
                margins_avgpredcorr = margins_avgpred + corr

                # plot where elections occured
                axes[0, ind].plot(margins_true[np.abs(margins_true) < 100],
                               margins_avgpredcorr[np.abs(margins_true) < 100],
                               'o', ms=4, color=corr_color, label='4 year window + corr')

                axes[1, ind].plot(margins_true[np.abs(margins_true) < 100],
                               (margins_true - margins_avgpredcorr)[np.abs(margins_true) < 100],
                               'o', ms=4, color=corr_color, label='4 year window + corr')

                r2_corr = r2_score(margins_true[np.abs(margins_true) < 100],
                                   margins_avgpredcorr[np.abs(margins_true) < 100])
                if r2_corr > r2_ldi:
                    improv2 = '**'
                else:
                    improv2 = ''
                print('  R2 score for {}-year window running average WITH trends correction: {}{}'.format(window, r2_corr, improv2))

            print('')

        fig.subplots_adjust(wspace=0.5, hspace=0.5)

        fig.subplots_adjust(bottom=0.15)
        leg = fig.add_axes([0.1,0.01,0.8,0.05])
        leg.plot([0.20], [0.5], 'o', ms=6, color=ldi_color)
        leg.text(0.21, 0.5, 'original LDI', color='k', verticalalignment='center', horizontalalignment='left')

        x0 = 0.43
        w = 0.25
        for i,col,lab in zip(range(2), [avg_color, corr_color],
                                ['{}-year running avg.'.format(window), '+ trends corr.']):
            leg.plot(x0+w*i, 0.5, 'o', ms=6, color=col)
            leg.text(x0+w*i+0.02, 0.5, lab, color='k', verticalalignment='center', horizontalalignment='left')
        leg.set_xlim(0,1)
        leg.set_ylim(0,1)
        leg.set_axis_off()
        return fig


    def _find_previous_trends(self, year):
        return [tup for tup in self.D_votes_trends.columns if int(tup[1].split()[0]) < year]


    @property
    def repeated_elections(self):
        election_pairs = []

        # search pattern
        pattern = r'(?P<yy>\d{4}) (?P<elect>\w+[\s\w\W]+)'

        # loop through
        for i, first_elect in enumerate(self.elections_chron):
            grps1 = re.search(pattern, first_elect)
            yr1, first_election_type = grps1.groups()

            for second_elect in self.elections_chron[i+1:]:
                grps2 = re.search(pattern, second_elect)
                yr2, second_election_type = grps2.groups()

                if first_election_type in second_election_type or second_election_type in first_election_type:
                    election_pairs.append((first_elect, second_elect))
                    break

        return np.array(election_pairs)

    @property
    def district_shapes(self):
        district_shapes = gpd.read_file(os.path.join(self.config['border_path'],
                                                     self.config['{}_districts'.format(
                                                         self.district_type.lower())][self.state]['shape_file'])
                                       )
        return district_shapes.to_crs("EPSG:3857")

    @property
    def state_xy_limits(self):
        lon_range = tuple(self.config['{}_districts'.format(self.district_type.lower())][self.state]['lon_range'])
        lat_range = tuple(self.config['{}_districts'.format(self.district_type.lower())][self.state]['lat_range'])

        x_range, y_range = [list(r) for r in lnglat_to_meters(lon_range, lat_range)]
        return x_range, y_range

    @property
    def ldi_from_avg(self):
        return self.statewide_margins.mean(axis=1)

    @property
    def datasets(self):
        return list(self.data.keys())

    ########## MODELING BITS ############

    def grab_elections_in_window(self, year, window_years=4):
        window_elections = [et for (ey, et) in zip(self.election_years_chron, self.elections_chron) if
                            year - ey <= window_years and year - ey > 0]
        return window_elections

    def estimate_pop_D_proportion(self, ci=0.90):
        p_hat = self.D_votes/100 # best estimate
        se_phat = np.sqrt(p_hat*(1-p_hat)/self.n_votes) # standard error

        moe = st.norm.ppf(ci + (1-ci)/2) * se_phat
        return p_hat, moe

    def estimate_pop_prop_diffs(self, trends, ci=0.90):
        pdiffs = self.D_votes_trends[trends]/100 # best estimate

        se_trends = {}
        for trend in trends:
            p1_name, p2_name = trend

            p1 = self.D_votes[p1_name]/100
            p2 = self.D_votes[p2_name]/100
            n1 = self.n_votes[p1_name]
            n2 = self.n_votes[p2_name]

            se = np.sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2)) # standard error
            se_trends[trend] = se

        se_df = pd.DataFrame(data=se_trends, columns=pdiffs.columns)
        moe = st.norm.ppf(ci + (1-ci)/2) * se_df
        return pdiffs, moe

    def montecarlo_sampling(self, year=2020, window=4, n_iter=2000, sampling_dist_ci=0.90):
        # set up: election results distributions
        p_hat, moe = self.estimate_pop_D_proportion(ci=sampling_dist_ci)

        # set up: past trends
        trends = self._find_previous_trends(year)
        pdiffs, moe_diffs = self.estimate_pop_prop_diffs(trends, ci=sampling_dist_ci)

        # placeholder for results
        iterated_avg_results, iterated_corr_results = {}, {}

        # ready go
        for iter_ in range(n_iter):
            # grab the elections within the time window
            try_elections = self.grab_elections_in_window(year, window_years=window)

            # bootstrap elections and trends
            bs_elections = bootstrap_from(try_elections)
            bs_trends = bootstrap_from(trends)

            # number of voters for each bootstrapped election
            bs_numvoters = self.n_votes[bs_elections]

            # draw results from the distribution of each election
            # let the width of the distribution ~ 90% confidence interval
            sampled_p, sampled_n, sampled_t = {}, {}, {}

            for i, elec in enumerate(bs_elections):
                sampled_p[i] = sample_from(p_hat[elec], moe[elec])
                sampled_n[i] = self.n_votes[elec]

            for i, trend in enumerate(bs_trends):
                sampled_t[i] = sample_from(pdiffs[trend], moe_diffs[trend])

            # convert to dataframes
            sampled_demvotes = pd.DataFrame(data=sampled_p, index=self.D_votes.index)
            sampled_numvotes = pd.DataFrame(data=sampled_n, index=self.D_votes.index)
            sampled_trends = pd.DataFrame(data=sampled_t, index=self.D_votes.index)

            # weighted average
            w_avg = (sampled_demvotes.multiply(sampled_numvotes)).sum(axis=1).div(sampled_numvotes.sum(axis=1))

            # weighted average corrected by an average trend
            avg_trend = sampled_trends.mean(axis=1)
            w_avg_corr = w_avg + avg_trend

            # keep track of results
            iterated_avg_results[iter_] = w_avg
            iterated_corr_results[iter_] = w_avg_corr

        out_avg = pd.DataFrame(data=iterated_avg_results)
        out_avg.name = '{} model: {}-year window'.format(year, window)
        out_corr = pd.DataFrame(data=iterated_corr_results)
        out_corr.name = '{} model: {}-year window + trend corr'.format(year, window)

        return out_avg, out_corr



######### OTHER UTILITY FUNCTIONS, NOT PART OF THE CLASS ############

def demvotes2margin(demvotes):
    margin = 2*demvotes - 100
    return margin

def margin2demvotes(margin):
    demvotes = (margin + 100)/2
    return demvotes


def bootstrap_from(data):
    return random.choices(data, k=len(data))

def sample_from(mu, sigma):
    return [random.gauss(mu.loc[i], sigma.loc[i]) for i in mu.index]



def plot_montecarlo_model_viz(pred, test_data=None, ldi=None, figwidth=6, ref_color='#5C3099'):

    if test_data is not None:
        figheight = figwidth * 1.2
    else:
        figheight = figwidth
        figwidth = figwidth*2

    # define median
    med = pred.quantile(q=0.50, axis=1)
    sorted_med_inds = np.argsort(med)[::-1]

    # define inner 90%
    e_95_50 = pred.quantile(q=0.95, axis=1) - pred.quantile(q=0.50, axis=1)
    e_50_05 = pred.quantile(q=0.50, axis=1) - pred.quantile(q=0.05, axis=1)

    # define inner 50%
    e_75_50 = pred.quantile(q=0.75, axis=1) - pred.quantile(q=0.50, axis=1)
    e_50_25 = pred.quantile(q=0.50, axis=1) - pred.quantile(q=0.25, axis=1)

    # set up figure
    fig = plt.figure()
    fig.suptitle(pred.name, y=0.95)
    gs = fig.add_gridspec(4,4)
    fig.set_figwidth(figwidth)
    fig.set_figheight(figheight)

    if test_data is not None:
        fig.set_figwidth(figwidth*2)
        fig.set_figheight(figheight*1.2)

        ax_model = fig.add_subplot(gs[:,:2])
        ax_model.set_xlim(0,1)
        ax_compare = fig.add_subplot(gs[:2,2:])
        ax_compare.set_xlim(0,1)
        ax_ldi = fig.add_subplot(gs[2:,2:])
        ax_ldi.set_xlim(0,1)

    else:
        ax_model = fig.add_subplot(gs[:,:2])
        ax_model.set_xlim(0,1)
        ax_ldi = fig.add_subplot(gs[:,2:])
        ax_ldi.set_xlim(0,1)

    ax_model.axvspan(0.45, 0.55, alpha=0.15, color='indigo')
    ax_model.errorbar(med.iloc[sorted_med_inds], range(len(med)),
                      xerr=[e_50_05.iloc[sorted_med_inds], e_95_50.iloc[sorted_med_inds]],
                      fmt='o', ms=1.5, ecolor='#B2B2B3', elinewidth=2.5, color='k')
    ax_model.errorbar(med.iloc[sorted_med_inds], range(len(med)),
                      xerr=[e_50_25.iloc[sorted_med_inds], e_75_50.iloc[sorted_med_inds]],
                      fmt='o', ms=2.5, ecolor='#5E5E5E', elinewidth=2.5, color='k') #5E5C88, 5B5BA0
    ax_model.tick_params(left=False, labelleft=False)
    ax_model.set_xlabel('Predicted Dem Fraction')
    ax_model.spines['left'].set_visible(False);


    ax_ldi.plot(np.linspace(0.25, 1, 10), np.linspace(0.25, 1, 10), '-', color=ref_color, lw=0.85, alpha=0.75)
    ax_ldi.errorbar(med.values, (ldi + 100)/200,
                     yerr=[e_50_05, e_95_50],
                     fmt='o', ms=1.5, ecolor='#B2B2B3', elinewidth=1.9, color='k')
    ax_ldi.errorbar(med.values, (ldi + 100)/200,
                     yerr=[e_50_25, e_75_50],
                     fmt='o', ms=2, ecolor='#5E5E5E', elinewidth=1.9, color='k')
    ax_ldi.set_ylabel('original LDI')
    ax_ldi.set_xlabel('model');

    if test_data is not None:
        # compare model to real results
        ax_compare.axhline(y=0, ls='-', color=ref_color, lw=0.85, alpha=0.75)
        ax_compare.errorbar(med.values, test_data - med.values,
                         yerr=[e_50_05, e_95_50],
                         fmt='o', ms=1.5, ecolor='#B2B2B3', elinewidth=1.9, color='k')
        ax_compare.errorbar(med.values, test_data - med.values,
                         yerr=[e_50_25, e_75_50],
                         fmt='o', ms=2, ecolor='#5E5E5E', elinewidth=1.9, color='k')
        ax_compare.set_xlabel('model')
        ax_compare.set_ylabel('data - model')

    fig.subplots_adjust(hspace=0.5, bottom=0.1)


    return fig
