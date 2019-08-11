import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import pickle
from scipy import stats
from mpl_toolkits import mplot3d
import seaborn as sns
sns.set()

# Generate a double nested dictionary of lists of alpha values
def gen_nested_dict_1(raw_dict):  
    mag_dict_big = {}
    for element in raw_dict.keys():
        temp_dict = {}
        for alpha in raw_dict[element].keys():
            temp_dict[alpha]=raw_dict[element][alpha][0]
        mag_dict_big[element]=temp_dict
    return mag_dict_big

# turn double nested dictionary into pandas data frame
def df_from_dict_2nested(raw_dict):   
    df = pd.DataFrame.from_dict({(i,j): raw_dict[i][j]
                               for i in raw_dict.keys()
                               for j in raw_dict[i].keys()},
                               orient='index')
    # turn tuple columns into multiindex
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.transpose()
    return df

def histo_alphas(df,element,num_alpha):
    """Plat set of histograms for PBE and RSCAN"""
    fig, axs = plt.subplots(ncols=num_alpha,figsize=(15,5))
    for i in range(0,num_alpha):
        sns.distplot(df[element]['alpha_'+str(i+1)].dropna(),ax=axs[i],label='RSCAN').set_title(element+' alpha_'+str(i+1))
        axs[i].legend()
        axs[i].set(xlabel="alpha")
    
def summary_stats(df):
    """Generates summary statistics for alpha histograms"""
    data = []
    for alpha in df.columns.levels[1]:
        for element in df.columns.levels[0]:    
            max_a = np.max(df[element][alpha].dropna())
            min_a = np.min(df[element][alpha].dropna())
            ave_a = np.average(df[element][alpha].dropna())
            med_a = np.median(df[element][alpha].dropna())
            data.append([element,alpha,max_a,min_a, ave_a,med_a])
    data = np.array(data,ndmin=2)
    summary = pd.DataFrame(data=data,columns=['element','alpha',
                                    'max_a','min_a', 'ave_a','med_a'])
    for i in ['max_a','min_a', 'ave_a','med_a']:
        summary[i] = summary[i].astype('float')
    summary = summary.set_index(['element', 'alpha']).sort_index()
    return summary


def fraction_above(alpha_list,crit_value):
    count = 0
    for alpha in alpha_list:
        if alpha > crit_value :
            count += 1
    fraction = count / len(alpha_list)
    return fraction

def print_fractions(df,value):
    fract_list = []
    for element in df.columns.levels[0]:
        frac_list_inner = []
        index_list_inner = []
        for alpha in df.columns.levels[1]:
            frac_list_inner.append(fraction_above(df[element][alpha].dropna(),value))
        fract_list.append(frac_list_inner)
            
    df_temp = pd.DataFrame(fract_list,index=df.columns.levels[0],
                           columns=df.columns.levels[1])
    return df_temp

def return_intersection(hist_1, hist_2):
    """Returns intersection of two sets of histogram data"""
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

# Generate a tripple nested dictionary of lists of alpha values
def gen_nested_dict(mag,mag_pbe):  
    mag_dict_big = {}
    for element in mag.keys():
        temp_dict = {}
        for alpha in mag[element].keys():
            temp_dict_inner = {}
            temp_dict_inner['RSCAN']=mag[element][alpha][0]
            temp_dict_inner['PBE']=mag_pbe[element][alpha][0]
            temp_dict[alpha]=temp_dict_inner
        mag_dict_big[element]=temp_dict
    return mag_dict_big

# turn tripple nested dictionary into pandas data frame
def gen_df_from_dict(mag_dict_big):   
    mag_big_df = pd.DataFrame.from_dict({(i,j,k): mag_dict_big[i][j][k] 
                               for i in mag_dict_big.keys()
                               for j in mag_dict_big[i].keys()
                               for k in mag_dict_big[i][j].keys()},
                               orient='index')
    # turn tuple columns into multiindex
    mag_big_df.index = pd.MultiIndex.from_tuples(mag_big_df.index)
    mag_big_df = mag_big_df.transpose()
    return mag_big_df

def histo_mega_plot(df,element,num_alpha):
    """Plat set of histograms for PBE and RSCAN"""
    fig, axs = plt.subplots(ncols=num_alpha,figsize=(15,5))
    for i in range(0,num_alpha):
        sns.distplot(df[element]['alpha_'+str(i+1)]['RSCAN'].dropna(),ax=axs[i],label='RSCAN').set_title(element+' alpha_'+str(i+1))
        sns.distplot(df[element]['alpha_'+str(i+1)]['PBE'].dropna(),ax=axs[i],label='PBE')
        axs[i].legend()
        axs[i].set(xlabel="alpha")
       
def compare_hist(mag,mag_pbe):
    """Generates statistical comparison of alpha lists"""
    data = []
    for alpha in range(1,len(mag[list(mag.keys())[0]].keys())+1):
        for atom in mag.keys():    
            a1,b,c = plt.hist(mag[atom]['alpha_'+str(alpha)][0],bins=20,alpha=0.5);
            a2,b,c = plt.hist(mag_pbe[atom]['alpha_'+str(alpha)][0],bins=20,alpha=0.5);
            intersect = return_intersection(a1,a2)
            statistic , p_wilcox = stats.wilcoxon(a1,a2)
            statistic , p_ks = stats.ks_2samp(a1,a2)
            entropy = stats.entropy(a1, qk=a2, base=None)
            data.append([atom,alpha,p_wilcox,p_ks,intersect,entropy])
    data = np.array(data,ndmin=2)
    mag_comp = pd.DataFrame(data=data,columns=['element','alpha','p_wilcox','p_ks','intersect','entropy'])
    for i in ['p_wilcox','p_ks','intersect','entropy']:
        mag_comp[i] = mag_comp[i].astype('float')
    mag_comp = mag_comp.set_index(['element', 'alpha']).sort_index()
    return mag_comp

def histo_mega_plot_new(df,element):
    """Plot set of histograms"""
    num_cols = len(df.columns.levels[1])
    fig, axs = plt.subplots(ncols=num_cols,figsize=(15,5))
    count = 0 
    for alpha in df.columns.levels[1]:
        sns.distplot(df[element][alpha].dropna(),
                     ax=axs[count]).set_title(element+' '+alpha)
        count += 1
        
def histo_mega_plot_2(df,element):
    """Plot set of histograms"""
    num_cols = len(df.columns.levels[1][0:3])
    fig, axs = plt.subplots(ncols=num_cols,figsize=(15,5))
    count = 0 
    for alpha in df.columns.levels[1][0:3]:
        sns.distplot(df[element][alpha].dropna(),
                     ax=axs[count],label=alpha).set_title(element+' '+alpha)
        sns.distplot(df[element]['no_spin'].dropna(),ax=axs[count],label='no_spin')
        axs[count].legend()
        count += 1
        
# Modified Slightly
# turn tripple nested dictionary into pandas data frame
def gen_df_from_dict_new(mag_dict_big):   
    mag_big_df = pd.DataFrame.from_dict({(i,j): mag_dict_big[i][j][0] 
                               for i in mag_dict_big.keys()
                               for j in mag_dict_big[i].keys()},
                               orient='index')
    # turn tuple columns into multiindex
    mag_big_df.index = pd.MultiIndex.from_tuples(mag_big_df.index)
    mag_big_df = mag_big_df.transpose()
    return mag_big_df

def compare_hist_new(df,bins=20):
    data = []
    for alpha in df.columns.levels[1][:-1]:
        for element in df.columns.levels[0]:
            a1,b,c = plt.hist(df[element][alpha],bins=bins,alpha=0.5)
            a2,b,c = plt.hist(df[element]['no_spin'],bins=bins,alpha=0.5)
            intersect = return_intersection(a1,a2)
            statistic , p_wilcox = stats.wilcoxon(a1,a2)
            statistic , p_ks = stats.ks_2samp(a1,a2)
            entropy = stats.entropy(a1, qk=a2, base=None)
            data.append([element,alpha,p_wilcox,p_ks,intersect,entropy])
    data = np.array(data,ndmin=2)
    comp_df = pd.DataFrame(data=data,columns=['element','alpha',
                                                   'p_wilcox','p_ks','intersect','entropy'])
    for i in ['p_wilcox','p_ks','intersect','entropy']:
        comp_df[i] = comp_df[i].astype('float')
    comp_df = comp_df.set_index(['element', 'alpha']).sort_index()
    return comp_df

def plot_heat_triple(alpha_slice_1,alpha_slice_2,labels=['A','B'],set_label='',
                    vmin=None,vmax=None,vmin_diff=None,vmax_diff=None):
    diff_slice = alpha_slice_1 - alpha_slice_2
    fix, axs = plt.subplots(ncols=4,figsize=(20,5))
    sns.heatmap(data = alpha_slice_1, ax=axs[0], vmin=vmin, vmax=vmax).set_title(
            labels[0]+': '+set_label)
    sns.heatmap(data = alpha_slice_2, ax=axs[1], vmin=vmin, vmax=vmax).set_title(
            labels[1]+': '+set_label)
    sns.heatmap(data = diff_slice, ax=axs[2], vmin=vmin_diff, vmax=vmax_diff
            ).set_title('Difference')
    sns.heatmap(data = abs(diff_slice), ax=axs[3], vmin=vmin_diff, vmax=vmax_diff
            ).set_title('Difference,abs')
    
# Modified Slightly
# turn tripple nested dictionary into pandas data frame
def gen_df_from_dict_new_new(mag_dict_big):   
    mag_big_df = pd.DataFrame.from_dict({(i,j): mag_dict_big[i][j][0] 
                               for i in mag_dict_big.keys()
                               for j in mag_dict_big[i].keys()},
                               orient='index')
    # turn tuple columns into multiindex
    mag_big_df.index = pd.MultiIndex.from_tuples(mag_big_df.index)
    mag_big_df = mag_big_df.transpose()
    return mag_big_df

print ('Loaded all functions')