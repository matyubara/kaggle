import logging
import json
import requests

import numpy as np
import pandas as pd

from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

# =============================================================================
# utility
# https://amalog.hateblo.jp/entry/kaggle-snippets
# =============================================================================

def reduce_mem_usage(df, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    start_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != 'object' and col_type != 'datetime64[ns]':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # feather-format cannot accept float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print_('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def write_spreadsheet(*args):
    endpoint = 'https://script.google.com/macros/s/AKfycbxZhZc3lPJ6eLATt_r9dPVQZUjuzpIvQ6vjSYFZdLOlt1TqWvtC/exec'
    requests.post(endpoint, json.dumps(args))


# =============================================================================
# time feature
# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
# =============================================================================

def make_day_feature(df, offset=0, tname='TransactionDT'):
    """
    Creates a day of the week feature, encoded as 0-6.
    Parameters:
    -----------
    df : pd.DataFrame
        df to manipulate.
    offset : float (default=0)
        offset (in days) to shift the start/end of a day.
    tname : str
        Name of the time column in df.
    """
    # found a good offset is 0.58
    days = df[tname] / (3600*24)
    encoded_days = np.floor(days-1+offset) % 7
    return encoded_days

def make_hour_feature(df, tname='TransactionDT'):
    """
    Creates an hour of the day feature, encoded as 0-23.
    Parameters:
    -----------
    df : pd.DataFrame
        df to manipulate.
    tname : str
        Name of the time column in df.
    """
    hours = df[tname] / (3600)
    encoded_hours = np.floor(hours) % 24
    return encoded_hours


# =============================================================================
# metrics
# =============================================================================

def plot_learning_curve(estimator, X, y, scoring_metrix, cv, 
                        train_sizes=np.linspace(0.1, 1.0, 10), 
                        title="learning curve", ylim=(0.0, 1.01)):
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, 
        groups=None, train_sizes=train_sizes, cv=cv, scoring=scoring_metrix, 
        exploit_incremental_learning=False, n_jobs=None, pre_dispatch='all', 
        verbose=0, shuffle=False, random_state=0, error_score='raise-deprecation')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
        train_scores_mean + train_scores_std, alpha=0.1, color='royalblue')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
        test_scores_mean + test_scores_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='royalblue', label="training score")
    plt.plot(test_sizes, test_scores_mean, 'o-', color='royalblue', label="cross-validation score")

    plt.title(title)
    plt.grid()
    plt.xlabel("training sample")
    plt.ylabel("score")
    plt.ylim(ylim)
    plt.legend(loc='lower right')

    return plt

def plot_roc_curve(y, y_pred_prob, title="ROC curve"):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred_prob, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)

    plt.plot(false_positive_rate, true_positive_rate, label="ROC curve (area = %.2f)"%auc_score)
    plt.title(title)
    plt.grid()
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend()

    return plt

def plot_pr_curve(y, y_pred_prob, title="PR curve"):
    precision, recall, thresholds = precision_recall_curve(y, y_pred_prob)
    average_precision = average_precision(precision, recall)

    plt.plot(precision, recall, label="PR curve (ap = %.2f)"%average_precision)
    plt.title(title)
    plt.grid()
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.legend()

    return plt

def plot_score_changed_thresholds(y, y_pred_prob, title="model-score changed thresholds"):
    precision, recall, thresholds = precision_recall_curve(y, y_pred_prob, pos_label=1)
    thresholds = np.append(thresholds, 1)
    fscore = 2 * precision * recall / (precision + recall)

    queue_rate = []
    for threshold in thresholds:
        queue_rate.append((threshold <= y_pred_prob).mean())

    plt.plot(thresholds, precision, linestyle='--', color=sns.color_palette("tab10", 4)[0])
    plt.plot(recall, precision, linestyle='-', color=sns.color_palette("tab10", 4)[1])
    plt.plot(fscore, precision, linestyle='-.', color=sns.color_palette("tab10", 4)[2])
    # plt.plot(fscore, queue_rate, linestyle=':', color=sns.color_palette("tab10", 4)[3])

    plt.axvline(x=thresholds[np.argmax(fscore)], color=sns.color_palette("tab10", 4)[3], linewidth=0.5)

    plt.title(title)
    plt.grid()
    plt.xlim([0, 1])
    plt.xlabel("threshold")
    plt.ylim([0, 1])
    plt.ylabel("score")
    # leg = plt.legend(("precision", "recall", "f-measure", "queue_rate"), loc='upper right', frameon=True)
    leg = plt.legend(("precision", "recall", "f-measure", "max f-measure"), loc='upper right', frameon=True)
    leg.get_frame().set_edgecolor('k')

    return plt

def cumulative_gain_curve(y_true, y_score):
    pos_label = 1

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_true = (y_true == pos_label)

    sorted_indices = np.argsort(y_score)[::1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arrange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains

def plot_cumulative_gain_chart(y, y_pred_prob, title="cumulative gain chart"):
    y_true = np.array(y)
    y_probas = np.array(y_pred_prob)

    percentages, gains = cumulative_gain_curve(y, y_pred_prob)

    plt.plot(percentages, gains, lw=3)

    plt.title(title)
    plt.grid()
    plt.xlabel("percentage of sample")
    plt.ylabel("%pos label")

    return plt

def plot_lift_chart(y, y_pred_prob, title="lift chart"):
    y_true = np.array(y)
    y_probas = np.array(y_pred_prob)

    percentages, gains = cumulative_gain_curve(y, y_pred_prob)
    gains = gains / percentages

    plt.plot(percentages, gains, lw=3, label="lift")
    plt.plot([0, 1], [1, 1], "k--", lw=2, label="baseline")

    plt.title(title)
    plt.grid()
    plt.xlabel("percentage of sample")
    plt.ylabel("lift")
    plt.legend()

    return plt

# FREQUENCY ENCODE TOGETHER
def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm,', ',end='')

# LABEL ENCODE
def encode_LE(col,train=X_train,test=X_test,verbose=True):
    df_comb = pd.concat([train[col],test[col]],axis=0)
    df_comb,_ = df_comb.factorize(sort=True)
    nm = col
    if df_comb.max()>32000: 
        train[nm] = df_comb[:len(train)].astype('int32')
        test[nm] = df_comb[len(train):].astype('int32')
    else:
        train[nm] = df_comb[:len(train)].astype('int16')
        test[nm] = df_comb[len(train):].astype('int16')
    del df_comb; x=gc.collect()
    if verbose: print(nm,', ',end='')

# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
# GROUP AGGREGATION MEAN AND STD
# GROUP AGGREGATE　使い方の例
# encode_AG(['TransactionAmt','D9','D11'],['card1','card1_addr1','card1_addr1_P_emaildomain'],['mean','std'],usena=True)
def encode_AG(main_columns, uids, aggregations=['mean'], train_df=X_train, test_df=X_test, 
              fillna=True, usena=False):
    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')
                
                if fillna:
                    train_df[new_col_name].fillna(-1,inplace=True)
                    test_df[new_col_name].fillna(-1,inplace=True)
                
                print("'"+new_col_name+"'",', ',end='')
                
# COMBINE FEATURES
# COMBINE COLUMNS CARD1+ADDR1, CARD1+ADDR1+P_EMAILDOMAIN
# encode_CB('card1','addr1')
def encode_CB(col1,col2,df1=X_train,df2=X_test):
    nm = col1+'_'+col2
    df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 
    encode_LE(nm,verbose=False)
    print(nm,', ',end='')
    
# GROUP AGGREGATION NUNIQUE
# AGGREGATE
# encode_AG2(['P_emaildomain','dist1','DT_M','id_02','cents'], ['uid'], train_df=X_train, test_df=X_test)
def encode_AG2(main_columns, uids, train_df=X_train, test_df=X_test):
    for main_column in main_columns:  
        for col in uids:
            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)
            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')
            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')
            print(col+'_'+main_column+'_ct, ',end='')

def update_tracking(self, field, value, csv_file="../input/tracking.csv", integer=False, digits=None,
                    drop_incomplete_rows=False):
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except FileNotFoundError:
        df = pd.DataFrame()

    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)

    if drop_incomplete_rows:
        df = df.loc[~df['AUC'].isna()]
    df.loc[run_id, field] = str(value)  # Model number is index
    df.to_csv(csv_file)


# =============================================================================
# others
# =============================================================================

# df_corr_re = pd.DataFrame()
# series_corr = pd.Series()
# 
# for col in list_col_name:
#     corr = correlation_ratio(df_analysis[["~", col]])
#     series_corr[col] = corr
# df_corr_re["~"] = series_corr
# 
# sns.set()
# plt.figure(figsize=(25,5))
# sns.heatmap(df_corr_re.T, vmin=0, vmax=1, square=True, linewidths=0.5, annot=True, cmap='Reds')
# plt.savefig("./output/~")
def add(x, y):
    return x + y

def correlation_ratio(df):
    categories = set(df.iloc[:, 0])
    groups = {}
    for i in categories:
        for index, row in df.iterrows():
            if i == row[0]:
                val = groups.setdefault(i, [])
                val.append(row[1])
                groups[i] = val

    avgs = {k: ((reduce(add, v, 0.0)) / len(v)) for k, v in groups.items()}

    within_class_variation = {}
    for k, v in groups.items():
        avg = avgs.get(k)
        val = within_class_variation.setdefault(k, 0.0)
        v2 = [(x - avg) ** 2 for x in v]
        val = reduce(add, v2, 0.0)
        within_class_variation[k] = val

    all_avg = df.iloc[:,1].mean()
    #all_avg = reduce(add, [v[0] for k, v in m.items()], 0.0) / len(m)

    between_class_variation = 0.0
    for k, v in groups.items():
        between_class_variation += len(v) * ((avgs.get(k) - all_avg) ** 2)

    return (between_class_variation / (reduce(add, [v for v in within_class_variation.values()], 0.0) + between_class_variation))
