#!/usr/bin/env python
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from statis_f import statis
from pathlib import Path as P

koppen_path = lambda p: P(os.path.abspath(os.path.join("../../data/koppendata",p)))
QBF_LABEL, QBF_UNIT = "Bankfull discharge", "[m$^3$s$^{-1}$]"
degree_sign = u'\N{DEGREE SIGN}'
climate_order={
'A (tropical)': 0,
'B (arid)': 1,
'C (temperate)': 2,
'D (cold)': 3,
}

display_names = {
    'r2': "R²",
    'MPE': "MPE",
    'MAPE': "MAPE", 
    'MdPE': "MdPE",
    'MdAPE': "MdAPE",
    'NSE': "E",
    'RMSE': "RMSE",
    'nRMSE': "nRMSE"
}
# define the categories and bins for QBF classification
categories = ['Q>10000', 'Q5000-10000', 'Q1000-5000', 'Q500-1000', 'Q100-500', 'Q<=100']
bins = [0, 100, 500, 1000, 5000, 10000, float('inf')]

def mean_percentage_error(obs,est):
    # positive - overestimation
    mpe = 100*(np.mean((est-obs)/obs))
    return mpe
def mean_abs_percentage_error(obs,est):
    mape = 100*(np.mean(np.abs((est-obs))/obs))
    return mape
def median_percentage_error(obs,est):
    mdpe = 100*(np.median((est-obs)/obs))
    return mdpe
def median_absolute_percentage_error(obs,est):
    mdape = 100*np.median(np.abs((est-obs)/obs))
    return mdape
def RMSE(obs,est):
    rmse = np.sqrt(((est-obs) ** 2).mean())
    return rmse
def nRMSE(obs,est):
    nrmse = np.sqrt((((est-obs)/obs) ** 2).mean())
    return nrmse

def NSE(obs, est):
    # modified NSE (Legates & McCabe, 1999)
    # none square nse
    nse = 1 - (np.sum(np.abs(est - obs)) / np.sum(np.abs(obs - obs.mean())))
    return nse

def r2_cal(obs,est):
    r2 = 1-((sum((est-obs)**2))/sum((obs-np.repeat(obs.mean(),len(obs)))**2))
    return r2
def kge(obs, est):
    from scipy.stats import pearsonr
    # Calculate correlation coefficient
    r, _ = pearsonr(obs, est)

    # Calculate standard deviation ratio
    s = np.std(est) / np.std(obs)

    # Calculate mean ratio
    b = np.mean(est) / np.mean(obs)

    # Calculate KGE
    kge_value = 1 - np.sqrt((r - 1)**2 + (s - 1)**2 + (b - 1)**2)

    return kge_value

# define metrics
MPE = mean_percentage_error
MAPE = mean_abs_percentage_error
MdPE = median_percentage_error
MdAPE = median_absolute_percentage_error

def _ix_overlap(df1, df2):
    ixo = df1.index.intersection(df2.index)
    return df1.loc[ixo], df2.loc[ixo]

def _read_or_df(dforpath, index_col=0, **kw):
    if type(dforpath) == str:
        return pd.read_csv(dforpath, index_col=index_col, **kw)
    else:
        return dforpath
def _read_predictors_info(dforpath, active=True):
    if type(dforpath) == str:
        df = pd.read_csv(dforpath, index_col=0)
        if active is not None:
            df = df.loc[df.active == active]
        return df
    else:
        return dforpath
def customize(color_file,col,index='koppen_vals'):
    df = pd.read_csv(color_file)
    df = df.dropna()
    # convert to str
    df[[index]] = df[[index]].astype(str)
    df = df.set_index(index)
    color_dict = df.to_dict()[col]
    return color_dict


# def df_none_check(df, model_df, skip_col="Type"):
#     if df is None:
#         df = model_df.copy()
#     else:
#         cols = [c for c in df.columns if c != skip_col]
#         df[cols] += model_df[cols]
#     return df

def df_none_check(df, model_df, valid_counts=None, skip_col="Type"):
    if df is None:
        df = model_df.copy()
        if valid_counts is not None:
            for idx in model_df.index:
                numeric_cols = model_df.select_dtypes(include='number').columns
                valid_counts[idx] = 0 if model_df.loc[idx, numeric_cols].isna().all() else 1
    else:
        cols = [c for c in df.columns if c != skip_col]
        df[cols] = df[cols].add(model_df[cols], fill_value=0)
        if valid_counts is not None:
            for idx in model_df.index:
                numeric_cols = model_df.select_dtypes(include='number').columns
                if not model_df.loc[idx, numeric_cols].isna().all():
                    valid_counts[idx] = valid_counts.get(idx, 0) + 1
    return df

def df_error_mean(df, n_run, df_type, valid_counts=None):
    if valid_counts is not None:
        # divide each row by its own valid run count
        for idx in df.index:
            n_valid = valid_counts.get(idx, n_run)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    value = df.loc[idx, col] / n_valid if n_valid > 0 else np.nan
                    df.loc[idx, col] = int(round(value)) if col == 'Count' and not np.isnan(value) else value
    else:
        def divide_col(x):
            if pd.api.types.is_numeric_dtype(x):
                result = x / n_run
                return result.round().astype(int) if x.name == 'Count' else result
            return x
        df = df.apply(divide_col)

    if 'Type' not in df.columns:
        df.insert(loc=0, column='Type', value=df_type)
    return df

def stations_obs(station_qbf,koppen_color,stratify_col,quantile=True):

    # convert koppen value to five climate zone
    name_dict = customize(koppen_color,'Koppen_short')
    color_dict = customize(koppen_color,'Colors_c')

    st = _read_or_df(station_qbf,index_col='station_id')
    st = st.astype({"source_id": int})
    if quantile:
        # option 1 - use percentile to classify
        percentiles = [0.25, 0.50, 0.75]
        st['log10_qbf'] = np.log10(st['qbf'])
        percentile_values = st['log10_qbf'].quantile(percentiles)
        categories = ['v-large', 'large', 'medium', 'small']  # Correct order
        conditions = [
        (st['log10_qbf'] > percentile_values.loc[0.75]),
        (st['log10_qbf'] > percentile_values.loc[0.50]) & (st['log10_qbf'] <= percentile_values.loc[0.75]),
        (st['log10_qbf'] > percentile_values.loc[0.25]) & (st['log10_qbf'] <= percentile_values.loc[0.50]),
        (st['log10_qbf'] <= percentile_values.loc[0.25])
        ]
        st['qbf_cat'] = np.select(conditions, categories, default='Uncategorized')

    # use defined range to classify - for evaluation
    categories = ['Q>10000', 'Q5000-10000', 'Q1000-5000', 'Q500-1000', 'Q100-500', 'Q<=100']
    bins = [0, 100, 500, 1000, 5000, 10000, float('inf')]
    conditions = [
    (st['qbf'] > 10000),
    (st['qbf'] > 5000)& (st['qbf'] <= 10000),
    (st['qbf'] > 1000)& (st['qbf'] <= 5000),
    (st['qbf'] > 500)& (st['qbf'] <= 1000),
    (st['qbf'] > 100) & (st['qbf'] <= 500),
    (st['qbf'] <= 100)
    ]
    st['evaluation_cat'] = pd.cut(st['qbf'], bins=bins, labels=categories, right=False)

    st['climate'] = st['koppen'].astype(str).map(name_dict)

    # drop polar
    print('The number of training data in polar region is ',str(len(st[st['climate']=='E (polar)'])))
    st = st[st['climate']!='E (polar)']

    # add the color by climate
    st['koppen_color']=st['koppen'].astype(str).map(color_dict)
    order = np.lexsort([st['climate'].map(climate_order)])
    st = st.iloc[order]

    # generate the stratify column for later use in train_test_split
    st[stratify_col] = st['qbf_cat'].astype(str)+'_'+st['climate'].astype(str)
    value_counts = st[stratify_col].value_counts()
    return st,value_counts

def predictors_process(predictors_info,station_predictors):
    predictors_info = _read_predictors_info(predictors_info)
    print("The predictors used are ", predictors_info.index)
    predictors = _read_or_df(station_predictors, index_col='station_id',na_values="*")[predictors_info.index]

    # fill effective river width as 0 when null
    gswe_widths = ['gswe_width_occ_'+str(occ)+'_scaled' for occ in [1,10,20,30,40,50]]
    for col in gswe_widths:
        predictors[col] = predictors[col].fillna(0)
    cleaned_pred = predictors.dropna(how="any")

    if len(cleaned_pred) != len(predictors):
        prm = (len(predictors) - len(cleaned_pred)) * 100 / len(predictors)
        warnings.warn(f"Removed {prm:.2f}% of stations due to missing values.")
    cleaned_pred = cleaned_pred[~cleaned_pred.index.duplicated(keep='first')]
    
    return cleaned_pred

def model_training(i,cleaned_pred,qbf,st,stratify_col,output_hyparameters,\
    ncpus,permutation_importances,output_cross_val,y_target='qbf'):
    from sklearn.model_selection import train_test_split
    from sklearn.utils import class_weight
    from sklearn.metrics import make_scorer, r2_score
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    import sklearn.preprocessing as sklprep
    from joblib import dump, load
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold,cross_val_score,GridSearchCV

    print('ncpus= ',str(ncpus))
    print('This is the ', str(i),' run of RF!')
    # ensure the random state is different at each run of the model
    defkwargs = dict(bootstrap=True, random_state=42+i)
    # split, find best parameters
    # generate separate validation dataset
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_pred,
        qbf,
        test_size=0.2,
        random_state=defkwargs.get('random_state'), # set the random_state to allow reproduce
        stratify=st.loc[qbf.index][stratify_col],
    )
    # import ipdb;ipdb.set_trace()
    # generate weight by climate zone
    y_class=st.loc[y_train.index]['climate'].astype(str)
    class_weights = class_weight.compute_class_weight(class_weight='balanced',\
        classes=np.unique(y_class.values), y=y_class)
    class_weights_dict = dict(zip(np.unique(y_class.values),class_weights))
    
    # each run the random_state is different
    cv = KFold(n_splits=5, shuffle=True,random_state=defkwargs.get('random_state')) 

    model = make_pipeline(sklprep.RobustScaler(), RandomForestRegressor())
    param_grid = [{
   'randomforestregressor__n_estimators': [100,200,300,400,500], # 800 proved to be a good one in multiple tests
   'randomforestregressor__max_features': [0.2,0.3,0.4], # 0.2, 0.4 are good
   'randomforestregressor__min_samples_leaf': [2,3,4,5,6,7,8] # 6,8 are good
    }]

    # quick test can be done by passing a single parameter set below
#     param_grid = [{
#    'randomforestregressor__n_estimators': [200], 
#    'randomforestregressor__max_features': [0.4], 
#    'randomforestregressor__min_samples_leaf': [6]
#     }]
    # best parameter search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='r2',n_jobs=ncpus)
    grid_search.fit(X_train, y_train,\
        randomforestregressor__sample_weight=[class_weights_dict[label] for label in y_class])
    best_model = grid_search.best_estimator_ # best-fitted model
    best_hyperparameters = grid_search.best_params_
    print(best_hyperparameters)

    # save the calibrated hyperparameters
    if output_hyparameters:
        path = os.path.dirname(output_hyparameters)
        if not os.path.isdir(path):
            os.mkdir(path)
        name,ext = os.path.splitext(os.path.basename(output_hyparameters))[0],os.path.splitext(os.path.basename(output_hyparameters))[1]
        output_hyparameters = path+'/'+name+'_RUN_'+str(i)+ext
        with open(output_hyparameters,'w') as data:
            data.write(str(best_hyperparameters))

    # evaluate model - overall  
    scoring_functions = [r2_score, MPE, MAPE, MdPE, MdAPE, NSE, RMSE, nRMSE]
    score_names = ['r2','MPE','MAPE','MdPE','MdAPE', 'NSE', 'RMSE', 'nRMSE']

    cross_val_scores = {name: [] for name in score_names}
    cross_val_scores_climate = {cat: {name: [] for name in score_names} 
                            for cat in climate_order}
    climate_counts = {cat: [] for cat in climate_order}
    # set a different random_state than calibration with the cv above
    cv_crossval = KFold(n_splits=5, shuffle=True,random_state=defkwargs.get('random_state')+1) # generate random spliting each run
    # cross-validation metrics for overall; each climate
    for train_idx, val_idx in cv_crossval.split(X_train):
        X_tr_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        best_model.fit(X_tr_fold, y_tr_fold)
        y_pred = pd.Series(best_model.predict(X_valid_fold), index=y_valid_fold.index)

        for score_name, score_func in zip(score_names, scoring_functions):
            cross_val_scores[score_name].append(score_func(y_valid_fold.to_numpy().flatten(), y_pred.to_numpy()))
    
        # per-climate scores — filter the same val fold
        for cat in climate_order:
            index_cat = st[st['climate'] == cat].index.intersection(y_valid_fold.index)
            count = len(index_cat)
            if count == 0:
                print(f"Cross-validation, Run {i+1} {cat}: no sample, error metrics will be NaN.")
                continue
            elif count < 2:
                print(f"Cross-validation, Run {i+1} {cat}: only {count} sample, R²/E will be NaN.")

            y_valid_fold_cat  = y_valid_fold.loc[index_cat].to_numpy().flatten()
            y_pred_cat = y_pred.loc[index_cat].to_numpy()
            climate_counts[cat].append(len(y_valid_fold_cat))  # record count this fold
            for score_name, score_func in zip(score_names, scoring_functions):
                cross_val_scores_climate[cat][score_name].append(score_func(y_valid_fold_cat, y_pred_cat))
    
    # get the mean over 5-fold cross-validation
    cross_val_scores = {
    m: np.mean(cross_val_scores[m]) for m in score_names
    }
    # print('The r2 of cross-validation is '+str('{:.2f}'.format(cross_val_scores['r2'])))
    # Average sample count per climate across folds
    climate_sample_counts = {
        cat: int(round(np.mean(counts))) if counts else 0
        for cat, counts in climate_counts.items()
    }
    # get the mean over 5-fold cross-validation for each climate
    model_metrics = []
    for cat in climate_order:
        if cat not in cross_val_scores_climate or not cross_val_scores_climate[cat][score_names[0]]:
            model_metrics.append((cat, 0.0, *['-'] * len(score_names)))
            continue
        count = climate_sample_counts.get(cat, 0)
        metrics_row = [cat, count]
        metrics_row.extend([
            np.mean(cross_val_scores_climate[cat][m]) for m in score_names
        ])
        model_metrics.append(tuple(metrics_row))

    column_names = ['climate', 'Count']
    column_names.extend([display_names.get(name, name) for name in score_names])
    model_df = pd.DataFrame(model_metrics, columns=column_names)
    model_df = model_df.set_index('climate')
    model_df.insert(0,'Type','Cross-validation')
    print("Cross-validation is done!")

    # plot cross-validation results
    if output_cross_val:
        # import ipdb;ipdb.set_trace()

        path = os.path.dirname(output_cross_val)
        fig_cross_val = plot_cv_split(i,cv_crossval, cross_val_scores,X_train, y_train,best_model,\
                                st,path,permutation_importances,defkwargs,y_target)
        if not os.path.isdir(path):
            os.mkdir(path)
        name,ext = os.path.splitext(os.path.basename(output_cross_val))[0],os.path.splitext(os.path.basename(output_cross_val))[1]
        output_cross_val_fig = path+'/'+name+'_RUN_'+str(i+1)+ext
        fig_cross_val.supxlabel("Observed Q$_{BFobs}$ " + QBF_UNIT, \
                                fontsize=8, fontweight='normal')

        fig_cross_val.save(output_cross_val_fig)
        # before model evaluation

    y_pred = pd.Series(best_model.predict(cleaned_pred),index=cleaned_pred.index) # all the data
    best_model.qbf=qbf # obs
    best_model.y_pred=y_pred # pred
    best_model.data=st.loc[y_pred.index, ['koppen_color', 'climate']]
    best_model.X_train=X_train
    best_model.X_test=X_test
    best_model.Y_train=y_train
    best_model.Y_test=y_test
    # calculate the error metrics for the TRAIN dataset
    train_scores = {}
    for score_name, score_func in zip(score_names, scoring_functions):
        # import ipdb;ipdb.set_trace()
        result = score_func(y_train.to_numpy().flatten(), y_pred.loc[y_train.index].to_numpy())
        train_scores[score_name] = np.mean(result)

    # calculate the error metrics for the separate TEST dataset
    test_scores = {}
    for score_name,score_func in zip(score_names,scoring_functions):
        result = score_func(y_test.to_numpy().flatten(), y_pred.loc[y_test.index].to_numpy())
        test_scores[score_name] = np.mean(result)
    

    # metric calculation at TEST data for each climate
    validation_metrics = []
    for cat in climate_order:
        if cat not in st['climate'].unique():
            continue

        index_cat = st[st['climate'] == cat].index.intersection(y_test.index)
        count = len(index_cat)
        
        if count == 0:
            print(f"Test, Run {i+1} {cat}: no sample, error metrics will be NaN.")
            continue
        elif count < 2:
            print(f"Test, Run {i+1} {cat}: only {count} sample, R²/E will be NaN.")
        y_true = y_test.loc[index_cat]
        y_pred = best_model.y_pred.loc[index_cat]

        is_valid = cat != 'E (polar)' and not pd.isna(cat)
        scores = [
            func(y_true.to_numpy().flatten(), y_pred.to_numpy()) if is_valid else '-'
            for func in scoring_functions
        ]
        validation_metrics.append((cat, count, *scores))

    # list of tuple to dataframe
    validation_df = pd.DataFrame(
        validation_metrics,columns=['climate',\
        'Count',
        "R{}".format("\u00B2"),\
        'MPE','MAPE','MdPE','MdAPE',\
        "E","RMSE","nRMSE"])
    validation_df = validation_df.set_index('climate')
    validation_df.insert(0,'Type','Test')

    # metric calculation at TEST data for each range of QBF
    validation_metrics_QBFcat = []
    # import ipdb;ipdb.set_trace()
    # exclude polar
    st = st[st['climate']!='E (polar)']    
    for cat in categories:
        if cat not in st['evaluation_cat'].unique():
            continue
        index_cat = st[st['evaluation_cat'] == cat].index.intersection(y_test.index)
        count = len(index_cat)
        if count == 0:
            print(f"Test, Run {i+1} {cat}: no sample, error metrics will be NaN.")
            continue
        elif count < 2:
            print(f"Test, Run {i+1} {cat}: only {count} sample, R²/E will be NaN.")

        y_true = y_test.loc[index_cat]
        y_pred = best_model.y_pred.loc[index_cat]

        scores = [func(y_true.to_numpy().flatten(), y_pred.to_numpy()) for func in scoring_functions]
        validation_metrics_QBFcat.append((cat, count, *scores))

    # Convert list of tuples to DataFrame with specific column names
    validation_df_QBFcat = pd.DataFrame(
        validation_metrics_QBFcat, columns=[
            y_target.upper()+'cat',
            'Count',
            "R{}".format("\u00B2"),
            'MPE', 'MAPE', 'MdPE', 'MdAPE',
            "E", "RMSE", "nRMSE"
        ]
    )
    # Set the category column as index
    validation_df_QBFcat = validation_df_QBFcat.set_index(y_target.upper()+'cat')
    # Insert the Type column
    validation_df_QBFcat.insert(0, 'Type', 'Test')
    # reset r2 and nse to / as those are not meaning values for category error metrics
    

    # save train and test to csv
    qbf_train_test = pd.DataFrame(pd.concat([y_train,y_test]))
    qbf_train_test["type"] = ["Train" if i in y_train.index else "Test" for i in qbf_train_test.index]
    qbf_train_test["predicted"] = best_model.y_pred

    # feature importances
    best_model.importances = pd.DataFrame({
        "Importance": best_model.named_steps['randomforestregressor'].feature_importances_,
        "std": np.std([tree.feature_importances_ for tree in best_model.named_steps['randomforestregressor'].estimators_], axis=0),
        "5%": np.percentile([tree.feature_importances_ for tree in best_model.named_steps['randomforestregressor'].estimators_], 5, axis=0),
        "95%": np.percentile([tree.feature_importances_ for tree in best_model.named_steps['randomforestregressor'].estimators_], 95, axis=0),
        },
        index=cleaned_pred.columns,
    )

    label_map = {
    'r2': "R{}".format("\u00B2"),
    'NSE': "E",
    }
    best_model.performance = pd.DataFrame.from_dict(
    {label_map.get(m, m): (train_scores[m], cross_val_scores[m], test_scores[m]) for m in score_names},
    orient='index',
    columns=['Train', 'Cross-validation', 'Test'],
    )

    # best_model for current run, 5-fold for each climate, test for each climate, test for each QBF cat, train_test sample
    return best_model,model_df,validation_df,validation_df_QBFcat,qbf_train_test

def plot_cv_split(run_no,cv,cross_val_scores,X,y,model,st,
                  path,permutation_importances,
                  defkwargs,y_target='qbf'):
    import proplot as pplt
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from sklearn.inspection import permutation_importance

    title = f'Cross-validation of run {run_no+1}'
    fig, ax = pplt.subplots(
        [[1, 2, 3], [4, 5, 6]],  # 0 means empty/muted
        sharey=True, sharex=True, suptitle=title,titleloc='l',
        journal="nat2",
        tight=True,
    )
    ax[5].axis('off')
    ax[5].format(title='Model Parameters & CV Scores')
    for i in range(len(ax)-1):
        ax[i].format(abc='a.')

    rf_params = model.named_steps['randomforestregressor'].get_params()
    keys = ['n_estimators', 'max_features', 'min_samples_leaf']
    param_text = "\n".join([f"{k}: {rf_params[k]}" for k in keys])

    cv_text = "\n".join([f"{m}: {v:.2f}" for m, v in cross_val_scores.items() if m != 'NSE'])

    full_text = "— Parameters —\n" + param_text + "\n\n— CV Scores —\n" + cv_text

    ax[5].text(
        0.05, 0.95, full_text,
        transform=ax[5].transAxes,
        va='top', ha='left',
        fontsize=8,
    )

    scatterkw = dict(markersize=1.3, alpha=1, rasterized=True)
    markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
    lim = (1, 100000)
    cross = []

    if not os.path.isdir(path):
        os.mkdir(path)

    # model trained data is splited as train,test again for cross-validation
    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X = X.reset_index()
        y = y.reset_index()
        train_id = y.loc[train_index]['station_id']
        test_id = y.loc[test_index]['station_id']
        # import ipdb;ipdb.set_trace()
        X.set_index('station_id',drop=True,inplace=True)
        y.set_index('station_id',drop=True,inplace=True)
        model.fit(X.loc[train_id], y.loc[train_id][y_target])
        model.predicted_train = model.predict(X.loc[train_id])
        model.predicted_test = model.predict(X.loc[test_id])
        model.x_train = X.loc[train_id]
        model.x_test = X.loc[test_id]
        model.y_train = y.loc[train_id]
        model.y_test = y.loc[test_id]
        # import ipdb;ipdb.set_trace()
        r2_train,r2_test = r2_score(model.y_train, model.predicted_train), r2_score(model.y_test, model.predicted_test)
        # scatter plot
        ax[i].plot(lim, lim, "k--", alpha=0.5)
        lab = f"Train ({len(model.y_train)}, {'{:.2f}'.format(r2_train)})"
        ax[i].scatter(model.y_train, model.predicted_train, s=scatterkw.get('markersize'), label=lab)
        lab = f"Validation ({len(model.y_test)}, {'{:.2f}'.format(r2_test)})"
        ax[i].scatter(model.y_test, model.predicted_test, s=scatterkw.get('markersize'), label=lab)
        ax[i].format(
            xscale="log", yscale="log",
            ylim=lim, xlim=lim,
            # title="Q$_{BFobs}$ vs Q$_{BFest}$",
            # xlabel="Observed Q$_{BFobs}$ "+QBF_UNIT, 
            ylabel="Estimated Q$_{BFest}$ "+QBF_UNIT,
            fontsize=8,
        )
        ax[i].legend(loc="lr", ncol=1, fontsize=6, frame=False,
            title=f"CV Fold {i+1}" , titlefontsize="medium")
    
        cross.append([i,r2_train,r2_test])

        st.loc[train_id,"type"] = "Trainfold-"+str(i+1)
        st.loc[test_id,"type"] = "Validationfold-"+str(i+1)
        st.loc[train_id,"predicted"] = model.predicted_train.tolist()
        st.loc[test_id,"predicted"] = model.predicted_test.tolist()

        st.loc[y.index,"(obs-pred)/obs"] = (st.loc[y.index,y_target]-st.loc[y.index,'predicted'])/st.loc[y.index,y_target]
        # import ipdb;ipdb.set_trace()
        st['(obs-pred)/obs'] = st['(obs-pred)/obs'].map(lambda n: '{:.2f}'.format(float(n)) if not pd.isnull(n) and n != '' else '')
        output_cross_val_csv = path+'/'+'train_crossvalidation_run_'+str((run_no+1))+'_split_'+str(i+1)+'.csv'
        st.to_csv(output_cross_val_csv)
        if permutation_importances:
            # permutation_importance
            test_im = permutation_importance(model, model.x_test, model.y_test, n_repeats=10,\
            random_state=defkwargs.get('random_state'), n_jobs=defkwargs.get('ncpus'))
            sorted_importances_idx = test_im.importances_mean.argsort()
            test_importances = pd.DataFrame(test_im.importances[sorted_importances_idx].T,\
            columns=X.columns[sorted_importances_idx])

            train_im = permutation_importance(model, model.x_train, model.y_train, n_repeats=10, \
            random_state=defkwargs.get('random_state'), n_jobs=defkwargs.get('ncpus'))
            sorted_importances_idx = train_im.importances_mean.argsort()
            train_importances = pd.DataFrame(train_im.importances[sorted_importances_idx].T,\
            columns=X.columns[sorted_importances_idx])
            # train, test in two subplot
            fig2, ax2 = plt.subplots(
            ncols=2, nrows=1,sharex=True,figsize=(16,9) # sharey=False,
            )
            for j,importances in enumerate(zip(["train", "test"], [train_importances, test_importances])):
                name = importances[0]
                # import ipdb;ipdb.set_trace()
                importances[1].plot.box(vert=False, whis=10,ax=ax2[j])
                ax2[j].set_title(f"PIm ({name} set)")
                ax2[j].axvline(x=0, color="k", linestyle="--")
                ax2[j].figure.tight_layout()
                ax2[j].set_xlabel("Decrease AS")
            fig2.suptitle('Split '+str(i)+' r2-train='+ '{:.2f}'.format(r2_train) \
            +' r2-test='+'{:.2f}'.format(r2_test),fontsize=10)
            if permutation_importances:
                fig2.savefig(permutation_importances+'_run_'+str((run_no+1))+'_split_'+str(i+1)+'.png')
    print('The cross-validation score are ',cross)
    # print('The mean value of the cross-validation score is ',np.mean(cross))
    return fig


def train_model_all_rebuild(trained_model,station_qbf, station_predictors, predictors_info, \
                koppen_color,\
                ncpus=1,\
                output=None, \
                output_performance=None, \
                output_csv=None,\
                permutation_importances=None,n_run = 10, **kw):
    """
    n_run: the number of times to run the model
    """
    import time
    start_time = time.time()
    from sklearn.metrics import make_scorer, r2_score
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    import sklearn.preprocessing as sklprep
    from joblib import dump, load
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
    from multiprocessing import Pool
    from bankfull_discharge_estimates import _load_model
    
    # take the QBF samples from the submitted paper model
    df_ori = pd.read_csv(trained_model,index_col='station_id')
    

    stratify_col = 'combined_cat'
    # get the parent path from the output_csv
    output_path = os.path.dirname(output_csv)
    # below two files are saved at each run of training the model
    output_cross_val=output_path+"/"+"cross_val/train_test_cross_val_5fold.png"
    # output_cross_val = None
    output_hyparameters=output_path+"/"+"cross_val/best_hyperparameters.txt"

    # prepare training target and predictors
    st,value_counts = stations_obs(station_qbf,koppen_color,stratify_col,quantile=True)
    # import ipdb;ipdb.set_trace()
    # remove stations where the count of their category [QBF_magnitude,climate] is smaller than 5
    st = st[st[stratify_col].isin(value_counts.index[value_counts >= 5])]
    st = st[st.index.isin(df_ori.index)]
    # overlap by station_id

    # keep only one for duplicated station_id
    st = st[~st.index.duplicated(keep='first')]
    cleaned_pred = predictors_process(predictors_info,station_predictors)

    cleaned_pred, qbf = _ix_overlap(cleaned_pred,st[["qbf"]])
    cleaned_pred_cp = cleaned_pred.copy()

    # define an empty list to store the index for training data
    idx_train = set()
    model_importance = None
    model_performance = None
    models = []
    test_climate_df_final = None
    validation_climate_df_final = None
    test_QBFcat_df_final = None

    # model_training
    y_target = 'qbf'
    valid_run_counts_test = {}
    valid_run_counts_crossvalidation = {}
    valid_run_counts_QBFcat = {}  
    for i in np.arange(n_run): 
        best_model, test_climate_df, validation_climate_df, \
        test_QBFcat_df, qbf_train_test = model_training(
            i, cleaned_pred, qbf, st, stratify_col, output_hyparameters, 
            ncpus, permutation_importances, output_cross_val, y_target)
        
        models.append(best_model)
        
        # Initialize or accumulate metrics
        if i==0:
            model_importance = best_model.importances.copy()
            model_performance = best_model.performance.copy()
            best_model_performance = best_model.performance.copy()
            best_model_selected = best_model
        else:
            model_importance += best_model.importances
            model_performance += best_model.performance
            
            # Select best model by lowest RMSE sum
            current_rmse = best_model.performance.loc['RMSE'][['Train','Test']].sum()
            best_rmse = best_model_performance.loc['RMSE'][['Train','Test']].sum()
            if current_rmse < best_rmse:
                best_model_performance = best_model.performance.copy()
                best_model_selected = best_model

        # Update dataframes
        # import ipdb;ipdb.set_trace()
        test_climate_df_final = df_none_check(test_climate_df_final, test_climate_df,valid_run_counts_test)
        validation_climate_df_final = df_none_check(validation_climate_df_final, validation_climate_df,valid_run_counts_crossvalidation)
        test_QBFcat_df_final = df_none_check(test_QBFcat_df_final, test_QBFcat_df,\
                                                   valid_run_counts_QBFcat)
    
        # Track training indices (use set instead of list comprehension)
        idx_train.update(qbf_train_test[qbf_train_test["type"] == 'Train'].index)

    # print('The number of training data is ', str(len(set(idx_train))))
    print('The best model performance is \n', best_model_performance)
    print('The valid count for test_climate is\n',valid_run_counts_test)
    print('The valid count for validation_climate is\n',valid_run_counts_crossvalidation)
    print('The valid count for QBFcat is\n',valid_run_counts_QBFcat)

    # calculate the mean over all models
    model_importance /= n_run
    model_performance /= n_run

    # add the averaged value back to the best_model
    best_model = best_model_selected
    best_model.importances = model_importance
    best_model.performance = model_performance

    test_climate_df_final, validation_climate_df_final,test_QBFcat_df_final = [
        df_error_mean(df, n_run, source_df.pop('Type'),valid_counts=valid_counts)
        for df, source_df,valid_counts in [
            (test_climate_df_final, test_climate_df,valid_run_counts_test),
            (validation_climate_df_final, validation_climate_df,valid_run_counts_crossvalidation),
            (test_QBFcat_df_final,test_QBFcat_df,valid_run_counts_QBFcat),
        ]
    ]
    # set the r2, E as / for QBFcat
    test_QBFcat_df_final[["R{}".format("\u00B2"), 'E']] = '/'

    qbf_train_test[[stratify_col,'koppen_color']] = st[[stratify_col,'koppen_color']]
    qbf_train_test["koppen"] = st['koppen']
    # save the train,test of each time to csv
    qbf_train_test.to_csv(output_csv)

    # import ipdb;ipdb.set_trace()
    if output_performance:
        # save output_performance to csv
        df_perf = best_model.performance.T
        df_perf.insert(0,'Type','Overall')
        train_count = len(best_model.X_train)
        test_count = len(best_model.X_test)
        df_perf['Count'] = pd.Series({"Train": train_count, "Cross-validation":'/',"Test": test_count})
        # import ipdb;ipdb.set_trace()
        df_perf = pd.concat([df_perf,test_climate_df_final,validation_climate_df_final,test_QBFcat_df_final])
        df_perf = df_perf.set_index('Type',append=True).swaplevel()
        df_perf.to_csv(output_performance)
    if output:
        # save the best model
        dump(best_model, output)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

if __name__ == "__main__":
    from commandline import interface as _cli
    _cli()
