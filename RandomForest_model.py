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
def NSE(obs,est):
    # none square nse
    nse = 1-(sum(np.abs(est-obs))/sum(np.abs(obs-np.repeat(obs.mean(),len(obs)))))
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

def df_none_check(df,model_df):
    if df is None:
        df = model_df.copy()
    else:
        df += model_df
    return df

def df_error_mean(df, n_run, df_type):
    # Apply division only to numeric values, leave others unchanged
    df = df.apply(lambda x: x / n_run if pd.api.types.is_numeric_dtype(x) else x)

    # Insert the Type column
    if 'Type' not in df.columns:
        df.insert(loc=0, column='Type', value=df_type)
    return df

def stations_obs(station_qbf,koppen_color,stratify_col):

    # convert koppen value to five climate zone
    name_dict = customize(koppen_color,'Koppen_short')
    color_dict = customize(koppen_color,'Colors_c')

    st = _read_or_df(station_qbf,index_col='station_id')
    st = st.astype({"source_id": int})
    # option 1 - use defined range to classify
    # categories = ['Q>10000', 'Q5000-10000', 'Q1000-5000', 'Q500-1000', 'Q100-500', 'Q<=100']
    # bins = [0, 100, 500, 1000, 5000, 10000, float('inf')]
    # conditions = [
    # (st['qbf'] > 10000),
    # (st['qbf'] > 5000)& (st['qbf'] <= 10000),
    # (st['qbf'] > 1000)& (st['qbf'] <= 5000),
    # (st['qbf'] > 500)& (st['qbf'] <= 1000),
    # (st['qbf'] > 100) & (st['qbf'] <= 500),
    # (st['qbf'] <= 100)
    # ]
    st['qbf_cat'] = pd.cut(st['qbf'], bins=bins, labels=categories, right=False)

    # option 2 - use percentile to classify
    # percentiles = [0.25, 0.50, 0.75]
    # st['log10_qbf'] = np.log10(st['qbf'])
    # percentile_values = st['log10_qbf'].quantile(percentiles)
    # categories = ['v-large', 'large', 'medium', 'small']  # Correct order
    # conditions = [
    # (st['log10_qbf'] > percentile_values.loc[0.75]),
    # (st['log10_qbf'] > percentile_values.loc[0.50]) & (st['log10_qbf'] <= percentile_values.loc[0.75]),
    # (st['log10_qbf'] > percentile_values.loc[0.25]) & (st['log10_qbf'] <= percentile_values.loc[0.50]),
    # (st['log10_qbf'] <= percentile_values.loc[0.25])
    # ]
    # st['qbf_cat'] = np.select(conditions, categories, default='Uncategorized')
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

    # split, find best parameters
    # generate separate validation dataset
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_pred,
        qbf,
        test_size=0.2,
        random_state=42,
        stratify=st.loc[qbf.index][stratify_col],
    )

    # generate weight by climate zone
    y_class=st.loc[y_train.index]['climate'].astype(str)
    class_weights = class_weight.compute_class_weight(class_weight='balanced',\
        classes=np.unique(y_class.values), y=y_class)
    class_weights_dict = dict(zip(np.unique(y_class.values),class_weights))
    cv = KFold(n_splits=5, shuffle=True) # generate random spliting each run

    # best parameter search with cross-validation
    # ensure the random state is different at each run of the model
    defkwargs = dict(bootstrap=True, random_state=42+i)
    model = make_pipeline(sklprep.RobustScaler(), RandomForestRegressor())
    param_grid = [{
   'randomforestregressor__n_estimators': [100,200,300,400,500], # 800 proved to be a good one in multiple tests
   'randomforestregressor__max_features': [0.2,0.3,0.4], # 0.2, 0.4 are good
   'randomforestregressor__min_samples_leaf': [2,3,4,5,6,7,8] # 6,8 are good
    }]
#     param_grid = [{
#    'randomforestregressor__n_estimators': [200], # 800 proved to be a good one in multiple tests
#    'randomforestregressor__max_features': [0.4], # 0.2, 0.4 are good
#    'randomforestregressor__min_samples_leaf': [6] # 6,8 are good
#     }]
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
    cross_val_scores = {}
    for score_name,score_func in zip(score_names,scoring_functions):
        cross_val_scores[score_name] = cross_val_score(best_model, X_train, y_train, cv=cv, n_jobs=ncpus, scoring=make_scorer(score_func))
    r2,mpe = np.mean(cross_val_scores['r2']),np.mean(cross_val_scores['MPE'])
    mape,mdpe,mdape = np.mean(cross_val_scores['MAPE']),np.mean(cross_val_scores['MdPE']),np.mean(cross_val_scores['MdAPE'])
    nse,rmse,nrmse= np.mean(cross_val_scores['NSE']),np.mean(cross_val_scores['RMSE']),np.mean(cross_val_scores['nRMSE'])
    print('The r2 of cross-validation is '+str('{:.2f}'.format(r2)))

    y_pred = pd.Series(best_model.predict(cleaned_pred),index=cleaned_pred.index) # all the data
    print("Cross-validation is done!")

    # plot cross-validation results
    if output_cross_val:
        fig_cross_val = plot_cv_split(cv, X_train, y_train,best_model,st,permutation_importances,defkwargs,y_target)
        path = os.path.dirname(output_cross_val)
        if not os.path.isdir(path):
            os.mkdir(path)
        name,ext = os.path.splitext(os.path.basename(output_cross_val))[0],os.path.splitext(os.path.basename(output_cross_val))[1]
        output_cross_val = path+'/'+name+'_RUN_'+str(i)+ext
        fig_cross_val.save(output_cross_val)
        # before model evaluation

    # metric calculation at training data for each climate
    train_samples = len(y_train)
    model_metrics = []
    cross_val_scores_climate = {}
    for cat in climate_order:
        if cat in st['climate'].unique():
            index_cat = st[st['climate']==cat].index.intersection(y_train.index)
            ratio = len(index_cat) / train_samples
            if len(index_cat) >0:
                for score_name,score_func in zip(score_names,scoring_functions):
                    cross_val_scores_climate[score_name] = cross_val_score(best_model, X_train.loc[index_cat], \
                        y_train.loc[index_cat], cv=cv, n_jobs=ncpus, scoring=make_scorer(score_func))
                metrics_row = [cat, ratio]
                metrics_row.extend([np.mean(cross_val_scores_climate[score]) for score in score_names])
                model_metrics.append(tuple(metrics_row))
            else:
                model_metrics.append((cat, 0.0, *['-'] * len(score_names)))
        else:
            # Include missing categories with zero ratio
            model_metrics.append((cat, 0.0, *['-'] * len(score_names)))
    column_names = ['climate', 'Ratio']
    column_names.extend([display_names.get(name, name) for name in score_names])
    model_df = pd.DataFrame(model_metrics, columns=column_names)
    model_df = model_df.set_index('climate')
    model_df.insert(0,'Type','5-fold')

    best_model.qbf=qbf # obs
    best_model.y_pred=y_pred # pred
    best_model.data=st.loc[y_pred.index, ['koppen_color', 'climate']]
    best_model.X_train=X_train
    best_model.X_test=X_test
    best_model.Y_train=y_train
    best_model.Y_test=y_test
    
    # calculate the above value for the separate TEST dataset
    cross_val_scores_test = {}
    for score_name,score_func in zip(score_names,scoring_functions):
        result = score_func(y_test, y_pred.loc[y_test.index])
        cross_val_scores_test[score_name] = result
        if score_name in ['r2','MdAPE','NSE']:
            print('The test score for ',score_name,' is ',str('{:.2f}'.format(cross_val_scores_test[score_name])))
    r2_test,mpe_test = np.mean(cross_val_scores_test['r2']),np.mean(cross_val_scores_test['MPE'])
    mape_test,mdpe_test = np.mean(cross_val_scores_test['MAPE']),np.mean(cross_val_scores_test['MdPE'])
    mdape_test = np.mean(cross_val_scores_test['MdAPE'])
    nse_test,rmse_test= np.mean(cross_val_scores_test['NSE']),np.mean(cross_val_scores_test['RMSE'])
    nrmse_test = np.mean(cross_val_scores_test['nRMSE'])

    # Calculate the total number of test samples
    test_samples = len(y_test)
    # metric calculation at TEST data for each climate
    validation_metrics = []
    # import ipdb;ipdb.set_trace()
    for cat in climate_order:
        if cat in st['climate'].unique():
            index_cat = st[st['climate']==cat].index.intersection(y_test.index)
            ratio = len(index_cat) / test_samples
            mpe_cat = MPE(y_test.loc[index_cat],best_model.y_pred.loc[index_cat])
            mape_cat = MAPE(y_test.loc[index_cat],best_model.y_pred.loc[index_cat])
            mdpe_cat = MdPE(y_test.loc[index_cat],best_model.y_pred.loc[index_cat])
            mdape_cat = MdAPE(y_test.loc[index_cat],best_model.y_pred.loc[index_cat])
            if cat!='E (polar)' and not pd.isna(cat):
                r2_cat = r2_score(y_test.loc[index_cat],best_model.y_pred.loc[index_cat])
                nse_cat = NSE(y_test.loc[index_cat],best_model.y_pred.loc[index_cat])
                rmse_cat = RMSE(y_test.loc[index_cat],best_model.y_pred.loc[index_cat])
                nrmse_cat = nRMSE(y_test.loc[index_cat],best_model.y_pred.loc[index_cat])
            else:
                r2_cat = '-'
                nse_cat = '-'
                rmse_cat = '-'
                nrmse_cat = '-'
            validation_metrics.append((cat,ratio,r2_cat,mpe_cat,mape_cat,mdpe_cat,mdape_cat,nse_cat,rmse_cat,nrmse_cat))
    
    # list of tuple to dataframe
    validation_df = pd.DataFrame(
        validation_metrics,columns=['climate',\
        'Ratio',
        "R{}".format("\u00B2"),\
        'MPE','MAPE','MdPE','MdAPE',\
        "E","RMSE","nRMSE"])
    validation_df = validation_df.set_index('climate')
    validation_df.insert(0,'Type','climate')

    # metric calculation at TEST data for each range of QBF
    validation_metrics_QBFcat = []
    # exclude polar
    st = st[st['climate']!='E (polar)']    

    # Initialize the list to store validation metrics
    validation_metrics_QBFcat = []
    # import ipdb;ipdb.set_trace()
    # Loop through each category in the specified order
    for cat in categories:
        if cat in st[y_target+'_cat'].unique():
            # Get subset once
            mask = st[y_target+'_cat'] == cat
            index_cat = st[mask].index.intersection(y_test.index)
            y_true_cat = y_test.loc[index_cat]
            y_pred_cat = best_model.y_pred.loc[index_cat]
            
            # Calculate ratio
            ratio = len(index_cat) / test_samples
            
            # Dictionary of metric functions
            metrics = {
                'r2': lambda: r2_score(y_true_cat, y_pred_cat),
                'mpe': lambda: MPE(y_true_cat, y_pred_cat),
                'mape': lambda: MAPE(y_true_cat, y_pred_cat),
                'mdpe': lambda: MdPE(y_true_cat, y_pred_cat),
                'mdape': lambda: MdAPE(y_true_cat, y_pred_cat),
                'nse': lambda: NSE(y_true_cat, y_pred_cat),
                'rmse': lambda: RMSE(y_true_cat, y_pred_cat),
                'nrmse': lambda: nRMSE(y_true_cat, y_pred_cat),
            }
            
            # Calculate all metrics
            metric_values = tuple(func() for func in metrics.values())
            validation_metrics_QBFcat.append((cat, ratio, *metric_values))

    for cat in categories:
        # Check if this category exists in the data
        if cat in st[y_target+'_cat'].unique():
            # Get indices for this category
            index_cat = st[st[y_target+'_cat']==cat].index.intersection(y_test.index)
            
            # Calculate the ratio of samples in this category
            ratio = len(index_cat) / test_samples
            
            # Calculate all the metrics
            mpe_cat = MPE(y_test.loc[index_cat], best_model.y_pred.loc[index_cat])
            mape_cat = MAPE(y_test.loc[index_cat], best_model.y_pred.loc[index_cat])
            mdpe_cat = MdPE(y_test.loc[index_cat], best_model.y_pred.loc[index_cat])
            mdape_cat = MdAPE(y_test.loc[index_cat], best_model.y_pred.loc[index_cat])
            r2_cat = r2_score(y_test.loc[index_cat], best_model.y_pred.loc[index_cat])
            nse_cat = NSE(y_test.loc[index_cat], best_model.y_pred.loc[index_cat])
            rmse_cat = RMSE(y_test.loc[index_cat], best_model.y_pred.loc[index_cat])
            nrmse_cat = nRMSE(y_test.loc[index_cat], best_model.y_pred.loc[index_cat])
            
            # Append as tuple including the ratio
            validation_metrics_QBFcat.append((cat, ratio, r2_cat, mpe_cat, mape_cat, mdpe_cat, mdape_cat, nse_cat, rmse_cat, nrmse_cat))

    # Convert list of tuples to DataFrame with specific column names
    validation_df_QBFcat = pd.DataFrame(
        validation_metrics_QBFcat, columns=[
            y_target.upper()+'cat',
            'Ratio',
            "R{}".format("\u00B2"),
            'MPE', 'MAPE', 'MdPE', 'MdAPE',
            "E", "RMSE", "nRMSE"
        ]
    )
    # Set the category column as index
    validation_df_QBFcat = validation_df_QBFcat.set_index(y_target.upper()+'cat')

    # Insert the Type column
    validation_df_QBFcat.insert(0, 'Type', y_target.upper()+'cat')

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
    best_model.performance = pd.DataFrame.from_dict({
        "R{}".format("\u00B2"): (r2,r2_test),
        "MPE": (mpe,mpe_test),
        "MAPE": (mape,mape_test),
        "MdPE": (mdpe,mdpe_test),
        "MdAPE": (mdape,mdape_test),
        "E": (nse,nse_test),
        "RMSE": (rmse,rmse_test),
        "nRMSE": (nrmse,nrmse_test),
    },
    orient='index',
    columns=['Train','Test'],
    )

    return best_model,model_df,validation_df,validation_df_QBFcat,qbf_train_test

def plot_cv_split(cv,X,y,model,st,permutation_importances,defkwargs,y_target='qbf'):
    import proplot as pplt
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from sklearn.inspection import permutation_importance
    title = 'Cross validation'
    fig, ax = pplt.subplots(
        ncols=4, nrows=3,sharey=True, sharex=True, suptitle=title, abc="a.", titleloc='l',
        journal="nat2",
    )
    scatterkw = dict(markersize=1.3, alpha=1, rasterized=True)
    markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
    lim = (1, 100000)
    cross = []
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
        r2_train,r2_test = r2_score(model.y_train, model.predicted_train), r2_score(model.y_test, model.predicted_test)
        # scatter plot
        ax[i].plot(lim, lim, "k--", alpha=0.5)
        lab = f"training ({len(model.y_train)}, {'{:.2f}'.format(r2_train)})"
        ax[i].scatter(model.y_train, model.predicted_train, s=scatterkw.get('markersize'), label=lab)
        lab = f"test ({len(model.y_test)}, {'{:.2f}'.format(r2_test)})"
        ax[i].scatter(model.y_test, model.predicted_test, s=scatterkw.get('markersize'), label=lab)
        ax[i].format(
            xscale="log", yscale="log",
            ylim=lim, xlim=lim,
            title="Q$_{BFobs}$ vs Q$_{BFest}$",
            xlabel="Observed Q$_{BFobs}$ "+QBF_UNIT, ylabel="Estimated Q$_{BFest}$ "+QBF_UNIT,
            fontsize=8,
        )
        ax[i].legend(loc="ul", ncol=1, fontsize=4, frame=False,
            title="Indicators (n, r2)" , titlefontsize="small")

        cross.append([i,r2_train,r2_test])

        st.loc[train_id,"type"] = "Train"
        st.loc[test_id,"type"] = "Test"
        st.loc[train_id,"predicted"] = model.predicted_train.tolist()
        st.loc[test_id,"predicted"] = model.predicted_test.tolist()

        st.loc[y.index,"(obs-pred)/obs"] = (st.loc[y.index,y_target]-st.loc[y.index,'predicted'])/st.loc[y.index,y_target]
        st['(obs-pred)/obs'] = st['(obs-pred)/obs'].map(lambda n:'{:.2f}'.format(n) if not pd.isnull(n) else n)
        output_csv = 'cross_val/stations_train_test_'+str(i)+'.csv'
        st.to_csv(output_csv)

        # permutation_importance
        test_im = permutation_importance(model, model.x_test, model.y_test, n_repeats=10,\
        random_state=defkwargs.get('random_stat'), n_jobs=defkwargs.get('ncpus'))
        sorted_importances_idx = test_im.importances_mean.argsort()
        test_importances = pd.DataFrame(test_im.importances[sorted_importances_idx].T,\
        columns=X.columns[sorted_importances_idx])

        train_im = permutation_importance(model, model.x_train, model.y_train, n_repeats=10, \
        random_state=defkwargs.get('random_stat'), n_jobs=defkwargs.get('ncpus'))
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
            fig2.savefig(permutation_importances+'_split_'+str(i)+'.png')
    print('The cross-validation score are ',cross)
    # print('The mean value of the cross-validation score is ',np.mean(cross))
    return fig

def train_model_all(station_qbf, station_predictors, predictors_info, \
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


    stratify_col = 'combined_cat'
    # get the parent path from the output_csv
    output_path = os.path.dirname(output_csv)
    # below two files are saved at each run of training the model
    output_cross_val=output_path+"/"+"cross_val/train_test_cross_val_5fold.png"
    output_hyparameters=output_path+"/"+"cross_val/best_hyperparameters.txt"

    # prepare training target and predictors
    st,value_counts = stations_obs(station_qbf,koppen_color,stratify_col)
    cleaned_pred = predictors_process(predictors_info,station_predictors)
    cleaned_pred, qbf = _ix_overlap(cleaned_pred, st[st[stratify_col].isin(value_counts.index[value_counts >= 5])]["qbf"])
    cleaned_pred_cp = cleaned_pred.copy()

    # define an empty list to store the index for training data
    idx_train = set()
    model_importance = None
    model_performance = None
    models = []
    climate_df_final = None
    validation_df_final = None
    validation_df_QBFcat_final = None

    # model_training
    y_target = 'qbf'
    for i in np.arange(n_run): 
        best_model, model_df, validation_df, \
        validation_df_QBFcat, qbf_train_test = model_training(
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
        climate_df_final = df_none_check(climate_df_final, model_df)
        validation_df_final = df_none_check(validation_df_final, validation_df)
        validation_df_QBFcat_final = df_none_check(validation_df_QBFcat_final, validation_df_QBFcat)
    
        # Track training indices (use set instead of list comprehension)
        idx_train.update(qbf_train_test[qbf_train_test["type"] == 'Train'].index)

    print('The number of training data is ', str(len(set(idx_train))))
    print('The best model performance is \n', best_model_performance)
    # calculate the mean over all models
    model_importance /= n_run
    model_performance /= n_run

    # add the averaged value back to the best_model
    best_model = best_model_selected
    best_model.importances = model_importance
    best_model.performance = model_performance

    climate_df_final, validation_df_final, validation_df_QBFcat_final = [
        df_error_mean(df, n_run, source_df.pop('Type'))
        for df, source_df in [
            (climate_df_final, model_df),
            (validation_df_final, validation_df),
            (validation_df_QBFcat_final, validation_df_QBFcat),
        ]
    ]
    print('climate_df:',climate_df_final)
    print('validation_df:',validation_df_final)
    print('validation_df_QBFcat:',validation_df_QBFcat_final)

    # flag data that had been used in the training
    qbf_train_test["type"] = ["Train" if i in idx_train else "Test" for i in qbf_train_test.index]
    qbf_train_test[[stratify_col,'koppen_color']] = st[[stratify_col,'koppen_color']]
    qbf_train_test["koppen"] = st['koppen']
    # save the train,test of each time to csv
    qbf_train_test.to_csv(output_csv)

    if output_performance:
        # save output_performance to csv
        df_perf = best_model.performance.T
        df_perf.insert(0,'Type','Overall')
        df_perf = pd.concat([df_perf,climate_df_final,validation_df_final,validation_df_QBFcat_final])
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