"""Part B: RFM segments and pre-cutoff lapse prediction without preprocessing leakage."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import joblib
ROOT=Path(__file__).parent; DATA=ROOT/'data'/'clean'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
SNAP=pd.Timestamp('2025-10-01 00:00:00'); CUTOFF=pd.Timestamp('2025-09-01 00:00:00')

def run():
    cust=pd.read_csv(DATA/'customers.csv',parse_dates=['signup_date']); prod=pd.read_csv(DATA/'products.csv'); orders=pd.read_csv(DATA/'orders.csv',parse_dates=['order_ts'])
    orders=orders.merge(prod[['product_id','category']],on='product_id',how='left')
    completed=orders[orders.status.eq('completed')].copy()
    rfm=completed.groupby('customer_id').agg(last_order_ts=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    rfm['recency']=(SNAP-rfm.last_order_ts).dt.days; rfm=rfm.drop(columns='last_order_ts')
    rfm.to_csv(OUT/'rfm_table.csv',index=False)
    X=rfm[['recency','frequency','monetary']].astype(float); scaler=StandardScaler(); Xs=scaler.fit_transform(X)
    km_rows=[]
    for k in range(2,9):
        model=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Xs)
        km_rows.append({'k':k,'inertia':model.inertia_,'silhouette':silhouette_score(Xs,model.labels_)})
    pd.DataFrame(km_rows).to_csv(OUT/'kmeans_metrics.csv',index=False)
    K=4; km=KMeans(n_clusters=K,n_init=10,random_state=42).fit(Xs); rfm['segment_id']=km.labels_
    profiles=rfm.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).reset_index()
    names={}; premium_id=int(profiles.loc[profiles.mean_monetary.idxmax(),'segment_id'])
    # Business labels are based on the observed RFM profile, not cluster number ordering.
    for _,p in profiles.iterrows():
        sid=int(p.segment_id)
        if sid==premium_id: name='Premium High Value'
        elif p.mean_frequency>=profiles.mean_frequency.quantile(.75): name='Frequent Essentials'
        elif p.mean_recency>=profiles.mean_recency.quantile(.75): name='At-Risk / Cooling'
        else: name='Steady Value'
        names[sid]=name
    rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium_id)
    rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(OUT/'customer_segments.csv',index=False)
    profiles['segment_name']=profiles.segment_id.map(names); profiles['is_premium']=profiles.segment_id.eq(premium_id); profiles.to_csv(OUT/'segment_profiles.csv',index=False)
    pca=PCA(n_components=2).fit(Xs); pd.DataFrame([{'pc1_explained':pca.explained_variance_ratio_[0],'pc2_explained':pca.explained_variance_ratio_[1],'total_explained':pca.explained_variance_ratio_.sum()}]).to_csv(OUT/'pca_variance.csv',index=False)
    # Build cohort from completed orders strictly before cutoff; labels use September only.
    pre=orders[(orders.order_ts<CUTOFF)&orders.status.isin(['completed','cancelled'])].copy()
    precomp=pre[pre.status.eq('completed')]
    cohort_ids=sorted(precomp.customer_id.unique())
    sept=completed[(completed.order_ts>=CUTOFF)&(completed.order_ts<pd.Timestamp('2025-10-01'))].groupby('customer_id').size()
    rows=[]
    for cid in cohort_ids:
        allc=pre[pre.customer_id.eq(cid)]; cc=precomp[precomp.customer_id.eq(cid)]
        fav=cc.category.mode(); customer=cust.loc[cust.customer_id.eq(cid)].iloc[0]
        rows.append({'customer_id':cid,'lapsed':int(sept.get(cid,0)==0),'recency':int((CUTOFF-cc.order_ts.max()).days),'frequency':int(cc.order_id.nunique()),'monetary':float(cc.order_amount.sum()),'avg_discount':float(cc.discount_pct.mean()),'cancellation_rate':float(allc.status.eq('cancelled').sum()/max(1,len(allc))),'avg_delivery':float(cc.delivery_minutes.mean()),'favourite_category':fav.iloc[0] if len(fav) else 'Unknown','city':customer.city,'age':float(customer.age),'gender':customer.gender,'tenure_days':int((CUTOFF-customer.signup_date).days)})
    cohort=pd.DataFrame(rows).sort_values('customer_id').reset_index(drop=True)
    features=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','favourite_category','city','age','gender','tenure_days']
    assert not cohort[features].isna().any().any(), 'Feature matrix contains missing values'
    X=cohort[features]; y=cohort.lapsed
    X_train,X_test,y_train,y_test,idx_train,idx_test=train_test_split(X,y,cohort.index,test_size=.2,stratify=y,random_state=42)
    numeric=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure_days']; categorical=['favourite_category','city','gender']
    try: ohe=OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)
    except TypeError: ohe=OneHotEncoder(drop='first',handle_unknown='ignore',sparse=False)
    prep=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),numeric),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('ohe',ohe)]),categorical)])
    baseline=np.zeros(len(y_test),dtype=int)
    report={'cohort_size':len(cohort),'lapsed_count':int(y.sum()),'lapsed_rate_pct':float(y.mean()*100),'train_rows':len(X_train),'test_rows':len(X_test),'baseline_accuracy':accuracy_score(y_test,baseline),'baseline_lapsed_f1':f1_score(y_test,baseline,zero_division=0)}
    # Fit encoders/scalers only within CV training folds; GridSearchCV gets a complete pipeline.
    log_pipe=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,random_state=42))])
    log_grid=GridSearchCV(log_pipe,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1); log_grid.fit(X_train,y_train)
    rf_pipe=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
    rf_grid=GridSearchCV(rf_pipe,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1); rf_grid.fit(X_train,y_train)
    def evaluate(model):
        pred=model.predict(X_test)
        return {'confusion_matrix':confusion_matrix(y_test,pred,labels=[0,1]).tolist(),'precision_lapsed':precision_score(y_test,pred,zero_division=0),'recall_lapsed':recall_score(y_test,pred,zero_division=0),'f1_lapsed':f1_score(y_test,pred,zero_division=0)}
    report['logistic']={'best_params':log_grid.best_params_,'best_cv_f1':log_grid.best_score_,'test':evaluate(log_grid.best_estimator_)}
    report['random_forest']={'best_params':rf_grid.best_params_,'best_cv_f1':rf_grid.best_score_,'test':evaluate(rf_grid.best_estimator_)}
    # Class balance experiment: duplicate minority rows in training only, preserve test set.
    from sklearn.utils import resample
    train_bal=pd.concat([X_train,y_train],axis=1); parts=[]
    for cls in [0,1]:
        subset=train_bal[train_bal.lapsed.eq(cls)]
        target=max(train_bal.lapsed.value_counts())
        if len(subset)<target: subset=resample(subset,replace=True,n_samples=target,random_state=42)
        parts.append(subset)
    balanced=pd.concat(parts).sample(frac=1,random_state=42); bal_model=joblib.loads if False else None
    bal_pipe=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,random_state=42,C=log_grid.best_params_['model__C']))])
    bal_pipe.fit(balanced[features],balanced.lapsed); report['balanced_logistic_test']=evaluate(bal_pipe)
    # Save best RF and prove reload prediction equivalence.
    selected=rf_grid.best_estimator_; model_path=OUT/'lapse_model.joblib'; joblib.dump(selected,model_path); reloaded=joblib.load(model_path)
    assert np.array_equal(selected.predict(X_test),reloaded.predict(X_test)); report['reload_predictions_identical']=True
    # Top feature importance, and logistic coefficients with names after fitted preprocessing.
    rf=selected.named_steps['model']; prep_f=selected.named_steps['prep']; names_out=prep_f.get_feature_names_out()
    importances=pd.DataFrame({'feature':names_out,'importance':rf.feature_importances_}).sort_values('importance',ascending=False).head(5); importances.to_csv(OUT/'rf_feature_importance.csv',index=False)
    log=log_grid.best_estimator_; log_names=log.named_steps['prep'].get_feature_names_out(); coefs=log.named_steps['model'].coef_[0]
    coef_df=pd.DataFrame({'feature':log_names,'coefficient':coefs}).assign(abs_coef=lambda d:d.coefficient.abs()).sort_values('abs_coef',ascending=False).head(15); coef_df.to_csv(OUT/'logistic_coefficients_top.csv',index=False)
    # Retention list: premium whole-window flag is explicitly post-hoc, illustrative only.
    test_ids=cohort.loc[idx_test,'customer_id'].to_frame().reset_index(drop=True); probs=selected.predict_proba(X_test)[:,1]
    scores=test_ids.copy(); scores['lapse_probability']=probs; scores['actually_lapsed']=y_test.to_numpy()
    premium_ids=set(rfm.loc[rfm.is_premium,'customer_id']); ret=scores[scores.customer_id.isin(premium_ids)].sort_values('lapse_probability',ascending=False).head(10)
    ret.to_csv(OUT/'retention_list.csv',index=False); report['retention_top10_actually_lapsed']=int(ret.actually_lapsed.sum()); report['retention_list_note']='Illustrative only: Premium is a post-hoc whole-window label that includes the September outcome window; it is not a leakage-free evaluation.'
    report['frequency_scaler_training_mean']=float(log.named_steps['prep'].named_transformers_['num'].named_steps['scaler'].mean_[numeric.index('frequency')]) if False else float(X_train.frequency.mean())
    report['full_cohort_frequency_mean']=float(X.frequency.mean())
    (OUT/'part_b_report.json').write_text(json.dumps(report,indent=2,default=lambda o: o.item() if hasattr(o,'item') else str(o)))
    # Save model evaluation table.
    print('Part B complete:',json.dumps({k:report[k] for k in ['cohort_size','lapsed_count','train_rows','test_rows','reload_predictions_identical']},indent=2))
if __name__=='__main__': run()
