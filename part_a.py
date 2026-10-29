"""Part A: audit, clean, persist and query the synthetic exports."""
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
ROOT=Path(__file__).parent
RAW=ROOT/'data'/'raw'; CLEAN=ROOT/'data'/'clean'; OUT=ROOT/'outputs'; SQL=ROOT/'sql'
for p in (CLEAN,OUT,SQL): p.mkdir(parents=True,exist_ok=True)

def run():
    customers=pd.read_csv(RAW/'customers.csv',dtype={'age':'string','gender':'string'})
    products=pd.read_csv(RAW/'products.csv')
    orders=pd.read_csv(RAW/'orders.csv',keep_default_na=False)
    tickets=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
    # Audit table includes shape, dtype, missing count and percentage for each source column.
    audits=[]
    for name,df in [('customers',customers),('products',products),('orders',orders),('tickets',tickets)]:
        for col in df.columns:
            s=df[col]
            missing=(s.astype('string').str.strip().eq('') | s.isna())
            audits.append({'table':name,'rows':len(df),'columns':len(df.columns),'column':col,'dtype':str(s.dtype),'missing_count':int(missing.sum()),'missing_pct':round(float(missing.mean()*100),2)})
    pd.DataFrame(audits).to_csv(OUT/'raw_audit.csv',index=False)
    raw_order_audit={
      'raw_orders_rows':len(orders),'exact_duplicate_rows':int(orders.duplicated().sum()),
      'pending_rows':int(orders.status.eq('pending').sum()),'distinct_order_id':int(orders.order_id.nunique()),
      'nonnumeric_amounts':int(pd.to_numeric(orders.order_amount,errors='coerce').isna().sum()),
      'negative_delivery_minutes':int(pd.to_numeric(orders.delivery_minutes,errors='coerce').lt(0).sum()),
      'blank_delivery_minutes':int(orders.delivery_minutes.astype(str).str.strip().eq('').sum()),
      'blank_age':int(customers.age.fillna('').astype(str).str.strip().eq('').sum()),
      'age_outside_10_100':int(pd.to_numeric(customers.age,errors='coerce').pipe(lambda s: ((s<10)|(s>100)).sum())),
      'raw_city_spellings':int(customers.city.nunique()),'raw_gender_spellings_nonblank':int(customers.gender.fillna('').astype(str).str.strip().replace('',np.nan).nunique()),
      'blank_gender':int(customers.gender.fillna('').astype(str).str.strip().eq('').sum())}
    pd.DataFrame([raw_order_audit]).to_csv(OUT/'raw_audit_checkpoints.csv',index=False)
    # Customers: normalize city/gender; impossible ages become null then city-median imputation.
    customers['city']=customers.city.astype(str).str.strip().str.title()
    customers['gender']=customers.gender.fillna('').astype(str).str.strip().str.lower().map({'male':'male','m':'male','female':'female','f':'female'}).fillna('unknown')
    customers['age']=pd.to_numeric(customers.age,errors='coerce').astype(float)
    customers.loc[~customers.age.between(10,100),'age']=np.nan
    customers['age']=customers['age'].fillna(customers.groupby('city').age.transform('median')).round().astype(int)
    customers['signup_date']=pd.to_datetime(customers.signup_date)
    # Products
    products['price_inr']=pd.to_numeric(products.price_inr,errors='raise')
    # Orders: timestamp, exact dedup, latest record per order id, amount and delivery imputations.
    orders['order_ts']=pd.to_datetime(orders.order_ts)
    orders=orders.drop_duplicates().copy()
    orders=orders.sort_values('order_ts').drop_duplicates('order_id',keep='last').copy()
    for c in ['quantity','discount_pct','order_amount','delivery_minutes']:
        orders[c]=pd.to_numeric(orders[c].replace('',np.nan),errors='coerce')
    orders=orders.merge(products[['product_id','price_inr','category']],on='product_id',how='left',validate='many_to_one')
    missing_amount=orders.order_amount.isna()
    orders.loc[missing_amount,'order_amount']=(orders.loc[missing_amount,'price_inr']*orders.loc[missing_amount,'quantity']*(1-orders.loc[missing_amount,'discount_pct']/100)).round(2)
    orders.loc[orders.delivery_minutes<0,'delivery_minutes']=np.nan
    comp=orders.status.eq('completed')
    category_median=orders.loc[comp].groupby('category').delivery_minutes.transform('median')
    orders.loc[comp,'delivery_minutes']=orders.loc[comp,'delivery_minutes'].fillna(category_median)
    orders=orders.drop(columns=['price_inr','category'])
    # Stable text timestamp format in CSV and SQLite.
    orders['order_ts']=orders.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
    customers['signup_date']=customers.signup_date.dt.strftime('%Y-%m-%d')
    customers.to_csv(CLEAN/'customers.csv',index=False); products.to_csv(CLEAN/'products.csv',index=False); orders.to_csv(CLEAN/'orders.csv',index=False)
    db=ROOT/'data'/'instamart.db'; db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db) as con:
        customers.to_sql('customers',con,if_exists='replace',index=False)
        products.to_sql('products',con,if_exists='replace',index=False)
        orders.to_sql('orders',con,if_exists='replace',index=False)
    # SQL query pack
    queries='''-- Q1 WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS spend_over_500 FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000;
-- Q2 revenue by city/category
SELECT c.city,p.category,SUM(o.order_amount) AS revenue_inr FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;
-- Q3 products above average catalogue price
SELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products);
-- Q4 customers without completed orders
SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;
-- Q5 CTE total spend and average spend
WITH spend AS (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total_spend) avg_total FROM spend) SELECT spend.* FROM spend CROSS JOIN avg_spend WHERE total_spend>avg_total;
-- Q6 ranking functions by city
WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT *,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC,customer_id) AS row_num FROM spend;
-- Q7 running daily revenue
WITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day ROWS UNBOUNDED PRECEDING) cumulative_revenue FROM daily ORDER BY day;
'''
    (SQL/'queries.sql').write_text(queries,encoding='utf-8')
    # Query outputs and cross-checks
    # Remove SQL comments before splitting statements.
    sql_without_comments='\n'.join(line for line in queries.splitlines() if not line.strip().startswith('--'))
    cleaned=[s.strip() for s in sql_without_comments.split(';') if s.strip()]
    names=['q1','q2','q3','q4','q5','q6','q7']
    with sqlite3.connect(db) as con:
        for name,q in zip(names,cleaned): pd.read_sql_query(q,con).to_csv(OUT/f'{name}.csv',index=False)
    # Explore distribution, correlation, city summary and pivot grand total.
    comp_df=orders[orders.status.eq('completed')].copy()
    comp_df['order_amount']=pd.to_numeric(comp_df.order_amount)
    q1,q3=comp_df.order_amount.quantile([.25,.75]); iqr=q3-q1; lo=q1-1.5*iqr; hi=q3+1.5*iqr
    stats={'mean':comp_df.order_amount.mean(),'median':comp_df.order_amount.median(),'skewness':comp_df.order_amount.skew(),'q1':q1,'q3':q3,'lower_fence':lo,'upper_fence':hi,'outliers_below':int((comp_df.order_amount<lo).sum()),'outliers_above':int((comp_df.order_amount>hi).sum())}
    pd.DataFrame([stats]).to_csv(OUT/'amount_distribution.csv',index=False)
    with sqlite3.connect(db) as con:
        j=pd.read_sql_query("SELECT c.city,p.category,o.order_amount FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed'",con)
    pivot=pd.pivot_table(j,index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True)
    pivot.to_csv(OUT/'city_category_pivot.csv')
    city=comp_df.merge(customers[['customer_id','city']],on='customer_id').groupby('city').order_amount.agg(total='sum',mean='mean',count='count').reset_index(); city.to_csv(OUT/'city_spend_summary.csv',index=False)
    corr=comp_df[['order_amount','discount_pct','quantity','delivery_minutes']].corr(); corr.to_csv(OUT/'correlation_matrix.csv')
    # Validation: eight-plus checkpoints and exact message on success.
    assert len(customers)==1200 and customers.city.nunique()==6 and customers.age.notna().all()
    assert set(customers.gender.unique()) <= {'female','male','unknown'}
    assert len(orders)==12309 and orders.order_id.nunique()==12309
    assert not orders.status.eq('pending').any() and orders.order_amount.notna().all()
    assert int(orders.status.eq('completed').sum())==11110 and int(orders.status.eq('cancelled').sum())==1199
    assert orders.loc[orders.status.eq('completed'),'delivery_minutes'].notna().all()
    assert orders.loc[orders.status.eq('cancelled'),'delivery_minutes'].isna().all()
    assert round(orders.loc[orders.status.eq('completed'),'order_amount'].sum(),2)==2998668.80
    print('ALL CHECKPOINTS PASSED')
if __name__=='__main__': run()
