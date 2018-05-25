import os
from flask import Flask, request
from werkzeug.utils import secure_filename
import pandas as pd
from itertools import combinations
import datetime as dt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import boxcox,skew
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:/Users/RB287JD/Desktop/Dhivya_upload/uploads/'


@app.route('/getfile', methods=['POST'])
def getfile():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    bond_data = pd.read_excel(filepath,sheet_name='Debt',skiprows=3).set_index('ISIN')
    financial_data = pd.read_excel(filepath,sheet_name='Financials').set_index('ISIN')
    
    financial_data = financial_data.rename(columns=
                     {'IQ_TOTAL_ASSETS':'Assets','IQ_TOTAL_DEBT':'Total debt',
                     'IQ_TOTAL_REV':'Revenues','IQ_EBIT':'EBIT','IQ_NI':'Net income'})
    
    metrics = ['Assets','Total debt','Revenues','EBIT','Net income']
    for col in metrics:
        financial_data[col] = pd.to_numeric(financial_data[col],'coerce')
    for combo in combinations(metrics,2):
        col_name = combo[1]+'_'+combo[0]
        financial_data[col_name] = financial_data[combo[1]] / financial_data[combo[0]]
    
    
    def determine_skewed_vars(X,cont_features):
        X = X.copy()
                   
        # compute skew and do Box-Cox transformation
        skewed_features = X[cont_features].apply(lambda x: skew(x.dropna()))
        #print("\nSkew in numeric features:")
        #print(skewed_features)
        
        # transform features with skew > 0.25 (this can be varied to find optimal value)
        skewed_features = skewed_features[skewed_features > 0.05]
        skewed_features = skewed_features.index
        boxcox_lambdas = []
        add_value_list = []
        
        for feature in skewed_features:
            if X[feature].min() < 0:
                add_value = np.ceil(abs(X[feature].min()))
            else:
                add_value = 0
            X[feature] = X[feature] + add_value
            X[feature], lam = boxcox(X[feature])
            boxcox_lambdas.append(lam)
            add_value_list.append(add_value*10)
        
        return skewed_features,boxcox_lambdas,add_value_list
    
    def scale_data(X, scaler=None):
        X = X.copy()
            
        if not scaler:
            scaler = StandardScaler()
            scaler.fit(X)
        X = scaler.transform(X)
        return X, scaler
    
    def standard_feature_engineering(X,cat_features,skewed_features,
                                     boxcox_lambdas,add_value_list,scaler=None,pca=None):
        X = X.copy()
        
        if type(X) == pd.core.frame.DataFrame:
            feature_names = X.columns
        
        ix = X.index
        
        for feature,lmbda,add_value in zip(skewed_features,boxcox_lambdas,add_value_list):
            X[feature] = X[feature] + add_value
            X[feature] = boxcox(X[feature],lmbda)
    
        # factorize categorical features
        for feat in cat_features:
            X[feat] = pd.factorize(X[feat], sort=True)[0]
        
        #scales data
        if scaler!=False:
            X,scaler = scale_data(X,scaler)
            X = pd.DataFrame(X,index=ix,columns=feature_names)
        if pca:
            if type(pca) == int:
                if pca>0:
                    pca = PCA(n_components=pca)
                    X = pd.DataFrame(pca.fit_transform(X),index=ix)
            else:
                X = pca.transform(X)
                X = pd.DataFrame(X,index=ix)
    
        return X,scaler,pca
    
    
    
    cols = ['Coupon','TTM','Sector Level 3','Composite Rating','Ticker','OAS vs Govt','Assets','Revenues']
    
    def show_error(X_test,y_test,model):
        error_sq = (model.predict(X_test) - y_test)**2
        #print(model)
        #print('RMSE (Mean): {}'.format(np.sqrt(error_sq.mean())))
        return np.sqrt(error_sq.mean())
    
    
    def gen_industry_distance_var(df):
        df = df.copy()
        df['OAS Rating Mean'] = df.groupby(['Composite Rating',pd.qcut(df['TTM'],quantile_num)])['OAS vs Govt'].transform('mean')
        df['Greater than Mean'] = df['OAS vs Govt'] > df['OAS Rating Mean']
        tmp1 = df.groupby(['Ticker','Sector Level 3'])['Greater than Mean'].agg(['sum','count'])
        tmp1['Bayesian Prob'] = tmp1.apply(lambda x: beta(x['sum']+10,x['count']-x['sum']+10).cdf(0.5),axis=1)
        tmp2 = tmp1.groupby(level=1)['Bayesian Prob'].agg(['mean','count'])
        tmp2['sum'] = (tmp2['mean'] * tmp2['count']).astype(int)
        tmp3 = tmp2.apply(lambda x: beta(x['sum']+10,x['count']-x['sum']+10).cdf(0.5),axis=1)
        tmp3.name = 'Industry Distance'
        return tmp3
    
    
    def gen_train_test_split(reg_data,sample_size,scale=None,pca_elements=0,drop_company=False):
        
        
        if drop_company==True:
            test_sample = reg_data['Ticker'].drop_duplicates().sample(sample_size).values
            msk = reg_data['Ticker'].isin(test_sample)
        else:
            test_sample = reg_data.sample(sample_size)
            msk = reg_data.index.isin(test_sample.index)        
        reg_data = reg_data.drop('Ticker',axis=1)
        
        X = reg_data.loc[~msk].drop('OAS vs Govt',axis=1)
        
        y = reg_data.loc[~msk,'OAS vs Govt']
        X_test = reg_data.loc[msk].drop('OAS vs Govt',axis=1)
        
        y_test = reg_data.loc[msk,'OAS vs Govt']
        cat_features = []
        cont_features = [col for col in X.columns if col not in cat_features]
        
        skewed_features,boxcox_lambdas,add_value_list = determine_skewed_vars(X,cont_features)
        
        X_train,scaler,pca = standard_feature_engineering(X,cat_features,skewed_features,boxcox_lambdas,add_value_list,scale,pca_elements)
        X_test,trsh,trsh2 = standard_feature_engineering(X_test,cat_features,skewed_features,boxcox_lambdas,add_value_list,scaler,pca)
        return X_train,X_test,y,y_test
    
    
    def assemble_data(bond_data,financial_data,bond_rating='All'):
        df = bond_data.join(financial_data).dropna()
        msk = df['Assets'] > 0
        df = df[msk]
        #df['Maturity Date'] = df['Maturity'].apply(parse)
        df['Maturity Date'] = df['Maturity']
        df['TTM'] = (df['Maturity Date'] - dt.date(2016,12,31))/np.timedelta64(1,'D')/365.25
        
        msk = df['OAS vs Govt'] <= df['OAS vs Govt'].quantile([0.99]).iloc[0]
        df = df[msk]
        
        msk = df['Composite Rating'].isin(['BBB1','BBB2','BBB3','A1','A2','A3','AA1','AA2','AA3','AAA'])
        if bond_rating=='IG':
            df = df[msk]
        elif bond_rating=='HY':
            df = df[~msk]
        
        msk = (df['Type'] == 'SENR') & (df['Sector Level 2']=='Industrials')
        reg_data = df.loc[msk,cols].join(df[financial_data.columns[-10:].tolist()]).reset_index()
            
        industry_var = gen_industry_distance_var(reg_data)
        
        reg_data = reg_data.merge(industry_var.reset_index(),on='Sector Level 3')
        reg_data = reg_data.drop(['Sector Level 3','Composite Rating'],axis=1)
        reg_data = reg_data.set_index('ISIN')
        
        return reg_data
    
    
    def run_CV(reg_data,sample_size,mod,scale=None,pca_elements=0):
        X_train,X_test,y,y_test = gen_train_test_split(reg_data,sample_size,scale,pca_elements)
        #mod = KNeighborsRegressor(n_neighbors=num_neighbors)
        mod.fit(X_train,y)
        #rf= RandomForestRegressor(min_samples_leaf=5)
        #rf.fit(X_train,y)
        rmse = show_error(X_test,y_test,mod)
        X_test.loc[:,'OAS_pr'] = mod.predict(X_test)
        X_test['OAS'] = y_test
        X_test['Error'] = X_test['OAS_pr'] - X_test['OAS'] 
        X_test['Error_Sq'] = (X_test['OAS_pr'] - X_test['OAS'])**2
    
    
        return rmse,X_test[['OAS_pr','OAS','Error','Error_Sq']]
    
    
    def test_cusips(reg_data,cusips,scale=None,pca_elements=0,drop_company=False):
        reg_data = reg_data.drop('Ticker',axis=1)
        msk = reg_data.index.isin(cusips)        
        
        X = reg_data.loc[~msk].drop('OAS vs Govt',axis=1)
        
        y = reg_data.loc[~msk,'OAS vs Govt']
        X_test = reg_data.loc[msk].drop('OAS vs Govt',axis=1)
        
        y_test = reg_data.loc[msk,'OAS vs Govt']
        cat_features = []
        cont_features = [col for col in X.columns if col not in cat_features]
        
        skewed_features,boxcox_lambdas,add_value_list = determine_skewed_vars(X,cont_features)
        
        X_train,scaler,pca = standard_feature_engineering(X,cat_features,skewed_features,boxcox_lambdas,add_value_list,scale,pca_elements)
        X_test,trsh,trsh2 = standard_feature_engineering(X_test,cat_features,skewed_features,boxcox_lambdas,add_value_list,scaler,pca)
        return X_train,X_test,y,y_test
    
    
    def test_CUSIPS(reg_data,cusips,mod,scale=None,pca_elements=0):
        X_train,X_test,y,y_test = test_cusips(reg_data,cusips,scale,pca_elements)
        #mod = KNeighborsRegressor(n_neighbors=num_neighbors)
        mod.fit(X_train,y)
        #rf= RandomForestRegressor(min_samples_leaf=5)
        #rf.fit(X_train,y)
        rmse = show_error(X_test,y_test,mod)
        X_test.loc[:,'OAS_pr'] = mod.predict(X_test)
        X_test['OAS'] = y_test
        X_test['Error'] = X_test['OAS_pr'] - X_test['OAS'] 
        X_test['Error_Sq'] = (X_test['OAS_pr'] - X_test['OAS'])**2
    
    
        return rmse,X_test[['OAS_pr','OAS','Error','Error_Sq']]
    
    
    quantile_num = 10
    reg_data = assemble_data(bond_data,financial_data)
    X_train,X_test,y,y_test = gen_train_test_split(reg_data,1)
    mod = KNeighborsRegressor(n_neighbors=5,weights='distance',metric='minkowski')
    
    cusip = request.form['text']
    cusip = [x.strip() for x in cusip.split(',')]
    
    r = test_CUSIPS(reg_data,cusip,mod)[1]   
    return str(r)

    
if __name__ == '__main__':
    app.run(host = '0.0.0.0')
    
    


