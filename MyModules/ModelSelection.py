import itertools
from sklearn.metrics import mean_squared_error, r2_score


def best_selection(regr, X, y):
    best_subset_ = {'Predictors':[], 'NoPreditors':[], 'MSE': [], 'RSquared':[]} #'PredictorName':[], , 'Model':[]
    
    p_idx = list(range(X.shape[1]))
    for L in range(X.shape[1]+1):
        for subset in itertools.combinations(p_idx, L):
            if list(subset):
                x = X[:, subset]               
                regr.fit(x,y)
                
                best_subset_['Predictors'].append(list(subset))
                best_subset_['NoPreditors'].append(len(subset))
                best_subset_['MSE'].append(mean_squared_error(y, regr.predict(x)))
                best_subset_['RSquared'].append(r2_score(y, regr.predict(x)))
                #best_subset_['Model'].append(regr)
            else:
                x = np.ones((len(X),1))
                regr.fit(x, y)
                
                best_subset_['Predictors'].append(list(subset))
                best_subset_['NoPreditors'].append(len(subset))
                best_subset_['MSE'].append(mean_squared_error(y, regr.predict(x)))
                best_subset_['RSquared'].append(r2_score(y, regr.predict(x)))
                #best_subset_['Model'].append(regr)
    return best_subset_

###################  FORWARD STEP SELECTION #############################################################
def forward_selection(regr, X, y):
    p = X.shape[1]
    predictors_list = list(range(p))
    forward_selection_ =  {'k':[], 'BestModel':[], 'StepModel':[], 'MSE': [], 'RSquared':[]}    
    
    ## Step 1: Null Model M0 ###############
    x = np.ones((len(X),1))
    regr.fit(x, y)
    
    forward_selection_['k'].append(-1)
    forward_selection_['BestModel'].append([])
    forward_selection_['StepModel'].append([])
    forward_selection_['MSE'].append(mean_squared_error(y, regr.predict(x)))
    forward_selection_['RSquared'].append(r2_score(y, regr.predict(x)))
    
    model_p_idx = list() # 
    
    ## Step 2 ###############    
    for k in range(p):
        k_index = [i for i, v in enumerate(forward_selection_['k']) if v == k-1]
        mse = np.array(forward_selection_['MSE'])[k_index]
        r2 = np.array(forward_selection_['RSquared'])[k_index]
        best_index = k_index[np.argmin(mse)] # np.argmax(r2)
        
        model_p_idx.append(best_index)
        best_model = forward_selection_['StepModel'][best_index] 
        
        remaining_predictors = list(set(predictors_list) - set(best_model))
        
        for j in remaining_predictors:
            step_model = best_model.copy()
            step_model.append(j)
            x = X[:,step_model]
            regr.fit(x, y)
            
            ## Appending info to dictionary
            forward_selection_['k'].append(k)
            forward_selection_['BestModel'].append(best_model)
            forward_selection_['StepModel'].append(step_model)
            forward_selection_['MSE'].append(mean_squared_error(y, regr.predict(x)))
            forward_selection_['RSquared'].append(r2_score(y, regr.predict(x)))           
        
        #print(k,k_index, best_index, forward_selection_['MSE'][best_index])
    last_idx = [i for i, v in enumerate(forward_selection_['k']) if v == k][0]
    model_p_idx.append(last_idx)
    #print(model_p_idx)
    return  forward_selection_, model_p_idx  


###################################################################################################

def backward_selection(regr, X, y):
    p = X.shape[1]
    predictors_list = list(range(p))
    backward_selection_ =  {'k':[], 'BestModel':[], 'StepModel':[], 'MSE': [], 'RSquared':[]}    
    
    ##  Full Model M0 ###############
    regr.fit(X, y) 
    
    backward_selection_['k'].append(p)
    backward_selection_['BestModel'].append(predictors_list)
    backward_selection_['StepModel'].append(predictors_list)
    backward_selection_['MSE'].append(mean_squared_error(y, regr.predict(X)))
    backward_selection_['RSquared'].append(r2_score(y, regr.predict(X)))
    
    model_p_idx = list() # 
    
    ## Step 2 ###############    
    for k in range(p,0,-1):
        k_index = [i for i, v in enumerate(backward_selection_['k']) if v == k]
        mse = np.array(backward_selection_['MSE'])[k_index]
        r2 = np.array(backward_selection_['RSquared'])[k_index]

        best_index = k_index[np.argmin(mse)] # np.argmax(r2)
        model_p_idx.append(best_index)
        best_model = backward_selection_['StepModel'][best_index] 

        for j in best_model:
            step_model = best_model.copy()
            step_model.remove(j) ## removing one perdictor from Mk
            
            x = X[:,step_model]
            if x.shape[1] == 0:
                x = x = np.ones((len(X),1))##  Null Model M0 ###############
            regr.fit(x, y)
            
            ## Appending info to dictionary
            backward_selection_['k'].append(k-1)
            backward_selection_['BestModel'].append(best_model)
            backward_selection_['StepModel'].append(step_model)
            backward_selection_['MSE'].append(mean_squared_error(y, regr.predict(x)))
            backward_selection_['RSquared'].append(r2_score(y, regr.predict(x)))           
        
        #print(k,k_index, best_index, backward_selection_['MSE'][best_index])

    last_idx = [i for i, v in enumerate(backward_selection_['k']) if v == k][0]
    model_p_idx.append(last_idx)
    #print(model_p_idx)
    return  backward_selection_, model_p_idx
	
