# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

# model used for feature importances
import lightgbm as lgb

# assessment
from sklearn.metrics import r2_score

# utility for early stopping with a validation set
from sklearn.model_selection import train_test_split

# visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# memory management
import gc

# utilities
from itertools import chain


class FeatureSelector():
    
    def __init__(self, data, label=None):
        
        self.data = data
        self.label = label
        
        if label is None:
            print('No label provided. Feature importance based methods are not available.')
            
            
        self.feature_importances = None
        self.record_zero_importance = None
        self.record_low_importance = None
        
        # Dictionary to hold removal operations
        self.ops = {}
    
    
    
    
    def identify_zero_importance(self, task, eval_metric=None, n_iterations=10, early_stopping = True):
        
        if early_stopping and eval_metric is None:
            raise ValueError("""eval metric must be provided with early stopping. Examples include "auc" for classification or
                             "l2" for regression.""")
        
        if self.label is None:
            raise ValueError("No training labels provided.")
            
        # Extract feature names
        feature_names = list(self.data.columns)
            
        # Convert to np array
        X = np.array(self.data)
        y = np.array(self.label).reshape((-1, ))
    
        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))
        
        
        # Iterate through each fold
        for _ in range(n_iterations):
                
            if task == 'classification':
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1)
                    
            elif task == 'regression':
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate = 0.05, verbose = -1)
                    
            else:
                raise ValueError('Task must be either "classification" or "regression"')
                    
            # If training using early stopping need a validation set
            if early_stopping:
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
    
                # Train the model with early stopping
                model.fit(X_train, y_train, eval_metric = eval_metric,
                              eval_set = [(X_test, y_test)],
                              early_stopping_rounds = 100, verbose = -1)
                y_pred = model.predict(X_test)
                print('accuracy:', r2_score(y_test, y_pred))
                    
                # Clean up memory
                gc.enable()
                del X_train, y_train, X_test, y_test, y_pred
                gc.collect()
                
            else:
                model.fit(X, y)
        
                        
            # Record the feature importances
            feature_importance_values += model.feature_importances_ / n_iterations          
            
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)
    
        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
        
        # Extract the features with zero importance
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]
        
        to_drop = list(record_zero_importance['feature'])

        self.feature_importances = feature_importances
        self.record_zero_importance = record_zero_importance
        self.ops['zero_importance'] = to_drop
        
        print('\n%d features with zero importance.\n' % len(self.ops['zero_importance']))
        
        
        
        
        
    def identify_low_importance(self, threshold):
            """
            threshold : float between 0 and 1
            
            """   
            
            self.threshold = threshold
            
            # The feature importances need to be calculated before running
            if self.feature_importances is None:
                raise NotImplementedError("""Feature importances have not yet been determined. 
                                             Call the `identify_zero_importance` method first.""")
            
                
            # Make sure most important features are on top
            self.feature_importances = self.feature_importances.sort_values('cumulative_importance')
    
            # Identify the features not needed to reach the cumulative_importance
            record_low_importance = self.feature_importances[self.feature_importances['cumulative_importance'] > threshold]
            
            to_drop = list(record_low_importance['feature'])
            
            self.record_low_importance = record_low_importance
            self.ops['low_importance'] = to_drop
            
            print('%d features required for cumulative importance of %0.2f.' % (len(self.feature_importances) -
                                                                            len(self.record_low_importance), self.threshold))
            print('%d features do not contribute to cumulative importance of %0.2f.\n' % (len(self.ops['low_importance']),
                                                                                          self.threshold))
            
    
    
    
            
            
    def remove(self, methods):
        """
        Remove the features from the data according to the specified methods.
            
        Parameters
        --------
        methods : 'all' or list of methods
        If methods == 'all', any methods that have identified features will be used
        Otherwise, only the specified methods will be used.
        Can be one of ['zero_importance', 'low_importance']
                    
        Return
        --------
        data : dataframe
        Dataframe with identified features removed
        
        """
        
        
        features_to_drop = []
        data = self.data
         
        if methods == 'all':
        
            print('{} methods have been run\n'.format(list(self.ops.keys())))
            
            # Find the unique features to drop
            features_to_drop = set(list(chain(*list(self.ops.values()))))
            
        else:
            # Iterate through the specified methods
            for method in methods:
                
                # Check to make sure the method has been run
                if method not in self.ops.keys():
                    raise NotImplementedError('%s method has not been run' % method)
                    
                # Append the features identified for removal
                else:
                    features_to_drop.append(self.ops[method])
            
            # Find the unique features to drop
            features_to_drop = set(list(chain(*list(self.ops.values()))))
        
        
        features_to_drop = list(features_to_drop)
        
          
        # Remove the features and return the data
        data = data.drop(columns = features_to_drop)
        self.removed_features = features_to_drop
        
        
        print('Removed %d features.' % len(features_to_drop))
        
        return data
        
        
        
        
        
    def plot_feature_importances(self, plot_n = 35, threshold = None):
        """
        Plots `plot_n` most important features and the cumulative importance of features.
        If `threshold` is provided, prints the number of features needed to reach `threshold` cumulative importance.

        Parameters
        --------
        
        plot_n : int, default = 15
        Number of most important features to plot. Defaults to 35 or the maximum number of features whichever is smaller
        
        threshold : float, between 0 and 1 default = None
        Threshold for printing information about cumulative importances
    
        """
            
        if self.record_zero_importance is None:
            raise NotImplementedError('Feature importances have not been determined. Run `idenfity_zero_importance`')
            
        # Need to adjust number of features if greater than the features in the data
        if plot_n > self.feature_importances.shape[0]:
            plot_n = self.feature_importances.shape[0] - 1
            
            
        # Make a horizontal bar chart of feature importances
        plt.figure(figsize = (10, 15))
        ax = plt.subplot()
        
        # Need to reverse the index to plot most important on top
        # There might be a more efficient method to accomplish this
        ax.barh(list(reversed(list(self.feature_importances.index[:plot_n]))), 
                self.feature_importances['normalized_importance'][:plot_n], 
                align = 'center', edgecolor = 'k')

        # Set the yticks and labels
        ax.set_yticks(list(reversed(list(self.feature_importances.index[:plot_n]))))
        ax.set_yticklabels(self.feature_importances['feature'][:plot_n], size = 12)

        # Plot labeling
        plt.xlabel('Normalized Importance', size = 16); plt.title('Feature Importances', size = 18)
        plt.show()

        # Cumulative importance plot
        plt.figure(figsize = (6, 4))
        plt.plot(list(range(1, len(self.feature_importances) + 1)), self.feature_importances['cumulative_importance'], 'r-')
        plt.xlabel('Number of Features', size = 14); plt.ylabel('Cumulative Importance', size = 14); 
        plt.title('Cumulative Feature Importance', size = 16);

        if threshold:

            # Index of minimum number of features needed for cumulative importance threshold
            # np.where returns the index so need to add 1 to have correct number
            importance_index = np.min(np.where(self.feature_importances['cumulative_importance'] > threshold))
            plt.vlines(x = importance_index + 1, ymin = 0, ymax = 1, linestyles='--', colors = 'blue')
            plt.show();

        print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))