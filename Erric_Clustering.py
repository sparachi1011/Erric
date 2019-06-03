# -*- coding: utf-8 -*-
"""
Created on Thu May 23 20:23:10 2019

@author: sparachi
"""
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans  # For clustering
from sklearn.mixture import GaussianMixture  # For GMM clustering
from sklearn.preprocessing import StandardScaler


def viz_corr(in_data):
    """Calculate the correlation of the above variables 
        using Pearson Correlation"""
    try:
        plt.figure(figsize=(10, 10))
        corelation = in_data.corr()
# sns.heatmap(corelation, square = True) #Plot the correlation as heat map
        # Plot the correlation as heat map with Cor_values
        sns.heatmap(corelation, annot=True, cmap=plt.cm.Reds)
        plt.show()
    except Exception as e:
        print('viz_corr function failed', e)


def preprocess_data(in_data):
    try:
        # Ignoring unwanted features:::::::::::::::
        all_cols = in_data.columns.tolist()
        ignore_colums = ['beat.hostname', '@timestamp', 'system.network.name', 'ingestion_end', 'ingestion_start',
                         'num_records_ingested_per_sec', 'num_records_processed_per_sec', 'system.process.name',
                         'training_bucket_size', 'training_frequency', 'system.cpu.cores',
                         'beat.name', 'system.filesystem.mount_point']
        scale_columns = [
            item for item in all_cols if item not in ignore_colums]
        in_data_ignore = pd.DataFrame(data=in_data, columns=scale_columns)
#        in_data['@timestamp']=str(in_data.loc[:,'@timestamp']).replace("Z", ',').replace("T", ',')
#        in_data['@timestamp'] = pd.to_datetime(in_data['@timestamp'])

 # Imputation of missing values:::::::::::::::::

        for col in in_data_ignore:
            if in_data_ignore[col].isnull().sum() > 0:
                if str(in_data_ignore[col].dtype) == 'category':
                    in_data_ignore[col] = in_data_ignore[col].fillna(
                        value=in_data_ignore[col].mode()[0])
                elif (str(in_data_ignore[col].dtype) == 'int64'):
                    in_data_ignore[col] = in_data_ignore[col].fillna(
                        value=in_data_ignore[col].mean())
                elif (str(in_data_ignore[col].dtype) == 'float64'):
                    in_data_ignore[col] = in_data_ignore[col].fillna(
                        value=in_data_ignore[col].mean())
                elif (str(in_data_ignore[col].dtype) == 'object'):
                    in_data_ignore[col] = in_data_ignore[col].fillna(
                        value=in_data_ignore[col].mode()[0])
            else:
                pass

# Scaling of data:::::::::::::::::::::::

        ss = StandardScaler()

        in_data_ignore[in_data_ignore.columns] = ss.fit_transform(
            in_data_ignore[in_data_ignore.columns])

#        for col in scale_columns:
#            in_data[col] = in_data_ignore[col]

# Feature Encoding::::::::::::
        #from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#        encode_colums=['beat.hostname','system.network.name','beat.name','system.process.name']   ###','system.filesystem.mount_point'
#
#        #### imputing NAN of Categorical features with value='PreProcessedImputation'
#        for i in encode_colums:
#            in_data.loc[:,i]=in_data.loc[:,i].fillna(value='PreProcessedImputation')

        # le=LabelEncoder()
        # for i in encode_colums:
        #    in_data.loc[:,i]=le.fit_transform(in_data.loc[:,i])
        #
        # ohe=OneHotEncoder(categorical_features=in_data.loc[:,encode_colums])
        # in_data=ohe.fit_transform(in_data).toarray()
        # for i in encode_colums:
        #    in_data.loc[:,i]=ohe.fit_transform(in_data.loc[:,i])

        #from keras.utils import to_categorical
        #in_data = to_categorical(in_data)
        #in_data=pd.get_dummies(data=in_data, columns=encode_colums,drop_first=False)
        return in_data_ignore
    except Exception as e:
        print('preprocess_data function failed,', e)


def cluster_count(in_data_ignore):
    # Identifying number of Clusters K means Clustering
    #    in_data_ignore = preprocess_data(in_data)
    try:
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++',
                            max_iter=300, n_init=10, random_state=123)
            kmeans.fit(in_data_ignore)
            wcss.append(kmeans.inertia_)
        q1, q3 = np.percentile(
            wcss, [25, 75])
        no_of_clusters = len(
            list(filter(lambda x: x == True, (wcss > np.mean(wcss)))))  # q3))))
#        print(kmeans.n_iter_ )
#        print(wcss)
        """
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.show()
        """
        return no_of_clusters  # ,in_data_ignore
    except Exception as e:
        print('cluster_count function failed', e)


def apply_kmeans(in_data_ignore):
    # K means Clustering
    try:
        no_of_clusters = cluster_count(in_data_ignore)

        def doKmeans(X, nclust=no_of_clusters):
            model = KMeans(nclust, init='k-means++', max_iter=300,
                           n_init=10, random_state=123)
            model.fit(X)
            clust_labels = model.predict(X)
            cent = model.cluster_centers_
            return (clust_labels, cent)

        clust_labels, cent = doKmeans(in_data_ignore, no_of_clusters)
        kmeans = pd.DataFrame(clust_labels)
        in_data_ignore.insert((in_data_ignore.shape[1]), 'kmeans', kmeans)
        """
        #Plot the clusters obtained using k means
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(in_data_ignore['system.filesystem.available'],in_data_ignore['system.fsstat.total_size.used'],
                             c=kmeans[0],s=50)
        ax.set_title('K-Means Clustering')
        ax.set_xlabel('File Sys Availble')
        ax.set_ylabel('Fstat Tot Size')
        plt.colorbar(scatter)
        """
        clustered_data = []
        for i in list(range(no_of_clusters)):
            cluster_name = 'cluster_'+str(i)
            clustered_data.append(cluster_name)
            clustered_data[i] = pd.DataFrame(
                data=(in_data_ignore.loc[in_data_ignore['kmeans'] == float(i)]))
        return clustered_data
    except Exception as e:
        print('apply_kmeans function failed', e)


def apply_gaussianmix(in_data_ignore):
    # Guassian Mixture Modelling
    try:
        no_of_clusters = cluster_count(in_data_ignore)

        def doGMM(X, nclust=no_of_clusters):
            model = GaussianMixture(
                n_components=nclust, init_params='kmeans', max_iter=300,
                n_init=10, random_state=123, covariance_type='full')
            model.fit(X)
            clust_labels3 = model.predict(X)
            return (clust_labels3)

        clust_labels3 = doGMM(in_data_ignore, no_of_clusters)
        gmm = pd.DataFrame(clust_labels3)
        in_data_ignore.insert((in_data_ignore.shape[1]), 'gmm', gmm)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(in_data_ignore['system.filesystem.available'],in_data_ignore['system.fsstat.total_size.used'],
                             c=gmm[0],s=50)
        ax.set_title('Guassian Mixture Modelling')
        ax.set_xlabel('File Sys Availble')
        ax.set_ylabel('Fstat Tot Size')
        plt.colorbar(scatter)
        """
        clustered_data = []
        for i in list(range(no_of_clusters)):
            cluster_name = 'cluster_'+str(i)
            clustered_data.append(cluster_name)
            clustered_data[i] = pd.DataFrame(
                data=(in_data_ignore.loc[in_data_ignore['gmm'] == float(i)]))
        # Plotting the cluster data obtained using GMM
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(in_data_ignore['system.filesystem.available'],in_data_ignore['system.fsstat.total_size.used'],
                             c=gmm[0],s=50)
        ax.set_title('Guassian Mixture Modelling')
        ax.set_xlabel('File Sys Availble')
        ax.set_ylabel('Fstat Tot Size')
        plt.colorbar(scatter)
        """
        return clustered_data
    except Exception as e:
        print('apply_gaussianmix function failed', e)


def construct_anomaly(in_data, in_data_ignore):
    try:
        final_anomaly = pd.DataFrame()
        clustered_data = apply_gaussianmix(in_data_ignore)
        final_anomaies_count = []

        def high_clustor_anomaly(clustered_data, in_data):
            try:
                higher_cluster = clustered_data[-1]
                higher_cluster = higher_cluster.drop(
                    columns=[str((higher_cluster.columns.values)[-1])])  # to remove gmm column
                high_anomaly_df = pd.DataFrame()
                high_anomaly_counts = []
                for col in higher_cluster.columns:
                    if higher_cluster.loc[:, str(col)].max() > 0:
                        df_index = pd.DataFrame(data=(higher_cluster.loc[:, str(
                            col)] > higher_cluster.loc[:, str(col)].std()))
                        df_index = df_index.rename(columns={0: str(col)})
                        req_index = df_index.loc[df_index[str(
                            col)] == True].index.tolist()
                        high_anomaly_counts.append(
                            'Anomaies in high clustor :'+str(col)+'::'+str(len(req_index)))
                        col1_df = pd.DataFrame(
                            data=in_data.loc[req_index, ['@timestamp', str(col)]])
                        high_anomaly_df = high_anomaly_df.append(
                            col1_df, ignore_index=False)
                    else:
                        q1, q3 = np.percentile(
                            higher_cluster.loc[:, str(col)], [10, 90])
                        df_index = pd.DataFrame(
                            data=(higher_cluster.loc[:, str(col)].values > float(q1)))
                        df_index = df_index.rename(columns={0: str(col)})
                        req_index = df_index.loc[df_index[str(
                            col)] == True].index.tolist()
                        high_anomaly_counts.append(
                            'Anomaies in '+str(col)+'::'+str(len(req_index)))
                        col1_df = pd.DataFrame(
                            data=in_data.loc[req_index, ['@timestamp', str(col)]])
#
                        high_anomaly_df = high_anomaly_df.append(
                            col1_df, ignore_index=False)
                high_anomaly_df_index = high_anomaly_df.index.tolist()
                additional_column_df = pd.DataFrame(
                    data=in_data, columns=['beat.hostname', '@timestamp'], index=high_anomaly_df_index)
                for new_col in additional_column_df.columns:
                    high_anomaly_df[str(
                        new_col)] = additional_column_df[str(new_col)]
                return high_anomaly_df, high_anomaly_counts
            except Exception as e:
                print('high_clustor_anomaly function failed!', e)

        def low_clustor_anomaly(clustered_data, in_data):
            try:
                lower_cluster = clustered_data[0]
                lower_cluster = lower_cluster.drop(
                    columns=[str((lower_cluster.columns.values)[-1])])
                low_anomaly_df = pd.DataFrame()
                low_anomaly_counts = []
                for col in lower_cluster.columns:
                    if lower_cluster.loc[:, str(col)].max() > 0:
                        df_index = pd.DataFrame(data=(((lower_cluster.loc[:, str(col)]) > (
                            (lower_cluster.loc[:, str(col)].std())))))  # ,lower_cluster.loc[:, str(col)])-
                        df_index = df_index.rename(columns={0: str(col)})
                        req_index = df_index.loc[df_index[str(
                            col)] == True].index.tolist()
                        low_anomaly_counts.append(
                            'Anomaies in low clustor :'+str(col)+'::'+str(len(req_index)))
                        col1_df = pd.DataFrame(
                            data=in_data.loc[req_index, ['@timestamp', str(col)]])
                        low_anomaly_df = low_anomaly_df.append(
                            col1_df, ignore_index=False)
                    else:
                        q1, q3 = np.percentile(
                            lower_cluster.loc[:, str(col)], [10, 80])
                        df_index = pd.DataFrame(
                            data=(lower_cluster.loc[:, str(col)].values > float(q1)))
                        df_index = df_index.rename(columns={0: str(col)})
                        req_index = df_index.loc[df_index[str(
                            col)] == True].index.tolist()
                        low_anomaly_counts.append(
                            'Anomaies in '+str(col)+'::'+str(len(req_index)))
                        col1_df = pd.DataFrame(
                            data=in_data.loc[req_index, ['@timestamp', str(col)]])
                        low_anomaly_df = low_anomaly_df.append(
                            col1_df, ignore_index=False)  # , axis=1)
                low_anomaly_df_index = low_anomaly_df.index.tolist()
                additional_column_df = pd.DataFrame(
                    data=in_data, columns=['beat.hostname', '@timestamp'], index=low_anomaly_df_index)
                for new_col in additional_column_df.columns:
                    low_anomaly_df[str(new_col)
                                   ] = additional_column_df[str(new_col)]
                return low_anomaly_df, low_anomaly_counts
            except Exception as e:
                print('low_clustor_anomaly function failed!', e)
        # other_clustor_anomaly function can be used for Test of Infering

        def other_clustor_anomaly(other_clust_data, high_anomaly_df, low_anomaly_df, in_data):
            try:
                other_anomaly_counts = []
                other_clust_data = other_clust_data.drop(
                    columns=[str((other_clust_data.columns.values)[-1])])
                other_anomaly_df = pd.DataFrame()
                for col in other_clust_data.columns:
                    # to verify with higher clustered threholds
                    df_index = pd.DataFrame(data=other_clust_data.loc[:, str(col)] >
                                            (high_anomaly_df.loc[:, str(col)].min()))  # or (low_anomaly_df.loc[:, str(col)].min()) )  # ,lower_cluster.loc[:, str(col)])-
                    df_index = df_index.rename(columns={0: str(col)})
                    req_index = df_index.loc[df_index[str(
                        col)] == True].index.tolist()
                    other_anomaly_counts.append(
                        'Anomaies in Other Clusters '+str(col)+'::'+str(len(req_index)))
                    col1_df = pd.DataFrame(
                        data=in_data.loc[req_index, ['@timestamp', str(col)]])
                    other_anomaly_df = other_anomaly_df.append(
                        col1_df, ignore_index=False)  # , axis=1)
                    # to verify with lower clustered threholds
                    df_index = pd.DataFrame(data=other_clust_data.loc[:, str(col)] >
                                            (low_anomaly_df.loc[:, str(col)].min()))  # or (low_anomaly_df.loc[:, str(col)].min()) )  # ,lower_cluster.loc[:, str(col)])-
                    df_index = df_index.rename(columns={0: str(col)})
                    req_index = df_index.loc[df_index[str(
                        col)] == True].index.tolist()
                    other_anomaly_counts.append(
                        'Anomaies in Other Clusters '+str(col)+'::'+str(len(req_index)))
                    col1_df = pd.DataFrame(
                        data=in_data.loc[req_index, ['@timestamp', str(col)]])
                    other_anomaly_df = other_anomaly_df.append(
                        col1_df, ignore_index=False)  # , axis=1)
                low_anomaly_df_index = other_anomaly_df.index.tolist()
                additional_column_df = pd.DataFrame(
                    data=in_data, columns=['beat.hostname', '@timestamp'], index=low_anomaly_df_index)
                for new_col in additional_column_df.columns:
                    other_anomaly_df[str(
                        new_col)] = additional_column_df[str(new_col)]

                return other_anomaly_df, other_anomaly_counts
            except Exception as e:
                print('other_clustor_anomaly function failed!', e)

        # Cross verify in other clustoers for anomalies
        def get_other_clust_data(clustered_data):
            try:
                tot_clust = list(range(len(clustered_data)))
                other_clust_data = pd.DataFrame()
                for oth_clust in tot_clust:
                    if tot_clust[oth_clust] != 0 and tot_clust[oth_clust] != tot_clust[-1]:
                        other_data_df = pd.DataFrame(
                            data=clustered_data[oth_clust])
                        other_clust_data = other_clust_data.append(
                            other_data_df, ignore_index=False)
                return other_clust_data
            except Exception as e:
                print('get_other_clust_data function failed!', e)

        high_anomaly_df, high_anomaly_counts = high_clustor_anomaly(
            clustered_data, in_data)
        if len(clustered_data)>1:
            low_anomaly_df, low_anomaly_counts = low_clustor_anomaly(
                clustered_data, in_data)
        else:
            low_anomaly_df=pd.DataFrame()
            low_anomaly_counts=['No Low Clustered data available for Anomaly Detection']
        if len(clustered_data)>2:
            other_clust_data = get_other_clust_data(clustered_data)
            other_anomaly_df, other_anomaly_counts = other_clustor_anomaly(
                other_clust_data, high_anomaly_df, low_anomaly_df, in_data)
        else:
            other_anomaly_df=pd.DataFrame()
            other_anomaly_counts=['No Other Clustered data available for Anomaly Detection']
            
        """
        q1, q3 = np.percentile(
            low_anomaly_df, [25, 75])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(low_anomaly_df.loc[:,'system.filesystem.available'].values > q3,low_anomaly_df.loc[:,'system.fsstat.total_size.used'].values < q1)
                             
        ax.set_title('Anomaly ')
        ax.set_xlabel('Diesel1')
        ax.set_ylabel('Diesel1.1')
        plt.colorbar(scatter)
        """
        final_anomaly = final_anomaly.append(high_anomaly_df)
        final_anomaly = final_anomaly.append(low_anomaly_df)
        final_anomaly = final_anomaly.append(other_anomaly_df)
#        final_anomaly = pd.pivot_table(
#            final_anomaly, index=final_anomaly.index)  # '@timestamp')
        final_anomaies_count.append(high_anomaly_counts)
        final_anomaies_count.append(low_anomaly_counts)
        final_anomaies_count.append(other_anomaly_counts)
        return high_anomaly_df, low_anomaly_df, clustered_data, other_anomaly_df, final_anomaly, final_anomaies_count
    except Exception as e:
        print('construct_anomaly function failed', e)


def load_data():
    try:
        in_data = pd.DataFrame()
#        for file_name in glob.glob('C:/Koushik/Misc/Erricsion/May2019/batch_0326_allHistoricalDownsamp/'+'*.csv'):
#            fdata = pd.read_csv(str(file_name))
#            in_data = in_data.append(fdata)  # ,ignore_index=True)
        in_data  = pd.read_csv(
            'C:/Koushik/Misc/Erricsion/May2019/data_now-2880m.csv')
        return in_data
    except Exception as e:
        print('load_data  function failed!!', e)


def visualize_results(viz_col, plot_type, clustered_data, other_anomaly_df, final_anomaly, in_data):
    try:
        def visualize_all_anomaly(viz_col, final_anomaly, in_data):
            try:
                final_anomaly_df1 = pd.DataFrame(
                    data=final_anomaly.loc[:, [str(viz_col)]])
                final_anomaly_df1 = pd.DataFrame(
                    data=final_anomaly_df1.dropna())
                final_anomaly_df1_index = final_anomaly_df1.index.tolist()
                

                final_anomaly_df2 = pd.DataFrame(
                    data=in_data.loc[:, str(viz_col)], index=final_anomaly_df1_index)

                pad_index = list(
                    range(0, in_data.loc[:, str(viz_col)].shape[0]))
                reset_index = [
                    index for index in pad_index if index not in final_anomaly_df1_index]
                df_dummy = pd.DataFrame(np.nan, index=reset_index, columns=[str(viz_col)])
                final_anomaly_df2 = final_anomaly_df2.append(
                    df_dummy, ignore_index=False)
                final_anomaly_df2 = final_anomaly_df2.sort_index(ascending=[True])
                final_anomaly_df2_index = final_anomaly_df2.index.tolist()
                return final_anomaly_df2,final_anomaly_df2_index
            except Exception as e:
                print('visualize_all_anomaly function failed', e)

        def plot_figure_anomaly(plot_type, clustered_data, final_anomaly, in_data):
            try:
                
                high_anomaly_df, low_anomaly_df = clustered_data[-1], clustered_data[0]
                if plot_type == 'low':
                    plot_anomaly = low_anomaly_df
                elif plot_type == 'high':
                    plot_anomaly = high_anomaly_df
                elif plot_type == 'other':
                    plot_anomaly = other_anomaly_df
                elif plot_type == 'all':
                    plot_anomaly = final_anomaly
                clust_anomaly_viz,final_anomaly_df2_index = visualize_all_anomaly(
                    viz_col, plot_anomaly, in_data)
                
                ##Plot anomalies with line plot::
#                plt.plot(in_data.loc[:, '@timestamp'])
#                plt.plot(clust_anomaly_viz.loc[:, str(viz_col)],color='red')
                
                ##Plot anomalies with peaks::
                plt.plot(in_data.loc[final_anomaly_df2_index, '@timestamp'].tail(50),in_data.loc[:,str(viz_col)].tail(50))
                plt.scatter(in_data.loc[final_anomaly_df2_index, '@timestamp'].tail(50), clust_anomaly_viz.loc[:,str(viz_col)].tail(50), edgecolors='r') #,marker="^"

                plt.title('Anomalies', color='purple')
                plt.xlabel(str(viz_col) + ' timestamps', color='orange')
                plt.ylabel(str(viz_col) + ' values', color='orange')
                plt.show()
            except Exception as e:
                print('plot_figure_anomaly function failed', e)

        plot_figure_anomaly(plot_type, clustered_data, final_anomaly, in_data)

    except Exception as e:
        print('visualize_results function failed', e)


if __name__ == '__main__':

    try:
        in_data = load_data()
        in_data=pd.DataFrame(data= in_data,columns=['system.cpu.total.pct_preproc','beat.hostname', '@timestamp'])

        in_data_ignore = preprocess_data(in_data)

        high_anomaly_df, low_anomaly_df, clustered_data, other_anomaly_df, final_anomaly, final_anomaies_count = construct_anomaly(
            in_data, in_data_ignore)
        viz_col = 'system.cpu.total.pct_preproc'

        plot_type = 'all'
        visualize_results(viz_col, plot_type, clustered_data, other_anomaly_df,
                          final_anomaly, in_data)   
        print('Total records processed are :', in_data.shape[0])


    except Exception as e:
        print('__main__ function failed', e)
