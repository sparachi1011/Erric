# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:05:50 2019

@author: SaiKoushik
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import re
import datetime

os.chdir('D:/Koushik/Erricsion/jun/logdata')

def load_data():
    
    try:
         important = []
         for file_name in glob.glob('D:/Koushik/Erricsion/jun/logdata/'+'*.log'):
             with open(str(file_name), "r") as in_file:
                 # Loop over each log line
                 for line in in_file:
                     important.append(line)
         log_df=pd.DataFrame(important)
         return log_df,important
    except Exception as e:
        print('load_data  function failed!!', e)

def create_log_df():
    try:
        in_data,important = load_data()
        Log_df=pd.DataFrame()
        df_cols=['TimeStamp','LogType','Logger','ErrCode','LoggedBy','ConfigTimeOut','ResponseTime','processName','DS','Class','Method','Message_SQLQuery','Scrap_data']
        for test_line in in_data.iloc[:,0]:
        #test_line="2019-05-09 00:00:01,488 INFO  [MonitorLogger_MBI1] [80063381] SQL-->[ConfigTimeOut: 10000][ResponseTime: 15][processName: MBI1][DS: ServiceRequestDS][Class: com.mobily.mbi.dao.serviceorder.ServiceOrderDAO][Method: hasPendingRequest][SQLQuery: SELECT sr_id FROM sr_servicerequest_tbl WHERE line_number = '966549226005' and func_id IN (111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 301, 300) and sr_date > (SYSDATE - (?/24)) and status in (1,2,4)]"
        #test_line='2019-05-09 00:00:00,020 INFO  [MonitorLogger_MBI1] [] Connection-->[ResponseTime: 1][processName: MBI1][DS: BMSPORTALDS][Class: com.mobily.mbi.dao.serviceorder.BMSServiceOrderDAO][Method: check1100SMSMODIFYED]'
        #test_line='?????lcascap'
        
            def verify_test_line(check_valid):
                try:
                    check_datetime=datetime.datetime.strptime(check_valid, '%Y-%m-%d')
                    return check_datetime
                except:
                    pass
            check_datetime=  verify_test_line(test_line[0:10]) 
            
            if isinstance(check_datetime, datetime.date)==True:
                df_list=[]
                test_line=test_line.replace("'",'"')
                df_list.append(test_line[0:23].replace(',',':'))
                test_line=re.sub(test_line[0:23],'',test_line)
                df_list.append(test_line[0:6].replace(' ',''))
                test_line=re.sub(test_line[0:6],'',test_line)
                if df_list[-1]=='INFO':
                    pass
                else:
                    df_list[-1]=''
                df_list.append(test_line[test_line.find('[')+1:test_line.find(']')].replace('[','').replace(']',''))
            #    test_line=re.sub(test_line[1:(len(df_list[-1]))],'',test_line)
                mix_val=test_line[test_line.find(r'->['):test_line.find(r'].')].replace('->','')
                res=test_line.replace(mix_val,'')
                res=res[1:]
                res_list=res.split(' ')
                df_list.append(str(res_list[1]).replace('[','').replace(']',''))
                df_list.append(str(res_list[2]).replace('[','').replace(']','')) 
                mix_list=(mix_val.split(']['))
                
                ConfigTimeOut_list,ResponseTime_list,processName_list,DS_list,Class_list,Method_list,Message_SQLQuery_list='','','','','','',''
                for val in list(range(len(mix_list))):
                    mix_list[val]=mix_list[val].replace('[','').replace(']','')
                    inner_list=(mix_list[val].split(': '))
                    if str(inner_list[0])=='ConfigTimeOut':
                        ConfigTimeOut_list=ConfigTimeOut_list+mix_list[val][mix_list[val].find(':')+1:].replace(' ','')
                    else:
                        ConfigTimeOut_list+str('')
                    if str(inner_list[0])=='ResponseTime':
                        ResponseTime_list=ResponseTime_list+mix_list[val][mix_list[val].find(':')+1:].replace(' ','')
                    else:
                        ResponseTime_list+str('')
                    if str(inner_list[0])=='processName':
                        processName_list=processName_list+mix_list[val][mix_list[val].find(':')+1:].replace(' ','')
                    else:
                        processName_list+str('')
                    if str(inner_list[0])=='DS':
                        DS_list=DS_list+mix_list[val][mix_list[val].find(':')+1:].replace(' ','')
                    else:
                        DS_list+str('')
                    if str(inner_list[0])=='Class':
                        Class_list=Class_list+mix_list[val][mix_list[val].find(':')+1:].replace(' ','')
                    else:
                        Class_list+str('')
                    if str(inner_list[0])=='Method':
                        Method_list=Method_list+mix_list[val][mix_list[val].find(':')+1:].replace(' ','')
                    else:
                        Method_list+str('')
                    if str(inner_list[0])=='SQLQuery':
                        Message_SQLQuery_list=Message_SQLQuery_list+mix_list[val][mix_list[val].find(':')+1:].replace(' ','')
                    else:
                        Message_SQLQuery_list+str('')
                    if str(inner_list[0])=='Message':
                        Message_SQLQuery_list=Message_SQLQuery_list+mix_list[val][mix_list[val].find(':')+1:].replace(' ','')
                    else:
                        Message_SQLQuery_list+str('')
                
                df_list1=[ConfigTimeOut_list,ResponseTime_list,processName_list,DS_list,Class_list,Method_list,Message_SQLQuery_list]
                
                df_list.extend(df_list1)
                scrap_data=' '
                df_list.extend(scrap_data)               
                
            else:
                df_list,scrap_data=[],[]
                for col in df_cols[:-1]:
                    df_list.append('')
                scrap_data.append(test_line)
                df_list.extend(scrap_data)   
            arr=np.array(df_list)
            arr=arr.reshape(1,-1)
            df=pd.DataFrame(arr,columns=df_cols)
            Log_df=Log_df.append(df,ignore_index=False)
        return in_data,important,Log_df
    except Exception as e:
        print('create_log_df  function failed!!', e)


if __name__ == '__main__':
    try:
        
        in_data,important,Log_df=create_log_df()
        
        print('Total records processed are :', in_data.shape[0])
        

    
    except Exception as e:
        print('_main_ function failed', e)
