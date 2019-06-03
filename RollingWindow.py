# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:07:11 2019

@author: sparachi
"""
import datetime
from dateutil import rrule
from datetime import date,timedelta
 

Start_Datetime = datetime.datetime.strptime('Jun 11 2005  1:33PM', '%b %d %Y %I:%M%p')
Start_Year=Start_Datetime.year
Start_Month=Start_Datetime.month
Start_Day=Start_Datetime.day

End_Datetime = datetime.datetime.strptime('Aug 12 2005  1:33PM', '%b %d %Y %I:%M%p')

End_Year=End_Datetime.year
End_Month=End_Datetime.month
End_Day=End_Datetime.day


#print(list(rrule.rrule(rrule.MONTHLY, dtstart=date(2013, 11, 1), until=date(2014, 2, 1))))

months_list=list(rrule.rrule(rrule.WEEKLY, dtstart=date(Start_Year, Start_Month, Start_Day), until=date(End_Year, End_Month, End_Day)))

#Start_Datetime_1=datetime.strftime(months_list[0],'%Y-%m-%dT%H:%M:%S')

type(End_Datetime)

NextWeek_Date = months_list[0] + datetime.timedelta(weeks=1)


Min_Week = 4
Max_Week = 5
Month_value=[]

for i in list(range(len(months_list))):
#    print(months_list[i])
    Month_value.append(str(months_list[i].month))
