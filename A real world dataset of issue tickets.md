There is a company providing IT services. IT requests are reported to the company by a reporter. We call these request issues. The issue is than assigned to an IT specialist on the company indicated in the worker column. Issue_id is the id field which must be removed before your analysis. Jiraname field can also be considered as an id field. The important fields that can be used for supervised models are work_log (numeric prediction - regression), issue_category (classification), worker (classification), employee_type (classification). 

Work_log is the total time spent to close an issue. It may be important to predict that time. 
It is also important to assign correct worker to an issue therefore worker classification system can be developed. You may reduce the number of classes in this case by choosing the most frequent N workers and combining other workers to other class. N can be 100 or 200. 
You can follow a similar approach for  issue_category classification.

If you choose work_log as the class attribute then you must delete work_log_total and work_log_ratio since they are derived from work_log.
Work_log_total is larger than work_log  if employees other than assigned worker also spent time on the issue. 
If you choose issue_category as the class attribute then you must delete issue_sub_category. 

We will work with supervised models and continue with unsupervised ones to have a better understanding of the data.
