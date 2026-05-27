# ================================================================
# IBM HR ATTRITION ANALYSIS 
# Dataset: 1470 employees, 35 columns
# Objective: Understand what drives employee attrition
# ================================================================

print('='*60)
print('SECTION 1 — DATASET OVERVIEW')
print('='*60)
# overview code
import pandas as pd

df = pd.read_csv('D:\python data analysis\WA_Fn-UseC_-HR-Employee-Attrition.csv')

print(df.shape)
print(df.columns.to_list())
print(df.dtypes)
print(df.head())


print(df.isnull().sum())

print(df.describe())


categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    print(f'{col}: {df[col].unique()}')

print(df['Attrition'].value_counts())
print(df['Attrition'].value_counts(normalize=True).round(2))
print(df['MonthlyIncome'].describe())

"""
1470 employees,35 columns, no null values at all, 16% of employees left the company, departments are [sales,'research and development','hr'],avg monthly income is 6503
"""




print('\n' + '='*60)
print('SECTION 2 — ATTRITION OVERVIEW')
print('='*60)
# attrition analysis
attrition_summary = df.groupby('Attrition').agg(
    count=('EmployeeNumber', 'count')
).reset_index()
attrition_summary['percentage'] = (attrition_summary['count']/len(df) * 100).round(2)
print(attrition_summary)



print('\n' + '='*60)
print('SECTION 3 — DEPARTMENT ANALYSIS')
print('='*60)
# department breakdown

# Salary by department
print(df.groupby('Department')['MonthlyIncome'].mean().round(0))

# Salary by attrition status
print(df.groupby('Attrition')['MonthlyIncome'].mean().round(0))


dept_attrition_summary = df.groupby(['Department', 'Attrition']).agg(
    emp_count=('EmployeeNumber','count')
).reset_index()

#dept_attrition_summary['percentage'] = dept_attrition_summary['emp_count']/len((df[df['Department'] == dept_attrition_summary['Department']]).reset_index())
dept_attrition_summary['total'] = dept_attrition_summary.groupby('Department')['emp_count'].transform('sum')
dept_attrition_summary['percentage'] = (dept_attrition_summary['emp_count']/dept_attrition_summary['total'] * 100).round(1)
dept_attrition_summary = dept_attrition_summary.drop(columns=['total'])
print(dept_attrition_summary)

#sales dept loses most employees. around 20.6% and also HR dept loses 19% employees

print('\n' + '='*60)
print('SECTION 4 — SALARY ANALYSIS')
print('='*60)
# salary analysis
salary_summ = df.groupby('Department')['MonthlyIncome'].agg(
    avg_income='mean'
).reset_index()


salary_dept_summ = pd.merge(dept_attrition_summary,salary_summ,on='Department',how='inner')
print(salary_dept_summ)

salary_job_summ = df.groupby('JobRole')['MonthlyIncome'].agg(
    avg_salary='mean'
).reset_index()


attrition_by_job = df.groupby(['JobRole','Attrition']).agg(
    count=('EmployeeNumber', 'count')
).reset_index()
attrition_by_job['total'] = attrition_by_job.groupby('JobRole')['count'].transform('sum')
attrition_by_job['percentage'] = (attrition_by_job['count']/attrition_by_job['total'] * 100).round(1)
attrition_by_job.drop(columns='total',inplace=True)
df2 = pd.merge(attrition_by_job,salary_job_summ,on='JobRole', how="inner")
print(df2)

# salary is a factor for attrition. the job that pays below the company average makes employees to quit

print('\n' + '='*60)
print('SECTION 5 — KEY DRIVERS OF ATTRITION')
print('='*60)
# overtime, satisfaction, distance, age

df['age_group'] = pd.cut(df['Age'],
                         bins=[0,25,35,45,100],
                         labels=['Under25','25-35','35-45','above45'])

age_attrition = df.groupby(['age_group','Attrition']).agg(
    count=('EmployeeNumber','count'),
)
age_attrition['total'] = age_attrition.groupby('age_group')['count'].transform('sum')
age_attrition['percentage'] = (age_attrition['count']/age_attrition['total']*100).round(1)
age_attrition.drop(columns=['total'])
print(age_attrition)


df['exp_group'] = pd.cut(
    df['TotalWorkingYears'],
    bins=[0,5,10,15,40],
    labels=['below 5','5-10','10-15','above15']
)

exp_attrition = df.groupby(['exp_group','Attrition']).agg(
    count=('EmployeeNumber','count'),
    #avg_yrs_at_cmpn=('YearsAtCompany','mean')
)

exp_attrition['total'] = exp_attrition.groupby('exp_group')['count'].transform('sum')
exp_attrition['avg_years_at_cmpn'] = df.groupby(['exp_group','Attrition'])['YearsAtCompany'].agg('mean').round(1)
exp_attrition['percentage'] = (exp_attrition['count']/exp_attrition['total']*100).round(1)
exp_attrition.drop(columns=['total'],inplace=True)
print(exp_attrition)


overtime_attr = df.groupby(['OverTime','Attrition']).agg(
    count=('EmployeeNumber','count')
)
overtime_attr['total'] = overtime_attr.groupby('OverTime').transform('sum')
overtime_attr['percentage'] = (overtime_attr['count']/overtime_attr['total']*100).round(1)
overtime_attr.drop(columns=['total'],inplace=True)
print(overtime_attr)


job_stat_attr = df.groupby('Attrition')['JobSatisfaction'].agg(
    avg_job_stsfctn='mean'
).reset_index()
print(job_stat_attr)

stat_attr_per_level =df.groupby(['JobSatisfaction','Attrition']).agg(
    count=('EmployeeNumber','count')
    
)
stat_attr_per_level['total'] = stat_attr_per_level.groupby('JobSatisfaction')['count'].transform('sum')
stat_attr_per_level['percentage'] = (stat_attr_per_level['count']/stat_attr_per_level['total']*100).round(1)
stat_attr_per_level.drop(columns=['total'],inplace=True)
print(stat_attr_per_level)


df['dstnc_group'] = pd.cut(
    df['DistanceFromHome'],
    bins=[0,10,20,100],
    labels=['Near','Mid','Far']
)

dstnc_attr = df.groupby(['dstnc_group','Attrition']).agg(
    count=('EmployeeNumber','count')
)
dstnc_attr['total'] = dstnc_attr.groupby('dstnc_group')['count'].transform('sum')
dstnc_attr['percentage'] = (dstnc_attr['count']/dstnc_attr['total']*100).round(1)
dstnc_attr.drop(columns=['total'],inplace=True)
print(dstnc_attr)


salary_per_jobrole = df.groupby('JobRole').agg(
    min_sal = ('MonthlyIncome','min'),
    max_sal=('MonthlyIncome','max'),
    avg_sal=('MonthlyIncome','mean'),
    median_sal=('MonthlyIncome','median')
).reset_index()
print(salary_per_jobrole)


gender_attr = df.groupby(['Gender','Attrition']).agg(
    count=('EmployeeNumber','count')
)
gender_attr['total'] = gender_attr.groupby('Gender')['count'].transform('sum')
gender_attr['percentage'] = (gender_attr['count']/gender_attr['total']).round(2)
gender_attr.drop(columns=['total'],inplace=True)
print(gender_attr)


print('\n' + '='*60)
print('SECTION 6 — RECOMMENDATIONS')
print('='*60)

print("""
KEY FINDINGS AND RECOMMENDATIONS
=================================

1. OVERTIME IS THE STRONGEST ATTRITION DRIVER
   30.5% of overtime employees left vs 10.4% of non-overtime employees.
   Recommendation: Audit departments with highest overtime — consider hiring
   or workload redistribution before more experienced staff leave.

2. SALES DEPARTMENT NEEDS URGENT ATTENTION
   Attrition rate of 20.6% — highest across all departments.
   Average salary in Sales is below R&D. Salary review recommended.

3. EARLY TENURE EMPLOYEES ARE HIGH RISK
   28.2% attrition among employees with under 5 years experience.
   Structured mentoring and career path clarity in first 2 years
   would likely reduce this significantly.

4. YOUNG EMPLOYEES LEAVE AT HIGHEST RATE
   35.8% attrition in Under 25 age group vs 12.5% in 35-45 group.
   Graduate onboarding programs and clear promotion timelines
   would improve retention in this segment.

5. SALARY IS A FACTOR BUT NOT THE ONLY FACTOR
   Leavers earn avg 4787 vs stayers at 6833 monthly.
   However job satisfaction difference is small (2.47 vs 2.73),
   suggesting non-monetary factors like overtime and career growth
   matter equally.
""")
