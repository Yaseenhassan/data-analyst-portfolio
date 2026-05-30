
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

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
#1470 employees,35 columns, no null values at all, 16% of employees left the company, departments are [sales,'research and development','hr'],avg monthly income is 6503
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


# overall attrition 
fig, ax = plt.subplots(figsize=(10,6))
attr = df.groupby('Attrition').size().reset_index(name='count')

attr['total'] = attr['count'].sum()

attr['rate'] = (attr['count']/attr['total']*100).round(1)
ax.bar(attr['Attrition'], attr['count'], color='#E74C3C', width=0.5)
ax.set_title('Overall attrition')
ax.set_xlabel('Attrition')
ax.set_ylabel('Count')

for i, v in enumerate(attr['rate']):
    ax.text(i, v+1, f'{str(v)}%', ha='center', fontsize=12)
for i, v in enumerate(attr['count']):
    ax.text(i, v+1, f'{str(v)}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('01_attrition_overview.png',dpi=150)
plt.show()

dept_attr = df.groupby(['Department', 'Attrition']).size().reset_index(name='count')
dept_attr['total'] = dept_attr.groupby('Department')['count'].transform('sum')
dept_attr['rate'] = (dept_attr['count']/dept_attr['total']*100).round(1)
dept_attr_yes = dept_attr[dept_attr['Attrition'] == 'Yes']
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(data=dept_attr_yes, x='Department', y='rate', palette='Reds_d', ax=ax)
ax.set_title('Attrition Rate by Department (%)', fontsize=16, fontweight='bold')
ax.set_xlabel('Department', fontsize=12)
ax.set_ylabel('Attrition Rate (%)', fontsize=12)

for i,v in enumerate(dept_attr_yes['rate']):
    ax.text(i, v+0.25, f'{str(v)}%', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('02_department_analysis.png', dpi=150)
plt.show()



fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome',
            palette={'Yes': '#E74C3C', 'No': '#2ECC71'}, ax=ax)
ax.set_title('Monthly Income Distribution — Leavers vs Stayers',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Attrition', fontsize=12)
ax.set_ylabel('Monthly Income (₹)', fontsize=12)
plt.tight_layout()
plt.savefig('03_salary_analysis.png', dpi=150)
plt.show()




fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('IBM HR Attrition Dashboard', fontsize=20, fontweight='bold')

df['distance_group'] = pd.cut(
    x=df['DistanceFromHome'],
    bins=[0,10,20,100],
    labels=['Near', 'Mid', 'Far']
)

dstnc_attr = df.groupby(['distance_group', 'Attrition']).size().reset_index(name='count')
dstnc_attr['total'] = dstnc_attr.groupby('distance_group')['count'].transform('sum')
dstnc_attr['rate'] = (dstnc_attr['count']/dstnc_attr['total']*100).round(1)
dstnc_attr_yes = dstnc_attr[dstnc_attr['Attrition'] == 'Yes']

sns.barplot(data=dstnc_attr_yes, x='distance_group', y='rate', ax=axes[0,0], palette='Reds_d')
axes[0,0].set_title('Attrition by Distance From Home')
axes[0,0].set_ylabel('rate')

# Chart 2 — top right
sns.boxplot(data=df, x='Attrition', y='TotalWorkingYears',
            palette={'Yes': '#E74C3C', 'No': '#2ECC71'}, ax=axes[0, 1])


# Chart 3 — bottom left
overtime_attr = df.groupby(['OverTime', 'Attrition']).size().unstack()
overtime_attr.plot(kind='bar', ax=axes[1, 0],
                   color=['#2ECC71', '#E74C3C'], rot=0)
axes[1, 0].set_title('Overtime vs Attrition')
axes[1, 0].set_xlabel('Works Overtime')

# Chart 4 — bottom right
age_attr = df[df['Attrition'] == 'Yes']['Age']
axes[1, 1].hist(age_attr, bins=20, color='#E74C3C', alpha=0.8)
axes[1, 1].set_title('Age of Employees Who Left')
axes[1, 1].set_xlabel('Age')

plt.tight_layout()
plt.savefig('04_key_drivers.png', dpi=150, bbox_inches='tight')
plt.show()


df['age_group'] = pd.cut(
    df['Age'],
    bins=[18,25,35,45,100],
    labels=['18-25', '26-35', '36-45', '46+']
)

sal_by_age = df.groupby(['age_group','Attrition'])['MonthlyIncome'].mean().reset_index(name='average_income')
sal_stayed = sal_by_age.loc[sal_by_age['Attrition'] == 'No', 'average_income'].to_list()
sal_left = sal_by_age.loc[sal_by_age['Attrition'] == 'Yes', 'average_income'].to_list()
age_groups = sal_by_age['age_group'].unique().to_list()
overtime_by_age = df.groupby(['age_group','OverTime']).size().reset_index(name='count')
overtime_by_age['total'] = overtime_by_age.groupby('age_group')['count'].transform('sum')
overtime_by_age['rate'] = (overtime_by_age['count']/overtime_by_age['total'] * 100).round(1)
overtime_by_age_yes = overtime_by_age.loc[overtime_by_age['OverTime'] == 'Yes','rate'].to_list()


x = np.arange(len(age_groups))
width = 0.35  

fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary Axis: Grouped Bars for Salary
bar1 = ax1.bar(x - width/2, sal_stayed, width, label='Salary: Stayed', color='#2b5c8f')
bar2 = ax1.bar(x + width/2, sal_left, width, label='Salary: Left', color='#7aa6c2')

ax1.set_xlabel('Age Groups', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Average Monthly Salary ($)', fontsize=12, color='#2b5c8f', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(age_groups)
ax1.tick_params(axis='y', labelcolor='#2b5c8f')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

ax2 = ax1.twinx()  
line = ax2.plot(x, overtime_by_age_yes, color='#d9534f', marker='o', linewidth=2.5, label='Overtime Rate (%)')

ax2.set_ylabel('Overtime Rate (%)', fontsize=12, color='#d9534f', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#d9534f')
ax2.set_ylim(0, 100)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.title('05_recommendations_summary.png', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

plt.show()



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
