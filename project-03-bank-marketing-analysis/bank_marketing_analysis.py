import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df = pd.read_csv('bank-additional-full.csv',sep=';')
#full data analysing

print(f'Shape of the dataset is: {df.shape}')
print(f'The datatype of the columns:{df.info()}')
print(f'basic stat of numerical columns: {df.describe()}')
print(f'The number of rows having null values before cleaning: {df.isnull().sum()}')
print(f'The number of duplicate rows before cleaning:{df.duplicated().sum()}')



# Data cleaning part
df = df.drop_duplicates()
df = df.rename(columns={'y': 'subscribed','loan':'personal'})
cat_columns = ['job','marital','education','default','housing','personal','contact','month','day_of_week','poutcome']

for col in cat_columns:
   df[col] = df[col].replace('unknown', df[col].mode()[0])

overall = df.groupby('subscribed')['age'].agg(count='count')
overall['rate'] = (overall['count']/len(df)*100).round(2)
print(overall)

# when we look into the overall rate we can see only 11.27% of customers had taken the subscription, its normal in the financial domain

# which age group took more subscription
df['age_group'] = pd.cut(df['age'],
                          bins=[0,25,35,45,60,100],
                         labels=['below25','26-35','36-45','46-60','60+'])
age_rate = df.groupby(['age_group','subscribed'])['age'].agg(count='count').reset_index()
age_rate['total'] = age_rate.groupby('age_group')['count'].transform('sum')
age_rate['rate'] = (age_rate['count']/age_rate['total']*100).round(2)

age_rate_yes = age_rate[age_rate['subscribed']=='yes']
print(age_rate_yes)

job_rate = df.groupby(['job','subscribed'])['age'].agg(count='count').reset_index()
job_rate['total'] = job_rate.groupby('job')['count'].transform('sum')
job_rate['rate'] = (job_rate['count']/job_rate['total']*100).round(2)
job_rate_yes = job_rate[job_rate['subscribed'] == 'yes']
print(job_rate_yes.sort_values('rate',ascending=False))

# the retired people take more subscription this backup the previous insight

marital_rate = df.groupby(['marital','subscribed'])['age'].agg(count='count').reset_index()
marital_rate['total']= marital_rate.groupby('marital')['count'].transform('sum')
marital_rate['rate'] = (marital_rate['count']/marital_rate['total']*100).round(2)
marital_rate_yes = marital_rate[marital_rate['subscribed'] == 'yes']
print(marital_rate_yes)

education_rate = df.groupby(['education','subscribed'])['age'].agg(count='count').reset_index()
education_rate['total'] = education_rate.groupby('education')['count'].transform('sum')
education_rate['rate'] = (education_rate['count']/education_rate['total']*100).round(2)
# removing the illetrate groups because of the small sample size.
education_rate = education_rate[education_rate['total']>=100]
education_rate_yes = education_rate[education_rate['subscribed'] == 'yes']
print(education_rate_yes.sort_values('rate',ascending=False))




print(df['default'].value_counts())
print(df[df['default'] == 'yes'])

# so the people who have default credit doesnt take the subscription based on this data 3/3 people doesnt take the subscription against 41173 have not default credit
# but we cannot test the stastical significance of this because of the small sample size.
 
 # let me know that if we can do a hypothesis for this and ist necessary

house_rate = df.groupby(['housing','subscribed'])['age'].agg(count='count').reset_index()
house_rate['total'] = house_rate.groupby('housing')['count'].transform('sum')
house_rate['rate'] = (house_rate['count']/house_rate['total']*100).round(2)
house_rate_yes = house_rate[house_rate['subscribed'] == 'yes']
print(house_rate_yes)

personal_rate = df.groupby(['personal','subscribed'])['age'].agg(count='count').reset_index()
personal_rate['total'] = personal_rate.groupby('personal')['count'].transform('sum')
personal_rate['rate'] = (personal_rate['count']/personal_rate['total']*100).round(2)
personal_rate_yes = personal_rate[personal_rate['subscribed'] == 'yes']
print(personal_rate_yes)

df['has_any_loan'] = 'no'
mask = (df['housing'] == 'yes')|(df['personal'] == 'yes')
df.loc[mask,'has_any_loan'] = 'yes'

loan_rate = df.groupby(['has_any_loan','subscribed'])['age'].agg(count='count').reset_index()
loan_rate['total'] = loan_rate.groupby('has_any_loan')['count'].transform('sum')
loan_rate['rate'] = (loan_rate['count']/loan_rate['total']*100).round(2)
loan_rate_yes = loan_rate[loan_rate['subscribed'] == 'yes']
print(loan_rate_yes)

print(df['duration'].describe())

df['duration_gap'] = pd.cut(df['duration'],bins=[0,30,60,120,180,300,420,5000],
                            labels=['below30s','30s-1min','1min-2min','2min-3min','3min-5min','5min-7min','above7min'])
# chose the 7min because its the 3rd quartile
duration_rate = df.groupby(['duration_gap','subscribed'])['age'].agg(count='count').reset_index()
duration_rate['total'] = duration_rate.groupby('duration_gap')['count'].transform('sum')
duration_rate['rate'] = (duration_rate['count']/duration_rate['total']*100).round(2)
duration_rate_yes = duration_rate[duration_rate['subscribed'] == 'yes']
print(duration_rate_yes)



month_rate = df.groupby(['month', 'subscribed'])['age'].agg(count='count').reset_index()
month_rate['total'] = month_rate.groupby('month')['count'].transform('sum')
month_rate['rate'] = (month_rate['count'] / month_rate['total'] * 100).round(2)
print(month_rate[month_rate['subscribed'] == 'yes'].sort_values('rate', ascending=False))

campaign_rate = df.groupby(['campaign', 'subscribed'])['age'].agg(count='count').reset_index()
campaign_rate['total'] = campaign_rate.groupby('campaign')['count'].transform('sum')
campaign_rate['rate'] = (campaign_rate['count']/campaign_rate['total']*100).round(2)
print(campaign_rate[campaign_rate['subscribed'] == 'yes'].sort_values('rate', ascending=False))

poutcome_rate = df.groupby(['poutcome', 'subscribed'])['age'].agg(count='count').reset_index()
poutcome_rate['total'] = poutcome_rate.groupby('poutcome')['count'].transform('sum')
poutcome_rate['rate'] = (poutcome_rate['count']/poutcome_rate['total']*100).round(2)
print(poutcome_rate[poutcome_rate['subscribed']=='yes'].sort_values('rate',ascending=False))

# visualization

num_cols = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.conf.idx','euribor3m','nr.employed']
print(df[num_cols].corr())
fig,ax = plt.subplots(figsize=(10,6))
sns.heatmap(df[num_cols].corr(),center=0.5,fmt='.2f',annot=True,ax=ax)
ax.set_title('correlation heatmap')
plt.show()


fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(overall,x='subscribed',y='rate')
ax.set_title('The overall subscription rate is 11.27%')
ax.set_xlabel('Subscribed')
ax.set_ylabel('count')
ax.bar_label(ax.containers[0])
plt.savefig('overall-subscription-rate.png')
#plt.show()

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(age_rate_yes,x='age_group',y='rate',ax=ax)
ax.set_title('subscription rate is high on people aged 60+(45.5%)')
ax.set_ylabel('percentage')
ax.set_xlabel('age group')
ax.bar_label(ax.containers[0])
plt.savefig('subscription based on age.png')
#plt.show()

contigency_table = pd.crosstab(df['age_group'],
                               df['subscribed'])
chi2, p_value , dof, excepted = stats.chi2_contingency(contigency_table)
print(f'p-value: {p_value}')
print('age gap is statistically significant' if p_value < 0.05 else 'age gap is not significant')


fig,ax = plt.subplots(figsize=(10,6))
sns.barplot(education_rate_yes,x='education',y='rate',ax=ax)
ax.set_title('subscription rate is high on retired customers(25.2%) and students(31.4%)')
ax.set_ylabel('percentage')
ax.set_xlabel('education')
ax.bar_label(ax.containers[0])
plt.savefig('subscription based on education.png')
#plt.show()

# iam not sure to report this because  in this data it seemed like illiterate customers are having high subsciption rate , but their sample size is very low
# need help to report this

fig,ax = plt.subplots(figsize=(10,6))
sns.barplot(job_rate_yes,x='job',y='rate',ax=ax)
ax.set_title('subscription rate is high on retired customers(25.2%) and students(31.4%)')
ax.set_ylabel('percentage')
ax.set_xlabel('job')
ax.bar_label(ax.containers[0])
plt.savefig('subscription based on job.png')
#plt.show()

fig,ax = plt.subplots(figsize=(10,6))
sns.barplot(marital_rate_yes,x='marital',y='rate',ax=ax)
ax.set_title('subscription rate is 4% high on single (14%) compared to married and divorced')
ax.set_ylabel('percentage')
ax.set_xlabel('marital')
ax.bar_label(ax.containers[0])
plt.savefig('subscription based on marital status.png')
#plt.show()
contigency_table = pd.crosstab(df['marital'],
                               df['subscribed'])
chi2, p_value , dof, excepted = stats.chi2_contingency(contigency_table)
print(f'p-value: {p_value}')
print('marital status is statistically significant' if p_value < 0.05 else 'marital status is not significant')





print("""
RECOMMENDATIONS
===============

DEMOGRAPHIC TARGETING
1. Prioritise customers aged 60+ — subscription rate 45.5% vs 11.3% overall
2. Retired and student segments — 25.2% and 31.4% subscription rates
3. Customers having a university degree have a slightly high subscription rate of 13.8%
   Action: Create separate campaign scripts for these segments

CAMPAIGN EXECUTION
3. Call duration is the strongest predictor of subscription
   Customers who talk 7+ minutes subscribe at 36.84%
   Action: Train agents on conversation extension techniques
   Note: Duration cannot be used to pre-select customers —
   it must be improved through agent training

4. Limit calls to maximum 3 per customer
   Subscription rate drops significantly after 3 contacts
   Action: Remove customers with 3+ previous contacts from active calling list

5. Focus on March, September, October, December
   These months show highest subscription rates historically(43%-50%)
   Action: Concentrate campaign budget in these months

6. Re-contact previous subscribers first
   Customers with successful previous outcome subscribe at significantly
   higher rates than new contacts
   Action: Build a 'warm list' from previous campaign successes

FINANCIAL PROFILE
7. Customers without any loans have marginally lower subscription rate
   Housing loan customers subscribe at similar rates to non-loan customers
   Action: Loan status alone should not disqualify a prospect
""")