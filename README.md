# Lending_Club_Loan_Interest_Prediction

This Repository has Notebook which shows loan prediction on Lending club loan dataset[1].

We have done Interest rate prediction using,
1. Linear Regression
2. Decision Trees.

Orignal Dataset has 10000 rows and 55 features.
# Data Preprocessing
But before proceeding to prediction, since data had issues, we needed to clean the data. Issues are,
1. Missing Data : 

Following are the columns and corresponding data count
num_accounts_120d_past_due 9682 (4% missing data)
months_since_last_credit_inquiry 8729 ( 12 % missing data)
months_since_90d_late	2285  (77% missing data)
months_since_last_delinq 4342 (66% missing data)
verification_income_joint 1455  (85% missing data)
annual_income_joint 1495 (85% missing data)
debt_to_income_joint 1495 (85% missing data)
debt_to_income	9976 (0.02% missing data)
emp_length 9183 (0.2% missing data)
emp_title 9167  (0.2% missing data)

We decide to drop features which have more than 50% missing data hence we are left with months_since_last_credit_inquiry, emp_title, emp_lenght, debt_to_income

Since emp_title prediction is difficult, as a prelimnary processing step, we delete rows with missing emp_title giving total of 9167 records.
This reduces the missing data in other features as well. Following is feature and missing count
emp_length  0
debt_to_income 1
months_since_last_credit_inquiry 1122
num_accounts_120d_past_due 295

We also observered similarity (line with slope approx. 1) between debt to income and debt to income joint and hence decided to replace missing value.
#future scope: predict debt to income rather than directly replacing same value.

months_since_last_credit_inquiry, num_accounts_120d_past_due values are numerical values hence replaced by mean values as prelimnary step.
#future scope : find if any other values are correlated with them which could help predict missing values.

We observed that few emp_title are frequent and others occur very rarely. We plotted distribution and decided to cap emp_titles above count of 
to be designated uniquely and others to be clubbed under 'others'
This helped to reduce the lables and also we focus on majority of data than invovle lowering prediction errror of those values which occur rarely 

2. Encoding
There are some categorical features which have large number of lables hence we used label encoding:
emp_title
state
loan_purpose
grade,
sub_grade
Other categorical features with fewer labels, we used one hot encoding:
homeownership
verified_income
application_type
loan_status
initial_listing_status
disbursement_method
issue_months

3. Feature selection based on r2 square of feature and interest rate
For r-square of non categorical featues showing variance from interest rate, we selected those which have r2 between +-0.4 and +-1
state  -0.5824840787793504
total_credit_lines -0.9744107810420859
open_credit_lines -0.7609641860012972
total_credit_limit -0.9814945506790422
total_credit_utilized  -0.936937063603636
num_satisfactory_accounts -0.7656690402816657
num_total_cc_accounts  -0.499472471902201
sub_grade 0.6541968005778498
interest_rate  1.0

We selected all categorical ones.
#Future scope: We would work in direction of feature selection of categorical features

4. Date Format
Converted the date format
5. Scaling
  We did scaling for selected features. We used Min max scaler


# Visualization and Observations:
From our visualizations, we observed few things
OBSERVATIONS
open credit lines and number of satisfactory accounts are same and one of them can be removed from features
for lower annual incomes, credit limits and credit utilized interest rates are high
Dataset is imbalanced for state. Very high data is for states like NJ, NY, TX, FL and very loweer for SD, DC, NE and others. For predictions of lower count of states may be affected.

# Prediction Model:
We used linear regression and we obtained:
Mean Absolute Error: 1.1150561874542772e-13
Mean Squared Error: 1.2460585339007213e-26
Root Mean Squared Error: 1.1162699198225854e-13

For decision Trees, we observed for varying depth, we observerd lowered error
We ploted this and saw that from depth=17 ownwards, error was somewhat same. Hence we selected depth = 17


Future Work:
1. Tuning of paramters for both prediction models
2. We would work in direction of feature selection of categorical features
3. predict debt to income rather than directly replacing same value.
4. Visualize data more and try finding more observations which could help for building prediction model and selection of prediction algorithm.




References
[1] https://www.openintro.org/data/index.php?data=loans_full_schema
