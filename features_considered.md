out of our 151 features, we are considering these for now

#	Feature	Type
1	loan_amnt	Numerical
2	term	Categorical
3	emp_length	Categorical
4	home_ownership	Categorical
5	annual_inc	Numerical
6	verification_status	Categorical
7	purpose	Categorical
8	dti	Numerical
9	delinq_2yrs	Numerical
10	inq_last_6mths	Numerical
11	open_acc	Numerical
12	pub_rec	Numerical
13	revol_bal	Numerical
14	revol_util	Numerical
15	total_acc	Numerical
16	fico_range_low	Numerical


Some particularly important exclusions:
- loan_status: Used to construct our target, not as a predictor.
- issue_d: Used for temporal splitting, not initially as a predictor.
- last_fico_range_low/high: Updated credit score information; potential leakage.
- total_pymnt, total_rec_prncp, recoveries: These are consequences of loan performance.
- hardship_* and settlement_*: Information from later stages of the loan lifecycle.
- grade, sub_grade: Existing lender-generated credit assessments.