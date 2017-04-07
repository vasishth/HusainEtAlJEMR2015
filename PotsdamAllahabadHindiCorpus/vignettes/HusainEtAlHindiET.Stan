data {
	int<lower=1> N;
	real y[N];   //outcome
	real sl[N];  //predictors
	real complexity[N];  //predictors
    real freq[N];
    real bifreq[N];
    real IC[N];
    real SC[N];
    int<lower=-1,upper=1> bl[N];
    real sl_bl[N];  
	real complexity_bl[N]; 
    real freq_bl[N];
    real bifreq_bl[N];
    real IC_bl[N];
    real SC_bl[N];
	int<lower=1> I;                  //number of subjects
	int<lower=1> K;                  //number of items
	int<lower=1, upper=I> subject[N];   //subject id
	int<lower=1, upper=K> item[N];   //item id
}

parameters {
	vector[14] beta;			// intercept and slope
	real<lower=0> sigma_e;		// residual sd
	vector<lower=0>[14] sigma_u;	// subj sd
	vector<lower=0>[14] sigma_w;	// item sd
	cholesky_factor_corr[14] L_u;
	cholesky_factor_corr[14] L_w;
	matrix[14,I] z_u;
	matrix[14,K] z_w;
}

model {
	real mu[N]; 	// mu for likelihood
	matrix[I,14] u;	// random intercept and slopes subj
	matrix[K,14] w;
	
	# priors:
	beta ~ normal(0,10);
	sigma_e ~ normal(0,8); 
	sigma_u ~ normal(0,8);
	sigma_w ~ normal(0,8);
	L_u ~ lkj_corr_cholesky(2.0);
	L_w ~ lkj_corr_cholesky(2.0);
	to_vector(z_u) ~ normal(0,1);
	to_vector(z_w) ~ normal(0,1);
	
	u <- (diag_pre_multiply(sigma_u,L_u) * z_u)';	
	w <- (diag_pre_multiply(sigma_w,L_w) * z_w)';	
	
	for (n in 1:N)
		mu[n] <- beta[1] + u[subject[n],1] + w[item[n],1] + 
		(beta[2] + u[subject[n],2] + w[item[n],2])*sl[n] +
		(beta[3] + u[subject[n],3] + w[item[n],3])*complexity[n] +
		(beta[4] + u[subject[n],4] + w[item[n],4])*freq[n] +
     	(beta[5] + u[subject[n],5] + w[item[n],5])*bifreq[n] +
        (beta[6] + u[subject[n],6] + w[item[n],6])*IC[n] +
        (beta[7] + u[subject[n],7] + w[item[n],7])*SC[n]+
        (beta[8] + u[subject[n],8] + w[item[n],8])*bl[n]+
        (beta[9] + u[subject[n],9] + w[item[n],9])*sl_bl[n]+
        (beta[10] + u[subject[n],10] + w[item[n],10])*complexity_bl[n]+
        (beta[11] + u[subject[n],11] + w[item[n],11])*freq_bl[n]+
        (beta[12] + u[subject[n],12] + w[item[n],12])*bifreq_bl[n]+
        (beta[13] + u[subject[n],13] + w[item[n],13])*IC_bl[n]+
        (beta[14] + u[subject[n],14] + w[item[n],14])*SC_bl[n];
	y ~ normal(mu,sigma_e);        // likelihood
}
