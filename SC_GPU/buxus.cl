#pragma OPENCL EXTENSION cl_khr_fp64 : enable

void kernel func(global double *y, global double *S, const double d, const int N)
{
	const int idx = get_global_id(0);
	S[idx] = 1;
		S[idx] = 2 * y[idx] / d;
	
}

void kernel SoftCombine(global double *M_l, global double *M_l_1, global bool *C_l, const int N)
{
const int idx = get_global_id(0);
if (idx < N){
if (C_l[idx]) {
M_l[idx] = M_l_1[idx+N] - M_l_1[idx];
}
else {
M_l[idx] = M_l_1[idx+N] + M_l_1[idx];
}
}
}

void kernel SoftXOR(global double *M_l, global double *M_l_1, const int N) {

	const int idx = get_global_id(0);
	if (idx < N){

		if (M_l_1[idx] < M_l_1[idx + N])
		{
			if (M_l_1[idx] > -1 * M_l_1[idx + N])
			{
				M_l[idx] = M_l_1[idx];
			}
			else
			{
				M_l[idx] = -1 * M_l_1[idx + N];
			}
		}
		else
		{
			if (M_l_1[idx] > -1 * M_l_1[idx + N])
			{
				M_l[idx] = M_l_1[idx + N];
			}
			else
			{
				M_l[idx] = -1 * M_l_1[idx];
			}
		}
	}
}

void kernel XOR(global bool *M_l, global bool *M_l_1, const int N1, const int N2, const int N) {
	const int idx = get_global_id(0);
	if (idx < N){
		M_l[idx + N2] = M_l_1[idx] != M_l[idx + N1];
	}
}


void kernel wierdest_kernel_ever(global bool *C) {
	const int idx = get_global_id(0);
		C[idx] = false;
}

void kernel kernel_even_wierder(global bool *C, global double *S) {
	const int idx = get_global_id(0);
	if (S[idx] > 0) {
			C[idx] = false;
	}
	else {
			C[idx] = true;
	}
}


void kernel Q(global double *M_l, global double *M_l_1, const int N) {

	const int idx = get_global_id(0);

	if (idx < N){
		if ((M_l_1[2 * idx] < 0 && M_l_1[2 * idx + 1] < 0) || (M_l_1[2 * idx]>0 && M_l_1[2 * idx + 1]>0))
			M_l[idx] = (fabs((float)M_l_1[2 * idx]) < fabs((float)M_l_1[2 * idx + 1])) ? fabs((float)M_l_1[2 * idx]) : fabs((float)M_l_1[2 * idx + 1]);
		else
		M_l[idx] = (fabs((float)M_l_1[2 * idx]) < fabs((float)M_l_1[2 * idx + 1])) ? -1 * fabs((float)M_l_1[2 * idx]) : -1 * fabs((float)M_l_1[2 * idx + 1]);
	}
}

void kernel P_GPU(global bool *C, global double *S_, global double *S1, const int N) {
	const int idx = get_global_id(0);
	if (idx < N){
		if (C[idx] == 0) {
			S_[idx] = S1[2 * idx] + S1[2 * idx + 1];
		}
		else {
			S_[idx] = S1[2 * idx + 1] - S1[2 * idx];
		}
	}
}

void kernel updC(global bool *C_, global bool *C0, global bool *C1, const int N) {
	const int idx = get_global_id(0);
	if (idx < N){
		C_[2 * idx] = C0[idx] != C1[idx];
		C_[2 * idx + 1] = C1[idx];
	}
}


/*void subUpdC(global bool *C0, global bool *C1, const int N, const int lambda, const int fi, const int m){
	const int idx = get_global_id(0);
	int lambda2 = lambda;
	int Depth = 0;
	while (true){
		int mid = 1< Depth;
		int mid2 = (fi + 1) % mid;
		if (mid2 == 0)
			Depth++;
		else break;

	}
	Depth--;
	unsigned lambda0 = lambda - Depth;
	
	for (lambda2; lambda > lambda0; lambda2--){
		int psi = fi / 2;
		int psm2 = psi % 2;
		int N1 = (1 << (m - lambda));

		if (psm2 == 0){
			if (idx < N1){
				C0[N - (1 << (4 - lambda - 1)) + 2 * idx] = C0[N - (1 << (4 - lambda)) + idx] != C1[N - (1 << (4 - lambda)) + idx];
				C0[N - (1 << (4 - lambda)) + 2 * idx + 1] = C1[N - (1 << (4 - lambda)) + idx];
			}
		}
		else{
			if (idx < N1){
				C1[N - (1 << (4 - lambda - 1)) + 2 * idx] = C0[N - (1 << (4 - lambda)) + idx] != C1[N - (1 << (4 - lambda)) + idx];
				C1[N - (1 << (4 - lambda)) + 2 * idx + 1] = C1[N - (1 << (4 - lambda)) + idx];
			}
		}

	}

}*/

void kernel recursivelyUpdC(global bool *C0, global bool *C1, const int N, int lambda, const int fi, const int m) {

	const int idx = get_global_id(0);
	int Depth = 0;
	while ((fi+1) % (1<<Depth)) Depth++;
	Depth--;
	int lambda0 = lambda - Depth;
	int psi = fi;
	for (lambda; lambda > lambda0; lambda--){
		psi = psi / 2;
		int psm2 = psi % 2;
		int N1 = (1 << (m - lambda - 1));

		if (psm2 == 0){
			if (idx < N1){
				C0[N - (1 << (m - lambda + 1)) + 2 * idx] = C0[N - (1 << (m - lambda)) + idx] != C1[N - (1 << (m - lambda)) + idx];
				C0[N - (1 << (m - lambda + 1)) + 2 * idx + 1] = C1[N - (1 << (m - lambda)) + idx];
			}
		}
		else{
			if (idx < N1){
				C1[N - (1 << (m - lambda + 1)) + 2 * idx] = C0[N - (1 << (m - lambda)) + idx] != C1[N - (1 << (m - lambda)) + idx];
				C1[N - (1 << (m - lambda + 1)) + 2 * idx + 1] = C1[N - (1 << (m - lambda)) + idx];
			}
		}

	}
	
}

