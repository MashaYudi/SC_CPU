/*#include "SCdecoder.h"
#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;

SCdecoder::SCdecoder(int _n, set<int> _u, double _dispersion) {

	n = _n;
	m = log2(n);
	u = _u;
	dispersion = _dispersion;
	S = new double[n];
	out = new int[n];
}

int*  SCdecoder::decode(long double* y){

	for (int beta = 0; beta < n; beta++) {
		S[beta] = 2 * y[beta] / dispersion;
	}

	for (int i = 0; i < n; i++){
		if (u.find(i) != u.end()){
			out[i] = 0;
		}
		else{
			if (S[i]>0)
				out[i] = 0;
			else
				out[i] = 1;
			}
	}
	return out;
}*/