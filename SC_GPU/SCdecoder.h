/*#pragma once
#include <stack>
#include <iostream>
#include <vector>
#include <set>
using namespace std;

class SCdecoder {
private:
	int n;
	int m;
	double *S;
	set<int> u;
	double dispersion;
	int* out;
public:
	SCdecoder::SCdecoder(int _n, set<int> _u, double _dispersion);
	int* decode(long double* y);
};*/