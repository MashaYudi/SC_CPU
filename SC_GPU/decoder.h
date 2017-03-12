#ifndef _DECODER_H_
#define _DECODER_H_

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
//#include <CL/cl.h>
#include <CL/cl.hpp>

using namespace std;
class Decoder {
	const int m;
	const int N;
	const int K;
	const double dispersion;
	//bool *frozen;
	//bool ***C;
	int *C0;
	int *C1;
	//bool **C;
	int *indeces;
	int *shuffle;
	double ***P;
	double **S;
	set<int> frozen;
	std::vector<cl::Buffer>buffer_S; 
	//std::vector<cl::Buffer>buffer_C0;
	//std::vector<cl::Buffer>buffer_C1;

	cl::Buffer y_buf;
	cl::Buffer fullbuf_C0;
	cl::Buffer fullbuf_C1;

	cl::CommandQueue queue;
	//cl::Context context;
	cl::Program::Sources sources;
	cl::Program program;
	cl::Kernel kernel_S;
	cl::Kernel kernel_softCombine;
	cl::Kernel kernel_softxor;
	cl::Kernel kernel_xor;
	cl::Kernel kernel_wke;
	cl::Kernel kernel_kew;
	cl::Kernel Q_GPU;
	cl::Kernel P_GPU;
	cl::Kernel kernel_updC;
	cl::Kernel recUpdC_subUpd;
public:
	Decoder(int n, int k, set<int> frozen, double sd);
	//~Decoder();
	void decode(double *y, bool *c);

private:
	inline double W(double y, double x);
	void recursivelyCalcP(int lambda, int fi);
	void recursivelyUpdateC(int lambda, int fi);
	void recursivelyUpdateC_GPU(int lambda, int fi);
	void recursivelyCalcS(int lambda, int phi);
	void recursivelyCalcS_GPU(int lambda, int phi);
	void IterativelyCalcS(unsigned lambda//layer ID
		, int phi
		);
	void IterativelyCalcS_CPU(unsigned lambda//layer ID
		, int phi);
	void IterativelyUpdateC(unsigned lambda, unsigned phi);
	void IterativelyUpdateC_CPU(unsigned lambda, unsigned phi);
	cl::Buffer getC(unsigned lambda, int phi);
};

#endif /* _DECODER_H_ */
