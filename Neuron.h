#pragma once
#include<iostream>
#include<cmath>
#include<vector>
#include<random>

using namespace std;

typedef float* fp;
typedef float flo;

class neuron{
public:
	neuron(vector<fp> weights,fp  bias); //Constructor1, weight provided
	neuron(int); //Constructor2, no weight provided, start of the program
	~neuron(); //Destructor
	void setWeights(vector<fp>); //Sets provided weights to vector Weights
	void setBias(fp); //Sets provided bias to variable Bias

	vector<fp> getWeights(); //get-function to access weights
	fp getBias(); //get-function to access bias
	const int getNumberOfInputs(); //get-function to access #inputs = size of Weights
	flo sigmoid(flo); //Sigmoid function
	flo dsigmoid(flo); //Derivative Sigmoid function
	flo activateFunc(vector <fp>); //Activate function, calls sigmoid
	fp resultFunc(vector <fp>); //Calculates the neuron output, calls activateFunc
	void operator()(vector<fp> inputs) { resultFunc(inputs); } //Overloading ()
protected:
	vector<flo> Weights;
	flo Bias;
        flo Output;
};
