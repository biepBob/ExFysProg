#pragma once
#include<iostream>
#include<cmath>
#include<vector>
#include<algorithm>
#include<random>

using namespace std;

typedef float* fp;
typedef float flo;

class neuron{
public:
	neuron(vector<flo> weights,flo  bias); //Constructor1, weight provided
	neuron(int); //Constructor2, no weight provided, start of the program
	~neuron(); //Destructor
        neuron(const neuron &n1); //Copy constructor for Weights, Bias and Output
        neuron& operator = (const neuron &otherNeuron); //Assignment operator for neuron
        float rngesus(flo min, flo max);
	void setWeights(vector<flo>); //Sets provided weights to vector Weights
	void setBias(flo); //Sets provided bias to variable Bias

	vector<flo> getWeights(); //get-function to access weights
	flo getBias(); //get-function to access bias
	const int getNumberOfInputs(); //get-function to access #inputs = size of Weights
	flo sigmoid(flo); //Sigmoid function
	flo dsigmoid(flo); //Derivative Sigmoid function
	flo activateFunc(vector <fp>); //Activate function, calls sigmoid
	flo resultFunc(vector <fp>); //calculates the neuron output, calls activateFunc
	flo  operator()(vector<fp> inputs) { return resultFunc(inputs); } //Overloading ()
protected:
	vector<flo> Weights;
	flo Bias;
        flo Output;

        //flo deref(fp in);
};
