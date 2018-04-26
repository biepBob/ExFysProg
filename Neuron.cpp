#include "Neuron.h"

neuron::neuron(vector<fp> weights,fp bias)
{
        setWeights(weights);
        setBias(bias);
}

neuron::neuron(int size)
{
        Weights.resize(size);
        for (int i = 0; i < size; i++)
        {
                random_device rd; //Initializes random engine
                mt19937 gen(rd()); //Mersenne Twister 19937 generator, rng
                uniform_real_distribution<flo> dis(0, 1); //uniform probability distribution
                Weights[i] = dis(gen); //Generate random weights
                Bias = dis(gen);
        }
}


neuron::~neuron()
{
}

void neuron::setWeights(vector<fp> w)
{
        Weights.resize(w.size());
	for (int i = 0; i < (int) w.size(); i++)
	{
		Weights[i] = *w[i];
	}
}

void neuron::setBias(fp b)
{
        Bias = *b;
}

vector<fp> neuron::getWeights()
{
    vector<fp> TempWeights(Weights.size());
    for (int i = 0; i < (int) Weights.size(); i++)
	{
		TempWeights[i] = &Weights[i];
	}
    return TempWeights;
}

fp neuron::getBias()
{
        return &Bias;
}

const int neuron::getNumberOfInputs()
{
        return Weights.size();
}

flo neuron::sigmoid(flo z)
{
        return 1 / (1 + exp(-z));
}

flo neuron::dsigmoid(flo z)
{
        return sigmoid(z)*(1 - sigmoid(z));
}

flo neuron::activateFunc(vector<fp> input)
{
        flo temp = 0;
        for (int i = 0; i < (int) input.size(); i++) 
        {
                temp += Weights[i] * *input[i]; //w.x dot product
        }
        temp += Bias;
        return temp;
}

fp neuron::resultFunc(vector<fp> input) 
{
        Output = sigmoid(activateFunc(input));
        return  &Output;
}

