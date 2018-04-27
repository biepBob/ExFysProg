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
            Weights[i] = rngesus(0,1); //Generate random weights
        }
        
        Bias = rngesus(0,1);
}


neuron::~neuron()
{
}


float neuron::rngesus(flo min, flo max){
    random_device rd; //Initializes random engine
    mt19937 gen(rd()); //Mersenne Twister 19937 generator, rng
    uniform_real_distribution<flo> dis(min, max); //uniform probability distribution
    return dis(gen);
}

void neuron::setWeights(vector<fp> w)
{
        Weights.resize(w.size());
	for (int i = 0; i < (int) w.size(); i++)
	{
		Weights[i] = w[i]; //noptr
	}
}

void neuron::setBias(fp b)
{
        Bias = b; //noptr
}

vector<fp> neuron::getWeights()
{
    vector<fp> TempWeights(Weights.size());
    for (int i = 0; i < (int) Weights.size(); i++)
	{
		TempWeights[i] = Weights[i]; //noptr
	}
    return TempWeights;
}

fp neuron::getBias()
{
        return Bias; //noptr
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
                temp += Weights[i] * input[i]; //w.x dot product //noptr
        }
        temp += Bias;
        return temp;
}

fp neuron::resultFunc(vector<fp> input) 
{
        Output = sigmoid(activateFunc(input));
        return  Output; //noptr
}

