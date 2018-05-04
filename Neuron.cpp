#include "Neuron.h"

neuron::neuron(vector<flo> weights,flo bias)
{
        setWeights(weights);
        setBias(bias);
}

neuron::neuron(int size)
{
        Weights.resize(size);
        vector<flo>::iterator it;
        for (it = Weights.begin(); it != Weights.end(); ++it)
        {
            *it = rngesus(0,1); //Generate random weights
        }
        
        Bias = rngesus(0,1);
}


neuron::~neuron()
{
}

neuron::neuron(const neuron &n1){
    Weights = n1.Weights;
    Bias = n1.Bias;
    Output = n1.Output;
}


neuron& neuron::operator = (const neuron &otherNeuron){

    if( &otherNeuron != this){
        Weights = otherNeuron.Weights;
        Bias = otherNeuron.Bias;
        Output = otherNeuron.Output;
    }
    return *this;
}


float neuron::rngesus(flo min, flo max){
    random_device rd; //Initializes random engine
    mt19937 gen(rd()); //Mersenne Twister 19937 generator, rng
    uniform_real_distribution<flo> dis(min, max); //uniform probability distribution
    return dis(gen);
}


void neuron::setWeights(vector<flo> w)
{
    //Weights.resize(w.size());
    //std::transform(w.begin(),w.end(),Weights.begin(), [](fp in) -> flo {return *in;});
   Weights = w; 
    //for (int i = 0; i < (int) w.size(); i++){
	    //Weights.at(i) = *w.at(i);
    //}
}

void neuron::setBias(flo b)
{
        Bias = b;
}

vector<flo> neuron::getWeights()
{
    return Weights;
}

flo neuron::getBias()
{
        return Bias;
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
        vector<flo> buff(input.size());
            std::transform(input.begin(),input.end(),buff.begin(),[](fp in) -> flo {return *in;}); //converts input, a vector of ptrs to buff, a vector of floats
            temp = std::inner_product(Weights.begin(),Weights.end(),buff.begin(), Bias); //std algorithm to calculate the inner product, i.e. sum of products
        return temp;
}

flo neuron::resultFunc(vector<fp> input) 
{
    Output = sigmoid(activateFunc(input));
        return  Output;
}

