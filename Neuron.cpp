#include "Neuron.h"

neuron::neuron(vector<flo> weights,flo bias) //constructor 1
{
        setWeights(weights); //gives provided parameters to the class
        setBias(bias);
}

neuron::neuron(int size) //constructor 2
{
        Weights.resize(size);

        std::generate(Weights.begin(),Weights.end(),[&](){return rngesus(-1,1);}); //generates random weights using algorithms and lambda function
        
        Bias = rngesus(-1,1);
}


neuron::~neuron() //destructor
{
}

neuron::neuron(const neuron &n1){ //copy constructor
    Weights = n1.Weights;
    Bias = n1.Bias;
    Output = n1.Output;
}


neuron& neuron::operator = (const neuron &otherNeuron){ //assignment operator

    if( &otherNeuron != this){
        Weights = otherNeuron.Weights;
        Bias = otherNeuron.Bias;
        Output = otherNeuron.Output;
    }
    return *this;
}


flo neuron::rngesus(flo min, flo max){ //random number generator
    random_device rd; //Initializes random engine
    mt19937 gen(rd()); //Mersenne Twister 19937 generator, rng
    uniform_real_distribution<flo> dis(min, max); //uniform probability distribution
    return dis(gen);
}


void neuron::setWeights(vector<flo> w)
{
   Weights = w; //sets the weights
}

void neuron::setBias(flo b)
{
        Bias = b; //sets the bias
}

vector<flo> neuron::getWeights()
{
    return Weights; //returns the weights
}

flo neuron::getBias()
{
        return Bias; //returns the bias
}

const int neuron::getNumberOfInputs()
{
        return Weights.size(); //returns the number of inputs for a neuron
}

flo neuron::sigmoid(flo z)
{
        return 1 / (1 + exp(-z)); //sigmoid function
} 

flo neuron::dsigmoid(flo z)
{
        return sigmoid(z)*(1 - sigmoid(z)); //sigmoid derivative
}

flo neuron::activateFunc(vector<fp> input)
{
        flo temp = 0; //temporary vector to store output
        vector<flo> buff(input.size()); //temporary vector to store transformed elements
            std::transform(input.begin(),input.end(),buff.begin(),[](fp in) -> flo {return *in;}); //converts input, a vector of ptrs to buff, a vector of floats
            //last argument is a lambda function. It takes input of type fp and sends it to code in {} to return a flo

            temp = std::inner_product(Weights.begin(),Weights.end(),buff.begin(), Bias); //std algorithm to calculate the inner product, i.e. sum of products
        return temp;
}

flo neuron::resultFunc(vector<fp> input) //calculates the output of a neuron
{
    Output = sigmoid(activateFunc(input));
        return  Output;
}

