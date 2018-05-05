#include "NeuronLayer.h"

layer::layer(vector<vector<flo> > LayerWeights, vector<flo> LayerBias, bool firstLayer){//constructor1
    FirstLayer = firstLayer; //sets the value for FirstLayer for further use in resultFunc and dsigmoid

    //argument check
    if (LayerWeights.size() != LayerBias.size()){
        throw std::invalid_argument("\n layer::layer1: dimension mismatch\n");
    }

    NumberofNeurons = LayerWeights.size(); //#neurons = size of LayerWeights
    NumberofInputs = LayerWeights.at(0).size(); //Because it is a priori known that every neuron takes an equal amount of inputs



    for (int i = 0; i < (int) LayerWeights.size(); ++i){ //creates a vector of neurons using neuron/constructor2
    Neurons.push_back(neuron(LayerWeights.at(i), LayerBias.at(i))); //using iterators here is impractical, though we might fix it in future versions
    }

                
}


layer::layer(int nNeurons, int nWeights, bool firstLayer){//in every layer each neuron has the same amount of inputs and thus the same amount of weights
    FirstLayer = firstLayer;

    if(nNeurons <= 0 || nWeights <= 0) {
        throw std::invalid_argument("\nlayer::layer2: invalid argument, argument must be of type int.\n");
    }
    Neurons.assign(nNeurons, neuron(nWeights));//calls constructor2
    NumberofNeurons = nNeurons; //sets the parameters
    NumberofInputs = nWeights;
}


layer::~layer(){ //destructor
}


layer::layer(const layer &layer1){ //copy constructor

    Neurons = layer1.Neurons;
    NumberofNeurons = layer1.NumberofNeurons;
    NumberofInputs = layer1.NumberofInputs;
    FirstLayer = layer1.FirstLayer;
}

layer& layer::operator = (const layer &otherLayer){ //assignment operator

    if(&otherLayer != this){
    Neurons = otherLayer.Neurons;
    NumberofNeurons = otherLayer.NumberofNeurons;
    NumberofInputs = otherLayer.NumberofInputs;
    FirstLayer = otherLayer.FirstLayer;       
    }
    return *this;
}

void layer::setWeights(vector<vector<flo> > LayerWeights){
    if(LayerWeights.size() != Neurons.size()){
        throw std::invalid_argument("\nlayer::setWeights: dimension mismatch\n");
    }

    int index = 0;
    std::for_each(Neurons.begin(),Neurons.end(), [&](neuron &Neuron){Neuron.setWeights(LayerWeights.at(index++));});
    //[&]: lambda function needs access to elements outside its scope: index and LayerWeights
    //(neuron &Neuron): for_each does something similar to f(*iterator), with f being the function from the last argument and iterator the current iterator of the container in the first argument. *iterator is thus an object of type neuron that will be used as input for the lambda function
    //{Neuron.setWeights(...)} will call neuron::setWeights on the provided Neuron
    //f(i++) = f(i); ++i;
    //f(++i) = ++i; f(i); this may cause an 'out_of_range' error in f
    //f(i++); h(i)  = f(i); ++i; h(i); this may cause an 'out_of_range' error in h, prevent it by only doing i++ in the function that will be called last

}

void layer::setBias(vector<flo> LayerBias){ //sets bias for every neuron

    if(LayerBias.size() != Neurons.size()){
        throw std::invalid_argument("\nlayer::setBias: dimension mismatch\n");
    }
    
    int index = 0;
    std::for_each(Neurons.begin(),Neurons.end(), [&](neuron &Neuron){Neuron.setBias(LayerBias.at(index++));});
}



vector<vector<flo> > layer::getWeights(){ //gets the weights from every neuron

    vector<vector<flo> > tmp(Neurons.size()); //temporary vector to store data
    
    std::transform(Neurons.begin(),Neurons.end(),tmp.begin(),[&](neuron &Neuron){return Neuron.getWeights();}); //std algorithm that transforms empty tmp to tmp filled with weight vectors

    return tmp;
}

vector<flo> layer::getBias(){ //gets the bias from every neuron
    vector<flo> tmp(Neurons.size());

    std::transform(Neurons.begin(),Neurons.end(),tmp.begin(),[&](neuron &Neuron){return Neuron.getBias();});

    return tmp;

}

int layer::getNumberofNeurons(){ //the number of neurons in the layer

    return Neurons.size();
}

vector<fp> layer::resultFunc(vector<fp>  LayerInputs){//calculates the output for each neuron in the layer
//LayerInputs.size = LayerWeights[i].size() = #neurons in the previous layer

    vector<fp> tmp(Neurons.size()); //temporary vector to store data

    //test if it's the first layer 
    if(FirstLayer){//passes the i-th element of the input vector to the i-th neuron
        if(LayerInputs.size() != Neurons.size()){
            throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
        }

        int index = 0; //indexer to access vector elements inside lambda function
        //lamda function: [capture1,...] (type1,...) -> type {code} : executes the code inside {} and returns a value of type 'type'. () contains the type of input parameters for the code. [] contains the variables that the lambda can capture from outside its scope. In this case it can capture the index.
        std::transform(Neurons.begin(),Neurons.end(),tmp.begin(), [&](neuron &Neuron) -> fp {return  Neuron.resultFunc({LayerInputs.at(index++)});});
        
    }
    //for other layers than the first
    else{//passes the whole input vector to every neuron

        if(LayerInputs.size() != getWeights()[0].size()){
            //[0] because it's a priori known that every sub vector in LayerWeights will be the same size. If that's not the case it needs to be placed in the for loop.
            throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
        }

        std::transform(Neurons.begin(),Neurons.end(),tmp.begin(),[&](neuron &Neuron){return Neuron.resultFunc(LayerInputs);});

    }
    return tmp;
}

vector<flo> layer::dsigmoid(vector<fp>  LayerInputs){

    vector<flo> tmp(Neurons.size());//LayerInputs.size = LayerWeights[i].size() = #neurons in the previous layer


    //test if it's the first layer 
    if(FirstLayer){//passes the i-th element of the input vector to the i-th neuron

        if(LayerInputs.size() != Neurons.size()){
            throw std::invalid_argument("\nlayer::dsigmoid: dimension mismatch\n");
        }

        int index = 0;
        std::transform(Neurons.begin(),Neurons.end(),tmp.begin(), [&](neuron &Neuron){ vector<fp> in = {LayerInputs.at(index++)}; fp t = Neuron.activateFunc(in); return  *Neuron.dsigmoid(t);});
    }

    //for other layers than the first
    else{//passes the whole input vector to every neuron
        if(LayerInputs.size() != getWeights()[0].size()){
            //[0] because it's a priori known that every sub vector in LayerWeights will be the same size. If that's not the case it needs to be placed in the for loop.
            throw std::invalid_argument("\nlayer::dsigmoid: dimension mismatch\n");
        }

        std::transform(Neurons.begin(),Neurons.end(), tmp.begin(), [&](neuron &Neuron){return *Neuron.dsigmoid(Neuron.activateFunc(LayerInputs));});

        }

    return tmp;
}

