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

    //experimenting with algorithms and lambda functions, not working atm so ignore it
    //size_t index = 0;
    //std::transform(LayerWeights.begin(),LayerWeights.end(), Neurons.begin(), [&](vector<flo>  lw)  {(Neurons.at(index)).setWeights(lw); ++index;});

    for(vector<neuron>::iterator it = Neurons.begin(); it != Neurons.end(); ++it){
        (*it).setWeights(LayerWeights.at(it-Neurons.begin())); //it-Neurons.begin() is way to calculate the integer index corresponding to the current  iterator
    }
}

void layer::setBias(vector<flo> LayerBias){ //sets bias for every neuron

    if(LayerBias.size() != Neurons.size()){
        throw std::invalid_argument("\nlayer::setBias: dimension mismatch\n");
    }
    

    for(vector<neuron>::iterator it = Neurons.begin(); it != Neurons.end(); ++it){
        (*it).setBias(LayerBias.at(it - Neurons.begin()));
    }

}



vector<vector<flo> > layer::getWeights(){ //gets the weights from every neuron

    vector<vector<flo> > tmp(Neurons.size()); //temporary vector to store data
    int count = 0;

    std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](vector<flo>) -> vector<flo> { vector<flo> temp = Neurons.at(count).getWeights(); ++count; return temp;}); //std algorithm that transforms empty tmp to tmp filled with weight vectors

    return tmp;
}

vector<flo> layer::getBias(){ //gets the bias from every neuron
    vector<flo> tmp(Neurons.size());
    int count = 0;

    std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](flo) -> flo { flo temp = Neurons.at(count).getBias(); ++count; return temp;});

    return tmp;

}

int layer::getNumberofNeurons(){ //the number of neurons in the layer

    return Neurons.size();
}

vector<flo> layer::resultFunc(vector<fp>  LayerInputs){//calculates the output for each neuron in the layer
//LayerInputs.size = LayerWeights[i].size() = #neurons in the previous layer

    vector<flo> tmp(Neurons.size()); //temporary vector to store data

    //test if it's the first layer 
    if(FirstLayer){//passes the i-th element of the input vector to the i-th neuron

        if(LayerInputs.size() != Neurons.size()){
            throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
        }

        int count = 0; //counter to access vector elements inside lambda function
        //lamda function: [capture1,...] (type1,...) -> type {code} : executes the code inside {} and returns a value of type 'type'. () contains the type of input parameters for the code. [] contains the variables that the lambda can capture from outside its scope. In this case it can capture the count.
        std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](flo) -> flo {flo temp = Neurons.at(count).resultFunc({LayerInputs.at(count)}); ++count; return temp; });

        
    }
    //for other layers than the first
    else{//passes the whole input vector to every neuron


        if(LayerInputs.size() != getWeights()[0].size()){
            //[0] because it's a priori known that every sub vector in LayerWeights will be the same size. If that's not the case it needs to be placed in the for loop.
            throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
        }

        int count = 0;
        std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](flo) -> flo {flo temp = Neurons.at(count).resultFunc(LayerInputs); ++count; return temp; });


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

        int count = 0;
        std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](flo) -> flo {flo temp = Neurons.at(count).dsigmoid(Neurons.at(count).activateFunc({LayerInputs.at(count)})); ++count; return temp; });


    }

    //for other layers than the first
    else{//passes the whole input vector to every neuron
        if(LayerInputs.size() != getWeights()[0].size()){
            //[0] because it's a priori known that every sub vector in LayerWeights will be the same size. If that's not the case it needs to be placed in the for loop.
            throw std::invalid_argument("\nlayer::dsigmoid: dimension mismatch\n");
        }


        int count = 0;
        std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](flo) -> flo {flo temp = Neurons.at(count).dsigmoid(Neurons.at(count).activateFunc(LayerInputs)); ++count; return temp; });

        }

    return tmp;
}

