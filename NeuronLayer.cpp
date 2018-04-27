#include "NeuronLayer.h"

layer::layer(vector<vector<fp> > LayerWeights, vector<fp> LayerBias, bool firstLayer){
    FirstLayer = firstLayer; //sets the value for FirstLayer for further use in resultFunc and dsigmoid

    //argument check
    if (LayerWeights.size() != LayerBias.size()){
        throw std::invalid_argument("\n layer::layer1: dimension mismatch\n");
    }

    NumberofNeurons = LayerWeights.size();
    NumberofInputs = LayerWeights[0].size(); //Because it is a priori known that every neuron takes an equal amount of inputs

    for (int i = 0; i < (int) LayerWeights.size(); ++i){
    Neurons.push_back(neuron(LayerWeights[i], LayerBias[i]));
    }


                
}


layer::layer(int nNeurons, int nWeights, bool firstLayer){//in every layer each neuron has the same amount of inputs and thus the same amount of weights
    FirstLayer = firstLayer;

    if(nNeurons <= 0 || nWeights <= 0) {
        throw std::invalid_argument("\nlayer::layer2: invalid argument, argument must be of type int.\n");
    }
    Neurons.assign(nNeurons, neuron(nWeights));
}


layer::~layer(){
}


layer::layer(const layer &layer1){

    Neurons = layer1.Neurons;
    NumberofNeurons = layer1.NumberofNeurons;
    NumberofInputs = layer1.NumberofInputs;
    FirstLayer = layer1.FirstLayer;
}

layer& layer::operator = (const layer &otherLayer){

    if(&otherLayer != this){
    Neurons = otherLayer.Neurons;
    NumberofNeurons = otherLayer.NumberofNeurons;
    NumberofInputs = otherLayer.NumberofInputs;
    FirstLayer = otherLayer.FirstLayer;       
    }
    return *this;
}

void layer::setWeights(vector<vector<fp> > LayerWeights){

    if(LayerWeights.size() != Neurons.size()){
        throw std::invalid_argument("\nlayer::setWeights: dimension mismatch\n");
    }


    for(int i = 0; i < (int) LayerWeights.size(); ++i){
        Neurons[i].setWeights(LayerWeights[i]);
    }

}

void layer::setBias(vector<fp> LayerBias){

    if(LayerBias.size() != Neurons.size()){
        throw std::invalid_argument("\nlayer::setBias: dimension mismatch\n");
    }
    for(int i = 0; i < (int) LayerBias.size(); ++i){
        Neurons[i].setBias(LayerBias[i]);
    }

}



vector<vector<fp> > layer::getWeights(){
    vector<vector<fp> > tmp(Neurons.size());

    for(int i = 0; i < (int) Neurons.size(); ++i){
        tmp[i] = Neurons[i].getWeights();
    }
    return tmp;
}

vector<fp> layer::getBias(){
    vector<fp> tmp(Neurons.size());

    for(int i = 0; i < (int) Neurons.size(); ++i){
        tmp[i] = Neurons[i].getBias();
    }
    return tmp;

}

int layer::getNumberofNeurons(){

    return Neurons.size();
}

vector<fp> layer::resultFunc(vector<fp>  LayerInputs){


    vector<fp> tmp(Neurons.size());//LayerInputs.size = Neurons.size

    //test if it's the first layer 
    if(FirstLayer){//passes the i-th element of the input vector to the i-th neuron

        if(LayerInputs.size() != Neurons.size()){
            throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
        }

        for(int i=0; i< (int) Neurons.size(); ++i){
            tmp[i] = Neurons[i].resultFunc({LayerInputs[i]});    
        }

    }
    //for other layers than the first
    else{//passes the whole input vector to every neuron

        if(LayerInputs.size() != getWeights()[0].size()){
            //[0] because it's a priori known that every sub vector in LayerWeights will be the same size. If that's not the case it needs to be placed in the for loop.
            throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
        }

        for(int i = 0; i < (int) Neurons.size(); ++i){
            tmp[i] = Neurons[i].resultFunc(LayerInputs);
        }
    }
    
    return tmp;

}

vector<flo> layer::dsigmoid(vector<fp>  LayerInputs){

    

    vector<flo> tmp(LayerInputs.size());//LayerInputs.size = Neurons.size

    //test if it's the first layer 
    if(FirstLayer){//passes the i-th element of the input vector to the i-th neuron

        if(LayerInputs.size() != Neurons.size()){
            throw std::invalid_argument("\nlayer::dsigmoid: dimension mismatch\n");
        }

        for(int i=0; i< (int) Neurons.size(); ++i){
            tmp[i] = Neurons[i].dsigmoid(Neurons[i].activateFunc({LayerInputs[i]}));        }
    }
    //for other layers than the first
    else{//passes the whole input vector to every neuron

        if(LayerInputs.size() != getWeights()[0].size()){
            //[0] because it's a priori known that every sub vector in LayerWeights will be the same size. If that's not the case it needs to be placed in the for loop.
            throw std::invalid_argument("\nlayer::dsigmoid: dimension mismatch\n");
        }

        for(int i = 0; i < (int) Neurons.size(); ++i){
            tmp[i] = Neurons[i].dsigmoid(Neurons[i].activateFunc(LayerInputs));
        }
    }

    return tmp;
}

