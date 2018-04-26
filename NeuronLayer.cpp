#include "NeuronLayer.h"

layer::layer(vector<vector<fp> > LayerWeights, vector<fp> LayerBias){

    for (int i = 0; i < (int) LayerWeights.size(); ++i){
    Neurons.push_back(neuron(LayerWeights[i], LayerBias[i]));
    }


                
}


layer::layer(int nNeurons, int nWeights){//in every layer each neuron has the same amount of inputs and thus the same amount of weights
    Neurons.assign(nNeurons, neuron(nWeights));
}


layer::~layer(){
}

void layer::setWeights(vector<vector<fp> > LayerWeights){

    for(int i = 0; i < (int) LayerWeights.size(); ++i){
        Neurons[i].setWeights(LayerWeights[i]);
    }

}

void layer::setBias(vector<fp> LayerBias){

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

vector<fp> layer::resultFunc(vector<vector<fp> > LayerInputs){

    vector<fp> tmp(LayerInputs.size());
    for(int i = 0; i < (int) Neurons.size(); ++i){
    tmp[i] = Neurons[i].resultFunc(LayerInputs[i]);
    }
    return tmp;

}

vector<flo> layer::dsigmoid(vector<flo>  LayerInputs){

    vector<flo> tmp(LayerInputs.size());
    for(int i = 0; i < (int) Neurons.size(); ++i){
    tmp[i] = Neurons[i].dsigmoid(LayerInputs[i]);
    }
    return tmp;
}

