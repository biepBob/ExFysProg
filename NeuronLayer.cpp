#include "NeuronLayer.h"

layer::layer(vector<vector<flo> > LayerWeights, vector<flo> LayerBias, bool firstLayer){
    FirstLayer = firstLayer; //sets the value for FirstLayer for further use in resultFunc and dsigmoid

    //argument check
    if (LayerWeights.size() != LayerBias.size()){
        throw std::invalid_argument("\n layer::layer1: dimension mismatch\n");
    }

    NumberofNeurons = LayerWeights.size(); //#neurons = size of LayerWeights
    NumberofInputs = LayerWeights.at(0).size(); //Because it is a priori known that every neuron takes an equal amount of inputs

    for (int i = 0; i < (int) LayerWeights.size(); ++i){ //creates a vector of neurons using neuron/constructor2
    Neurons.push_back(neuron(LayerWeights.at(i), LayerBias.at(i))); //using iterators here is impractical
    }


                
}


layer::layer(int nNeurons, int nWeights, bool firstLayer){//in every layer each neuron has the same amount of inputs and thus the same amount of weights
    FirstLayer = firstLayer;

    if(nNeurons <= 0 || nWeights <= 0) {
        throw std::invalid_argument("\nlayer::layer2: invalid argument, argument must be of type int.\n");
    }
    Neurons.assign(nNeurons, neuron(nWeights));
    NumberofNeurons = nNeurons;
    NumberofInputs = nWeights;
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

void layer::setWeights(vector<vector<flo> > LayerWeights){
    if(LayerWeights.size() != Neurons.size()){
        throw std::invalid_argument("\nlayer::setWeights: dimension mismatch\n");
    }

    //experimenting with algorithms and lambda functions, not working atm so ignore it
    //size_t index = 0;
    //std::transform(LayerWeights.begin(),LayerWeights.end(), Neurons.begin(), [&](vector<flo>  lw)  {(Neurons.at(index)).setWeights(lw); ++index;});

    for(vector<neuron>::iterator it = Neurons.begin(); it != Neurons.end(); ++it){
        (*it).setWeights(LayerWeights.at(it-Neurons.begin())); //it-Neurons.begin() is way to calculate the integer index corresponding to an iterator
    }



    //for(int i = 0; i < Neurons.size(); ++i){
        //Neurons[i].setWeights(LayerWeights[i]);
    //}

}

void layer::setBias(vector<flo> LayerBias){

    if(LayerBias.size() != Neurons.size()){
        throw std::invalid_argument("\nlayer::setBias: dimension mismatch\n");
    }
    //size_t index = 0;
    //vector<flo> tmp;
    //std::transform(Neurons.begin(),Neurons.end(), Neurons.begin(), [&](void) { Neurons.at(index).setBias(LayerBias[index]);});

    for(vector<neuron>::iterator it = Neurons.begin(); it != Neurons.end(); ++it){
        (*it).setBias(LayerBias.at(it - Neurons.begin()));
    }



    for(int i = 0; i < Neurons.size(); ++i){
        Neurons[i].setBias(LayerBias[i]);
    }

}



vector<vector<flo> > layer::getWeights(){

    vector<vector<flo> > tmp(Neurons.size());
    int count = 0;

    std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](vector<flo>) -> vector<flo> { vector<flo> temp = Neurons.at(count).getWeights(); ++count; return temp;});


    //for(int i = 0; i < Neurons.size(); ++i){
        //tmp[i] = Neurons[i].getWeights();
    //}

    return tmp;
}

vector<flo> layer::getBias(){
    vector<flo> tmp(Neurons.size());
    int count = 0;

    std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](flo) -> flo { flo temp = Neurons.at(count).getBias(); ++count; return temp;});

    //for(int i = 0; i < Neurons.size(); ++i){
        //tmp[i] = Neurons[i].getBias();
    //}

    return tmp;

}

int layer::getNumberofNeurons(){

    return Neurons.size();
}

vector<flo> layer::resultFunc(vector<fp>  LayerInputs){


    vector<flo> tmp(Neurons.size());//LayerInputs.size = Neurons.size

    //test if it's the first layer 
    if(FirstLayer){//passes the i-th element of the input vector to the i-th neuron

        if(LayerInputs.size() != Neurons.size()){
            throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
        }
        int count = 0;
        std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](flo) -> flo {flo temp = Neurons.at(count).resultFunc({LayerInputs.at(count)}); ++count; return temp; });

        //for(int i = 0; i < Neurons.size(); ++i){
            //tmp[i] = Neurons[i].resultFunc({LayerInputs[i]});
        //}

    }
    //for other layers than the first
    else{//passes the whole input vector to every neuron


        if(LayerInputs.size() != getWeights()[0].size()){
            //[0] because it's a priori known that every sub vector in LayerWeights will be the same size. If that's not the case it needs to be placed in the for loop.
            throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
        }

        int count = 0;
        std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](flo) -> flo {flo temp = Neurons.at(count).resultFunc(LayerInputs); ++count; return temp; });

        for(int i = 0; i < Neurons.size(); ++i){
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

        int count = 0;
        std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](flo) -> flo {flo temp = Neurons.at(count).dsigmoid(Neurons.at(count).activateFunc({LayerInputs.at(count)})); ++count; return temp; });

        //for(int i = 0; i < Neurons.size(); ++i){
            //tmp[i] = Neurons[i].dsigmoid(Neurons[i].activateFunc({LayerInputs[i]}));
        //}


    }

    //for other layers than the first
    else{//passes the whole input vector to every neuron

        if(LayerInputs.size() != getWeights()[0].size()){
            //[0] because it's a priori known that every sub vector in LayerWeights will be the same size. If that's not the case it needs to be placed in the for loop.
            throw std::invalid_argument("\nlayer::dsigmoid: dimension mismatch\n");
        }


        int count = 0;
        std::transform(tmp.begin(),tmp.end(),tmp.begin(), [&](flo) -> flo {flo temp = Neurons.at(count).dsigmoid(Neurons.at(count).activateFunc(LayerInputs)); ++count; return temp; });

        //for(int i = 0; i<Neurons.size(); ++i){
            //tmp[i] = Neurons[i].dsigmoid(Neurons[i].activateFunc(LayerInputs));
        //}


        //for(int i = 0; i < (int) Neurons.size(); ++i){
            //tmp[i] = Neurons[i].dsigmoid(Neurons[i].activateFunc(LayerInputs));
        //}
    }

    return tmp;
}

