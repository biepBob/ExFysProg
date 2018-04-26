#include"Neuron.h"


class layer{
    //typedef float* fp;
    //typedef float flo;
public :
    layer(vector<vector<fp> > LayerWeights, vector<fp>  LayerBias); //Constructor1, weights provided, use this for updating
    layer(int nNeurons, int nInputs); //Constructor2, number of neurons and inputs provided, use this for initialization
    ~layer();
    void setWeights (vector<vector<fp> > LayerWeights); //set-function for Weights across a layer
    void setBias(vector<fp> LayerBias); //set-function for bias across a layer

    vector<vector<fp> > getWeights(); //get-function to provide access to weights
    vector<fp> getBias(); //get-function to provide access to bias
    int getNumberofNeurons(); //get-function to provide acces to number of neurons

    vector<fp> resultFunc(vector<vector<fp> > LayerInputs); //calculates the output
    void operator()(vector<vector<fp> > LayerInputs){ resultFunc(LayerInputs);}
    vector<flo> dsigmoid(vector<flo>  LayerInputs); //calculates the derivative of sigmoid
protected :
    vector<neuron> Neurons;
    int NumberofNeurons;
};
