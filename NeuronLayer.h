#include"Neuron.h"
#include <stdexcept>

class layer{
    //typedef float* fp;
    //typedef float flo;
public :
    layer(vector<vector<flo> > LayerWeights, vector<flo>  LayerBias, bool firstLayer = false); //Constructor1, weights provided, use this when parameters are loaded from a file
    layer(int nNeurons, int nInputs, bool firstLayer = false); //Constructor2, number of neurons and inputs provided, use this for initialization
    ~layer();
    layer(const layer &layer1); //Copy constructor
    layer& operator = (const layer &otherLayer);
    void setWeights (vector<vector<flo> > LayerWeights); //set-function for Weights across a layer
    void setBias(vector<flo> LayerBias); //set-function for bias across a layer

    vector<vector<flo> > getWeights(); //get-function to provide access to weights
    vector<flo> getBias(); //get-function to provide access to bias
    int getNumberofNeurons(); //get-function to provide acces to number of neurons

    vector<flo> resultFunc(vector<fp>  LayerInputs); //calculates the output vector of the layer. Use FirstLayer true or false (or 1/0) to determine wheter it's the first layer or not. Depending on the value the inputs will be passed to the neuron in a different way.
    vector<flo>  operator()(vector<fp>  LayerInputs){ return resultFunc(LayerInputs);}
    vector<flo> dsigmoid(vector<fp>  LayerInputs); //calculates the derivative of sigmoid
protected :
    vector<neuron> Neurons;
    int NumberofNeurons;
    int NumberofInputs;
    bool FirstLayer;
};
