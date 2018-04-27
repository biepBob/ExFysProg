#include"Neuron.h"
#include <stdexcept>

class layer{
    //typedef float* fp;
    //typedef float flo;
public :
    layer(vector<vector<fp> > LayerWeights, vector<fp>  LayerBias); //Constructor1, weights provided, use this when parameters are loaded from a file
    layer(int nNeurons, int nInputs); //Constructor2, number of neurons and inputs provided, use this for initialization
    ~layer();
    layer(const layer &layer1); //Copy constructor
    layer& operator = (const layer &otherLayer);
    void setWeights (vector<vector<fp> > LayerWeights); //set-function for Weights across a layer
    void setBias(vector<fp> LayerBias); //set-function for bias across a layer

    vector<vector<fp> > getWeights(); //get-function to provide access to weights
    vector<fp> getBias(); //get-function to provide access to bias
    int getNumberofNeurons(); //get-function to provide acces to number of neurons

    vector<fp> resultFunc(vector<fp>  LayerInputs, bool FirstLayer); //calculates the output vector of the layer. Use FirstLayer true or false (or 1/0) to determine wheter it's the first layer or not. Depending on the value the inputs will be passed to the neuron in a different way.
    vector<fp>  operator()(vector<fp>  LayerInputs, bool FirstLayer){ return resultFunc(LayerInputs,FirstLayer);}
    vector<flo> dsigmoid(vector<fp>  LayerInputs); //calculates the derivative of sigmoid
protected :
    vector<neuron> Neurons;
    int NumberofNeurons;
    int NumberofInputs;
    bool FirstLayer;
};
