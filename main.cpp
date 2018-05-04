#include "NeuronLayer.h"

flo rng(flo min, flo max){
    random_device rd; //Initializes random engine
    mt19937 gen(rd()); //Mersenne Twister 19937 generator, rng
    uniform_real_distribution<flo> dis(min, max); //uniform probability distribution
    return dis(gen);
}
int main() {

try{
    int nneuron = 150; //number of neurons
    int ninputs = 110; //number of inputs per neuron
    layer tost(nneuron,ninputs,false); //constructor2 for initialization
    layer tast = tost;
    layer tust = layer(tast);

    layer tist = layer(2,2);

    //copy constructor and assignment operator test
    cout << tost.getWeights()[0][0] << endl;
    cout << tast.getWeights()[0][0] << endl;
    cout << tust.getWeights()[0][0] << endl;



    //performance test setup
    
    //create vectors for input values
    //vector<vector<fp> > ww; //weights
    //ww.resize(nneuron);
    vector<fp> input; //simulated pixel data
    input.resize(nneuron);
        vector<flo> wo(ninputs);
        vector<vector<flo> > w(nneuron,wo);
        vector<flo> b(nneuron);
        vector<flo> in(ninputs);
        
        

    for(int k = 0; k < 10; ++k){

        //convert the vectors of flos to vectors of pointers
        for( int i = 0; i < (int) w.size(); ++i){

            for(int  j = 0; j < (int) w[i].size(); ++j){

                w[i][j] = rng(-1,1);
                //tmp[j] = &(w[i][j]);
                in[j] = rng(-1,1);

                input[j] =  &in[j];
            }
            
        b[i] = rng(-1,1);
        }

        //update the parameters
        tost.setWeights(w); //update weights
        tost.setBias(b); //update bias
        tost.getWeights()[0][0]; //output weights
        //input.resize(nneuron); //this will shorten the input vector to test if the exception handling in resultFunc works, remove -1 to make it work
        
        cout<< tost.resultFunc(input)[0] <<endl; //neuron output
 
    }

}
catch(const std::invalid_argument& argerror){
    cout << argerror.what();
    return EXIT_FAILURE;
}



return 0;

}
