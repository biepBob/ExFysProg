#include "NeuronLayer.h"

flo rng(flo min, flo max){
    random_device rd; //Initializes random engine
    mt19937 gen(rd()); //Mersenne Twister 19937 generator, rng
    uniform_real_distribution<flo> dis(min, max); //uniform probability distribution
    return dis(gen);
}
int main() {



try{
    int nneuron = 10; //number of neurons
    int ninputs = 15; //number of inputs per neuron
    
    //copy constructor and assignment operator test
    layer tost(nneuron,ninputs,false); //constructor2 for initialization
    layer tast = tost;
    layer tust = layer(tast);
    layer tist = layer(2,2);

    cout << tost.getWeights()[0][0] << endl;
    cout << tast.getWeights()[0][0] << endl;
    cout << tust.getWeights()[0][0] << endl;



    //performance test setup
    
    //create vectors for input values
    vector<fp> input; //simulated pixel data
    input.resize(ninputs);
        vector<flo> wo(ninputs);
        vector<vector<flo> > w(nneuron,wo);
        vector<flo> b(nneuron);
        vector<flo> in(ninputs);
        
    float x = 1;    

    for(int k = 0; k < 1; ++k){

        //convert the vectors of flos to vectors of pointers
        for( int i = 0; i < (int) w.size(); ++i){
            for(int  j = 0; j < (int) w[i].size(); ++j){
                w[i][j] = rng(-1,1);
                //in[j] = &x; //rng(-1,1); //if you want to randomize the input values though it induces a performance hit

                input[j] =  &x;//in[j];
            }
        b[i] = rng(-1,1);
        }

        //update the parameters
        tost.setWeights(w); //update weights
        tost.setBias(b); //update bias
        tost.getWeights()[0][0]; //output weights
        
        cout<< tost.resultFunc(input)[0] <<endl; //neuron output
        cout<< tost.dsigmoid(input)[0] <<endl; //dsigmoid output
 
    }

}
catch(const std::invalid_argument& argerror){
    cout << argerror.what();//outputs the erro msg
    return EXIT_FAILURE;
}



return 0;

}
