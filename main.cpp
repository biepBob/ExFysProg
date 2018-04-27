#include "NeuronLayer.h"

flo rng(flo min, flo max){
    random_device rd; //Initializes random engine
    mt19937 gen(rd()); //Mersenne Twister 19937 generator, rng
    uniform_real_distribution<flo> dis(min, max); //uniform probability distribution
    return dis(gen);
}
int main() {

try{
    int nneuron = 150;
    int ninputs = 728;
    layer tost(nneuron,ninputs,false); //constructor2 for initialization
    layer tast = tost;
    layer tust = layer(tast);

    //copy constructor and assignment operator test
    cout << *tost.getWeights()[0][0] << endl;
    cout << *tast.getWeights()[0][0] << endl;
    cout << *tust.getWeights()[0][0] << endl;




    //performance test setup
    
    //create vectors for input values
    vector<vector<fp> > ww; //weights
    ww.resize(nneuron);
    vector<fp> input; //simulated pixel data
    input.resize(nneuron);
    vector<fp> bb; //bias
    bb.resize(nneuron);
    for(int k = 0; k < 10; ++k){
        
        //vectors of pointers 
        vector<flo> wo(ninputs);
        vector<vector<flo> > w(nneuron,wo);
        vector<flo> b(nneuron);
        vector<flo> in(ninputs);
        vector<fp> tmp;
        tmp.resize(wo.size());
        vector<fp> tmp2;
        tmp2.resize(in.size());

        //convert the vectors of flos to vectors of pointers
        for( int i = 0; i < (int) w.size(); ++i){
            for(int  j = 0; j < (int) w[i].size(); ++j){
                w[i][j] = rng(-1,1);
                tmp[j] = &(w[i][j]);
            }
        ww[i] = tmp;
        in[i] = rng(-1,1);
        input[i] =  &in[i];
        b[i] = rng(-1,1);
        bb[i] = &b[i];
        }

        //update the parameters
        tost.setWeights(ww); //update weights
        tost.setBias(bb); //update bias
        *tost.getWeights()[0][0]; //output weights
        input.resize(nneuron-1); //this will shorten the input vector to test if the exception handling in resultFunc works, remove -1 to make it work
        
        *tost.resultFunc(input)[0]; //neuron output
    
        //optional cout
        //cout << "nrUpdated Weight(0,0): " << *tost.getWeights()[0][0] << endl;
        //cout << "nrOutput(0,0): " << *tost.resultFunc(input,false)[0] << endl;
        //cout << "nrOutput(0,0): " << *tost(input,false)[0] << endl;
        //cout << "dsigmoid(0,0); " << tost.dsigmoid(input,false)[0] << endl;
    }
}
catch(const std::invalid_argument& argerror){
    cout << argerror.what();
    return EXIT_FAILURE;
}



return 0;

}
