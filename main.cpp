#include "NeuronLayer.h"

float rng(flo min, flo max){
    random_device rd; //Initializes random engine
    mt19937 gen(rd()); //Mersenne Twister 19937 generator, rng
    uniform_real_distribution<flo> dis(min, max); //uniform probability distribution
    return dis(gen);
}
int main() {

    //performance test of a simulated training session
    //random_device rd; //Initializes random engine
    //mt19937 gen(rd()); //Mersenne Twister 19937 generator, rng
    //uniform_real_distribution<float> dis(-1, 1); //uniform probability distribution

    
    int nneuron = 150;
    int ninputs = 728;
    layer tost(nneuron,ninputs);

    vector<vector<float*> > ww;
    ww.resize(nneuron);
    vector<float*> input;
    input.resize(nneuron);
    vector<float*> bb;
    bb.resize(nneuron);

    for(int k = 0; k < 10; ++k){
        
    vector<float> wo(ninputs);
    vector<vector<float> > w(nneuron,wo);
    vector<float> b(nneuron);
    vector<float> in(ninputs);
    vector<float*> tmp;
    tmp.resize(wo.size());
    vector<float*> tmp2;
    tmp2.resize(in.size());

    //convert the vectors of floats to vectors of pointers
    for( int i = 0; i < w.size(); ++i){
        for(int  j = 0; j < w[i].size(); ++j){
            w[i][j] = rng(-1,1);
            tmp[j] = &(w[i][j]);
        }
    ww[i] = tmp;
    in[i] = rng(-1,1);
    input[i] =  &in[i];
    b[i] = rng(-1,1);
    bb[i] = &b[i];
    }
    //Constructor2

    //update the parameters
    tost.setWeights(ww);
    tost.setBias(bb);
    *tost.getWeights()[0][0];
    //calculate output
    *tost.resultFunc(input,false)[0];
    //cout << "nrUpdated Weight(0,0): " << *tost.getWeights()[0][0] << endl;
    //cout << "nrOutput(0,0): " << *tost.resultFunc(input,false)[0] << endl;
    //cout << "nrOutput(0,0): " << *tost(input,false)[0] << endl;

    //cout << "dsigmoid(0,0); " << tost.dsigmoid(input,false)[0] << endl;
    //cout << k << endl;
    }
    return 0;
}
