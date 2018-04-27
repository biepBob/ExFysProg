#include "NeuronLayer.h"

flo rng(flo min, flo max){
    random_device rd; //Initializes random engine
    mt19937 gen(rd()); //Mersenne Twister 19937 generator, rng
    uniform_real_distribution<flo> dis(min, max); //uniform probability distribution
    return dis(gen);
}
int main() {

int nneuron = 150;
int ninputs = 728;
layer tost(nneuron,ninputs); //constructor2 for initialization
//performance test setup

//create vectors for input values
vector<vector<fp> > ww = tost.getWeights(); //weights
//ww.resize(nneuron);
vector<fp> input; //simulated pixel data
input.resize(nneuron);
vector<fp> bb = tost.getBias(); //bias
//bb.resize(nneuron);


//vector<fp> Bias = tost.getBias();
//cout << "b1: " << *Bias[0] << endl;
//*Bias[0] = 1;
//cout << "b2: " << *tost.getBias()[0] << endl;




for(int k = 0; k < 10; ++k){
//vectors of pointers 
//vector<flo> wo(ninputs);
//vector<vector<flo> > w(nneuron,wo);
//vector<flo> b(nneuron);
vector<flo> in(ninputs);
//vector<fp> tmp;
//tmp.resize(wo.size());
//vector<fp> tmp2;
    //tmp2.resize(in.size());

    //convert the vectors of flos to vectors of pointers
    for( int i = 0; i < (int) ww.size(); ++i){
        for(int  j = 0; j < (int) ww[i].size(); ++j){
            *ww[i][j] = rng(-1,1);
            //tmp[j] = &(w[i][j]);
        }
    //ww[i] = tmp;
    in[i] = rng(-1,1);
    input[i] =  &in[i];
    *bb[i] = rng(-1,1);
    //bb[i] = &b[i];
    } 
    //update the parameters
    tost.setWeights(ww); //update weights
    tost.setBias(bb); //update bias
    *tost.getWeights()[0][0]; //output weights
    *tost.resultFunc(input,false)[0]; //neuron output
    
    //optional cout
    //cout << "nrUpdated Weight(0,0): " << *tost.getWeights()[0][0] << endl;
    //cout << "nrOutput(0,0): " << *tost.resultFunc(input,false)[0] << endl;
    //cout << "nrOutput(0,0): " << *tost(input,false)[0] << endl;
    //cout << "dsigmoid(0,0); " << tost.dsigmoid(input,false)[0] << endl;
    }
    return 0;
}
