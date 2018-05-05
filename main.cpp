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
    int ninputs = 150; //number of inputs per neuron
    
    //copy constructor and assignment operator test
    layer tost(nneuron,ninputs,true); //constructor2 for initialization
    layer tast = tost;
    layer tust = layer(tast);
    layer tist = layer(2,2);

    cout << tost.getWeights()[0][0] << endl;
    cout << tast.getWeights()[0][0] << endl;
    cout << tust.getWeights()[0][0] << endl;
    
    float x = 1;    
    float y = 0;    
    

    //performance test setup
    
    //create vectors for input values
    vector<fp> input; //simulated pixel data
    input.resize(ninputs);
        vector<flo> wo(ninputs);
        vector<vector<flo> > w(nneuron,wo);
        vector<flo> b(nneuron);
        vector<flo> in(ninputs);
        

    for(int k = 0; k < 100; ++k){

        //convert the vectors of flos to vectors of pointers
        //these algorithms replace the loops from previous version
        std::for_each(w.begin(),w.end(),[&](vector<flo> &subw){
                std::generate(subw.begin(),subw.end(),[&](){return rng(-1,1);});
        });

        std::generate(input.begin(),input.end(),[&](){return &x;});
        std::generate(b.begin(),b.end(),[&](){return rng(-1,1);});

        
        //update the parameters
        tost.setWeights(w); //update weights
        tost.setBias(b); //update bias
        tost.getWeights()[0][0]; //output weights
        
        //cout<< "weights " <<tost.getWeights()[0][0]<<endl;
        //cout<< "weights2 " <<tost.getWeights()[0][1]<<endl;
        //cout<< "weights3 " <<tost.getWeights()[1][0]<<endl;
        //cout<< "bias " <<tost.getBias()[0]<<endl;
        cout<< "result " <<*tost.resultFunc(input)[0] <<endl; //neuron output
        cout<< "dsigma " <<tost.dsigmoid(input)[0] <<endl; //dsigmoid output
 
    }

}
catch(const std::invalid_argument& argerror){
    cout << argerror.what();//outputs the erro msg
    return EXIT_FAILURE;
}



return 0;

}
