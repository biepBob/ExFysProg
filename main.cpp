#include "NeuronLayer.h"

int main() {

    float x = 1;
    float y = 1;

    //Constructor1
    layer test(2,2);
    cout << "rWeight(0,0): " << *test.getWeights()[0][0] << endl;
    test.setWeights(test.getWeights());
    cout << "rUpdated Weight(0,0): " << *test.getWeights()[0][0] << endl;
    cout << "rOutput(0,0): " << *test.resultFunc({{&x,&y},{&y,&x}})[0] << endl;
    //
    vector<vector<float> >w = {{0.4,0.6},{0.7,0.3}};
    vector<vector<float*> >ww = {{&w[0][0],&w[0][1]}, {&w[1][0],&w[1][1]}};
    vector<float> b = {0.5,0.6};
    vector<float*> bb = {&b[0],&b[1]};
    //Constructor2
    layer tost(ww,bb);
    cout << "nrWeight(0,0): " << *tost.getWeights()[0][0] << endl;
    tost.setWeights(tost.getWeights());
    cout << "nrUpdated Weight(0,0): " << *tost.getWeights()[0][0] << endl;
    cout << "nrOutput(0,0): " << *tost.resultFunc({{&x,&y},{&y,&x}})[0] << endl;


    
    return 0;
}
