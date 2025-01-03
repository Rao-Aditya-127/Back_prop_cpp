#include "value.h" 
#include <random>
#include <ctime> 

class neuron{

    public:

    vector<value*> weights;
    value* bias = new value("b", 0.0);

    neuron(int nin) {  
        // Shared random number generator
        static std::random_device rd;
        static std::mt19937 gen(rd() ^ std::time(nullptr)); // Use shared generator
        std::uniform_real_distribution<> dis(-1.0, 1.0);   // Uniform distribution [-1, 1]

        // Initialize weights
        for (int i = 0; i < nin; ++i) {
            weights.emplace_back(new value("w" + std::to_string(i + 1), dis(gen)));
        }

        // Initialize bias
        bias->data = dis(gen);
    }

    value* operator()(vector<value> &inputs) {

        vector<value*> midNode;
        
        for(int i = 0 ; i < inputs.size() ; i++){
            midNode.push_back((*weights[i] * inputs[i]));
        }

        value* sum = bias;

        for(auto x : midNode){
            sum = *sum + *x;
        }

        sum = sum->tanh();

        return sum;
    }

    value* operator()(vector<value*> &inputs) {

        vector<value*> midNode;
        
        for(int i = 0 ; i < inputs.size() ; i++){
            midNode.push_back((*weights[i] * (*inputs[i])));
        }

        value* sum = bias;

        for(auto x : midNode){
            sum = *sum + *x;
        }

        sum = sum->tanh();

        return sum;
    }

    vector<value*> parameters(){
        weights.push_back(bias);
        return weights;
    }

};

class layer{
    public:

        vector<neuron> list;
        vector<value*> output;

        //nin is the number of input to each neuron that is previous layer or the size of input 
        //nout is number of neuron needed in that layer and also defines the out put of that layer hence named as nout 

        layer(int nin , int nout){
            for(int i = 0 ; i < nout ; i++){
                list.emplace_back(neuron(nin));
            }
        }

        vector<value*> operator()(vector<value> &inputs){
            output.clear();
            for(auto x : list){
                output.push_back(x(inputs));
            }

            return output;
        }

        vector<value*> operator()(vector<value*> &inputs){
            output.clear();
            for(auto x : list){
                output.push_back(x(inputs));
            }

            return output;
        }

        vector<value*> parameters(){

            vector<value*> param;

            for(auto x : list){
                auto temp = x.parameters();
                for(auto y : temp){
                    param.push_back(y);
                }
            }

            return param;
        }
};


class MLP{
    public:

        vector<layer> mlp;

        MLP(int nin , vector<int> nouts){

            nouts.insert(nouts.begin(), nin);

            for(int i = 0 ; i < nouts.size() - 1 ; i++){
                mlp.push_back(layer(nouts[i] , nouts[i+1]));
            }
        }

        vector<value*> operator()(vector<value> &inputs){
            auto input_val = mlp[0](inputs);

            for(int i = 1 ; i < mlp.size() ; i++){
              input_val = mlp[i](input_val);
            }

            return input_val;
        }

        vector<value*> parameters(){

            vector<value*> param;

            for(auto x : mlp){
                auto temp = x.parameters();
                for(auto y : temp){
                    param.push_back(y);
                }
            }

            return param;
        }
};

int main(){

    vector<vector<value>> inputs = {{{"v1" , 2} , {"v2" , 3} , {"v3" , -1}} , {{"v1" , 3} , {"v2" , -1} , {"v3" , 5}} , {{"v1" , 0.5} , {"v2" , 1} , {"v3" , 1}} , {{"v1" , 1} , {"v2" , 1} , {"v3" , -1}}};
    vector<double> output = {1 , -1 , -1 , 1}; 

    MLP multi(3 , {4 , 4 , 1}); // Multi-Layer Perceptron (MLP): 3 nodes in the input layer, two hidden layers with 4 nodes each, and 1 node in the output layer.

    for(int i = 0 ; i < 50  ; i++){ 
        
        vector<value*> params = multi.parameters(); // Retrieves the parameters (weights and bias) associated with each neuron in the MLP.

        vector<value*> ypred;
        
        for(int i = 0 ; i < inputs.size() ; i++){
            ypred.push_back(multi(inputs[i])[0]);  // Each input is passed to MLP and the output is stored in ypred vector *imp point* - in this case as output layer contain only one neuron we are intrested in first value in the vector that is returned as output 
        }
            
        vector<value*> square;

        for(int i = 0 ; i < output.size() ; i++){
            square.push_back((*ypred[i] - output[i])->val_pow(2)); // calculating the difference between predicted and expected output and squaring it 
        } 

        value* loss = square[0];

        for(int i = 1 ; i < square.size() ; i++){
            loss = *loss + *square[i]; // adding the squared difference
        }

        cout<<"loss = "<<loss->data<<endl;

        loss->grad = 1;
        loss->backward();  // calculating gradinent associated with each neuron learn more about this in value class

        for(auto x : params){  // change the parameter with small value depending upon the gradient 
            x->data = x->data + (-0.05 * x->grad);  //  here -0.05 is the learning rate 
            x->grad = 0; // always remember to clear out grad or they will accumulated
        }

    }

    // prediction after training 

    cout<<endl<<"predictions -"<<endl;

    for(int i = 0 ; i < inputs.size() ; i++){
        cout<<multi(inputs[i])[0]->data<<endl;
    }

    return 0;
}