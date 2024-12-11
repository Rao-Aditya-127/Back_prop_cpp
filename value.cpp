#include "value.h"

// Constructor implementation
value::value(string node_name, double data, vector<value*> _children, char _op)
    : node_name(node_name), data(data), _prev(_children), _op(_op), grad(0.0) {
    _backward = []() {}; // Default backward function does nothing
}

// Operator overloads
value* value::operator+(value &other) {
    value* out = new value(this->node_name + "+" + other.node_name, this->data + other.data, {this, &other}, '+');
    out->_backward = [this, &other, out]() {
        this->grad += out->grad;
        other.grad += out->grad;
    };
    return out;
}

value* value::operator+(double other) {
    value temp(to_string(other), other);
    value* out = new value(this->node_name + "+" + to_string(other), this->data + other, {this}, '+');
    out->_backward = [this, out]() {
        this->grad += out->grad;
    };
    return out;
}

value* value::operator*(value &other) {
    value* out = new value(this->node_name + "*" + other.node_name, this->data * other.data, {this, &other}, '*');
    out->_backward = [this, &other, out]() {
        this->grad += other.data * out->grad;
        other.grad += this->data * out->grad;
    };
    return out;            
}

value* value::operator*(double other) {
    value* out = new value(this->node_name + "*" + to_string(other), this->data * other, {this}, '*');
    out->_backward = [this, out, other]() {
        this->grad += other * out->grad;
    };
    return out;
}

value* value::operator-(double num) {
    double neg = num * -1;
    return *this + neg;
}

value* value::operator/(value &other) {
    auto inverse = this->val_pow(-1);
    auto ans = *this * (*inverse);
    return ans;
}

value* value::val_pow(double exponent) {
    double result = std::pow(this->data, exponent);
    value* out = new value("power ^ " + this->node_name, result, {this}, '^');
    out->_backward = [this, out, exponent]() {
        this->grad += exponent * std::pow(this->data, exponent - 1) * out->grad;
    };
    return out;
}

value* value::exp() {
    double x = this->data;
    double exp_result = std::exp(x);
    value* out = new value("exp", exp_result, {this}, 'e');
    out->_backward = [this, out]() {
        this->grad += out->data * out->grad; // Grad of exp(x) is exp(x) itself
    };
    return out;
}

value* value::tanh() {
    double t = std::tanh(this->data);
    value* out = new value("tanh(" + this->node_name + ")", t, {this}, 't');
    out->_backward = [this, out, t]() {
        this->grad += (1 - t * t) * out->grad;
    };
    return out;
}

// Backward propagation
void value::backward() {
    vector<value*> sorted;
    unordered_set<value*> visited;
    topo_sort(this, sorted, visited);
    reverse(sorted.begin(), sorted.end());

    for (auto &x : sorted) {
        x->_backward();
    }
}

// Print methods
void value::print() const {
    cout << "The value: " << this->data << "\n";
}

void value::printPrev() const {
    cout << "Previous values (" << this->_op << ") -> ";
    for (size_t i = 0; i < _prev.size(); ++i) {
        cout << _prev[i]->data;
        if (i < _prev.size() - 1) {
            cout << ", ";
        }
    }
    cout << endl;
}

// Topological sort for backpropagation
void value::topo_sort(value* head, vector<value*>& sorted, unordered_set<value*>& visited) {
    if (visited.find(head) != visited.end()) return;
    visited.insert(head);

    for (auto &c : head->_prev) {
        topo_sort(c, sorted, visited);
    }
    sorted.push_back(head);
}

// Print the computational graph
void value::printGraph(const value& node, const string& prefix, bool isLeft, unordered_set<const value*>& visited) {
    if (visited.find(&node) != visited.end()) {
        return;
    }

    visited.insert(&node);

    cout << prefix;
    cout << (isLeft ? "|-- " : "`-- ");
    cout << node.node_name << " (Data: " << node.data << ", Op: " << node._op
         << ", Grad: " << node.grad << ") " << "Address: " << &node << endl;

    string newPrefix = prefix + (isLeft ? "|   " : "    ");

    for (size_t i = 0; i < node._prev.size(); ++i) {
        printGraph(*node._prev[i], newPrefix, i < node._prev.size() - 1, visited);
    }
}
