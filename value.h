#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <functional>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <string>

using namespace std;

class value {
public:
    string node_name;
    double data;
    vector<value*> _prev;
    char _op;
    double grad;
    function<void()> _backward;

    value(string node_name, double data, vector<value*> _children = {}, char _op = ' ');

    value* operator+(value &other);
    value* operator+(double other);
    value* operator*(value &other);
    value* operator*(double other);
    value* operator-(double num);
    value* operator/(value &other);
    value* val_pow(double exponent);
    value* exp();
    value* tanh();

    void backward();
    void print() const;
    void printPrev() const;
    void topo_sort(value* head, vector<value*>& sorted, unordered_set<value*>& visited);

    // For printing computational graph
    static void printGraph(const value& node, const string& prefix, bool isLeft, unordered_set<const value*>& visited);
};

#endif
