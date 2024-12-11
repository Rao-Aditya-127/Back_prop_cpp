#include "value.h"

int main() {
    value x1("x1", 2.0), x2("x2", 0.0), b("b", 6.8813735), w1("w1", -3.0), w2("w2", 1.0);
    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = *x1w1 + *x2w2;
    auto n = *x1w1x2w2 + b;

    auto x = *n * 2;
    auto e = (*x).exp();
    auto f = *e - 1;
    auto g = *e + 1;

    auto inverse = g->val_pow(-1);
    auto out = *f * *inverse;

    out->grad = 1.0;
    out->backward();

    unordered_set<const value*> visited;
    value::printGraph(*out, "", true, visited);

    // Additional operations...
}
