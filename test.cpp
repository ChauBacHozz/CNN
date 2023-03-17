#include <iostream>
#include <xtensor/xtensor.hpp>
using namespace std;
int main(int argc, char const *argv[])
{
    xt::xtensor<double, 2> a = {{1., 2.}, {3., 4.}};
    cout << a << endl; 
    return 0;
}
