// #include <numeric>                        // Standard library import for std::accumulate
#include <pybind11/pybind11.h>            // Pybind11 import to define Python bindings
#include <xtensor/xmath.hpp>              // xtensor import for the C++ universal functions
#include <xtensor/xpad.hpp>              // xtensor import for the C++ universal functions
#define FORCE_IMPORT_ARRAY
#include <xtensor/xtensor.hpp>
#include <Eigen/Dense>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-python/pyarray.hpp>     // Numpy bindings
#include <xtensor-python/pytensor.hpp>     // Numpy bindings
#include <xtensor/xadapt.hpp>
#include <xtensor/xreducer.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <iostream>
#include <typeinfo>
#include <xsimd/xsimd.hpp>
using namespace std;
using namespace xt::placeholders;

std::vector<double> sumVector(std::vector<double>, std::vector<double>);
xt::xarray<double> conv_3d(xt::pyarray<double>, xt::pyarray <double>);
std::vector<double> convolution(xt::xarray<double>, xt::xarray<double>, int);
xt::xarray<double> Z_convolution(xt::pyarray<double>, xt::pyarray <double>);
xt::xarray<double> dK_convolution(xt::pyarray<double>, xt::pyarray <double>);
xt::xarray<double> dX_convolution(xt::pyarray<double>, xt::pyarray <double>);
// xt::xarray<double> max_pooling()
xt::xarray<double> paddedM(xt::pyarray<double>, int);
PYBIND11_MODULE(CppModule, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";
    m.def("convolution", &convolution);
    m.def("Z_convolution", &Z_convolution);
    m.def("dX_convolution", &dX_convolution);
    // m.def("rotM", &rotM);
    m.def("conv_3d", &conv_3d);
}

int main(int argc, char const *argv[])
{

    return 0;
}

std::vector<double> sumVector(std::vector<double> a, std::vector<double> b){
    std::vector<double> res(a.size(), 0);
    for (int i = 0; i < a.size(); i++)
    {
        res[i] = a[i] + b[i];
    }
    return res;
}
std::vector<double> convolution(xt::xarray<double> X, xt::xarray<double> K, int conv_stride = 1)
{
    // int col = X.shape()[0];
    int X_nrows = X.shape()[0];
    int X_ncols = X.shape()[1];
    int K_nrows = K.shape()[0];
    int K_ncols = K.shape()[1];
    int avalrow = X_nrows - K_nrows + 1;
    int avalcol = X_ncols - K_ncols + 1;
    std::vector<double> res = {};
    for (int i = 0; i < avalrow; i += conv_stride) {
        for (int k = 0; k < avalcol; k += conv_stride) {
            res.push_back(xt::sum<double>(xt::view(X, xt::range(i, i + K_nrows), xt::range(k,k + K_ncols)) * K, xt::evaluation_strategy::immediate)());
        }
    }
    return res;
}
xt::xarray<double> Z_convolution(xt::pyarray<double>X, xt::pyarray <double>K){
    xt::xarray<int> X_shape = xt::adapt(X.shape());
    xt::xarray<int> K_shape = xt::adapt(K.shape());
    int kernel_size = K.shape()[3];
    xt::xarray<int> Z = xt::zeros<int>({X_shape[0], K_shape[0],X_shape[2] - kernel_size + 1, X_shape[3] - kernel_size + 1}); 
    std::vector<xt::xarray<double>> a = {};
    int count = 0;
    for (int i = 0; i < X_shape[0]; i++)
    {
        for (int j = 0; j < K_shape[0]; j++)
        {
            std::vector<double> sum((X_shape[2] - kernel_size + 1) * (X_shape[3] - kernel_size + 1), 0);
            for (int k = 0; k < X_shape[1]; k++)
            {
                sum = sumVector(sum, convolution(xt::view(X, i, k, xt::all(), xt::all()),xt::view(K, j, k, xt::all(), xt::all()), 1));
            }
            xt::view(Z, i, j, xt::all(), xt::all()) = xt::adapt(sum, {X_shape[2] - kernel_size + 1, X_shape[3] - kernel_size + 1});
        }
    }
    return Z;
}
xt::xarray<double> conv_3d(xt::pyarray<double> X, xt::pyarray <double> K) {
    xt::xarray<int> X_shape = xt::adapt(X.shape());
    xt::xarray<int> K_shape = xt::adapt(K.shape());
    xt::xarray<double> res = xt::zeros <double> ({K_shape[0], X_shape[0],X_shape[1] - K_shape[2] + 1, X_shape[2] - K_shape[2] + 1});
    for (int i = 0; i < K_shape[0]; i++)
    {
        for (int k = 0; k < X_shape[0]; k++)
        {
            xt::view(res, i, k, xt::all()) = xt::adapt(convolution(xt::view(X, k, xt::all(), xt::all()),xt::view(K, i, xt::all(), xt::all())),
            {X_shape[1] - K_shape[2] + 1, X_shape[2] - K_shape[2] + 1});
        }
        
    }
    return res;
}
xt::xarray<double> dK_convolution(xt::pyarray<double>& X, xt::pyarray <double>& dZ) {
    xt::xarray<int> X_shape = xt::adapt(X.shape());
    xt::xarray<int> dZ_shape = xt::adapt(dZ.shape());
    xt::xarray<double> dK = xt::zeros <float> ({dZ_shape[1], X_shape[1],X_shape[2] - dZ_shape[3] + 1, X_shape[3] - dZ_shape[3] + 1});
    for (int i = 0; i < X_shape[0]; i++)
    {
        dK += conv_3d(xt::view(X, i, xt::all(), xt::all(), xt::all()),xt::view(dZ, i, xt::all(), xt::all(), xt::all()));
    }
    return dK;
    
}
xt::xarray<double> paddedM(xt::pyarray<double> X, int n_pads = 1) {
    xt::xarray<double> res = xt::zeros<double> ({X.shape()[0] + n_pads * 2, X.shape()[1] + n_pads * 2});
    xt::view(res, xt::range(n_pads,  res.shape()[0] - n_pads), xt::range(n_pads, res.shape()[1] - n_pads)) = X;
    return res;
}
xt::xarray<double> dX_convolution(xt::pyarray<double> K, xt::pyarray <double> dZ) {
    int kernel_size = K.shape()[3];
    xt::xarray<int> K_shape = xt::adapt(K.shape());
    xt::xarray<int> dZ_shape = xt::adapt(dZ.shape());
    xt::xarray<double> dX = xt::zeros <double> ({dZ_shape[0], K_shape[1],dZ_shape[2] + kernel_size - 1, dZ_shape[3] + kernel_size - 1});
    for (int i = 0; i < dZ_shape[0]; i++)
    {
        for (int j = 0; j < K_shape[0]; j++)
        {
            for (int k = 0; k < K_shape[1]; k++)
            {
                xt::view(dX, i, k, xt::all(), xt::all()) = xt::adapt(convolution(
                    paddedM(xt::view(dZ, i, j, xt::all(), xt::all()), kernel_size - 1),
                    xt::rot90<2>(xt::view(K, j,k, xt::all(), xt::all())), 1
                ), {dZ_shape[2] + kernel_size - 1, dZ_shape[3] + kernel_size - 1});
            }
        }
    }
    return dX;
}
