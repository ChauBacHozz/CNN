// #include <numeric>                        // Standard library import for std::accumulate
#include <pybind11/pybind11.h>            // Pybind11 import to define Python bindings
#include <xtensor/xmath.hpp>              // xtensor import for the C++ universal functions
#include <xtensor/xpad.hpp>              // xtensor import for the C++ universal functions
#define FORCE_IMPORT_ARRAY
#include <xtensor/xtensor.hpp>
#include <xtensor/xsort.hpp>
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
xt::xarray<double> max_pooling(xt::pyarray<double>, int);
xt::xarray<double> maxpool_de(xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, int);
xt::xarray<double> avg_pooling(xt::pyarray<double>, int);
xt::xarray<double> avgpool_de(xt::xarray<double>, xt::xarray<double>, int);
xt::xarray<double> pool_3d(xt::pyarray<double>, int, string);
xt::xarray<double> max_pool_de_3d(xt::pyarray<double>,xt::pyarray<double>,xt::pyarray<double>, int, string);
xt::xarray<double> paddedM(xt::pyarray<double>, int);
PYBIND11_MODULE(CppModule, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";
    m.def("convolution", &convolution);
    m.def("Z_convolution", &Z_convolution);
    m.def("dX_convolution", &dX_convolution);
    m.def("max_pooling", &max_pooling);
    m.def("avg_pooling", &avg_pooling);
    m.def("maxpool_de", &maxpool_de);
    m.def("avgpool_de", &avgpool_de);
    m.def("conv_3d", &conv_3d);
}

int main(int argc, char const *argv[])
{
    xt::xarray<double> a1({3,3}, 2.5);
    a1.fill(5);
    cout << a1 << endl;
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
xt::xarray<double> max_pooling(xt::pyarray<double> input, int pool_size = 2) {
    int pool_stride = pool_size;
    xt::xarray<int> input_shape = xt::adapt(input.shape());
    xt::xarray<double> res = xt::zeros<double> ({2 , input_shape[0] / pool_size, input_shape[1] / pool_size});
    for (int i = 0; i < input.shape()[0] - pool_size + 1; i+= pool_size)
    {
        for (int k = 0; k < input.shape()[1] - pool_size + 1; k+= pool_size)
        {
            res(0, i / pool_size, k / pool_size) = xt::argmax(xt::view(input, xt::range(i, i + pool_size), xt::range(k, k + pool_size)))();
            res(1, i / pool_size, k / pool_size) = xt::amax(xt::view(input, xt::range(i, i + pool_size), xt::range(k, k + pool_size)))();
        }
        
    }
    return res;
}
xt::xarray<double> avg_pooling(xt::pyarray<double> input, int pool_size = 2) {
    int pool_stride = pool_size;
    xt::xarray<double> res = xt::zeros<double> ({input.shape()[0] / pool_size, input.shape()[1] / pool_size});
    for (int i = 0; i < input.shape()[0] - pool_size + 1; i+= pool_size)
    {
        for (int k = 0; k < input.shape()[1] - pool_size + 1; k+= pool_size)
        {
            res(i / pool_size, k / pool_size) = xt::mean(xt::view(input, xt::range(i, i + pool_size), xt::range(k, k + pool_size)))();
        }
        
    }
    return res;
}

xt::xarray<double> pool_3d(xt::pyarray<double> input, int pool_size = 2, string pool_type = "max") {
    xt::xarray<double> res = xt::zeros<double> ({input.shape()[0], (input.shape()[1] - pool_size) / pool_size + 1, (input.shape()[2] - pool_size) / pool_size + 1});
    if (pool_type == "avg") {
        for (int i = 0; i < input.shape()[0]; i++)
        {
            xt::view(res, i, xt::all(), xt::all()) = avg_pooling(xt::view(input, i, xt::all(), xt::all()), pool_size);
        }
    } else {
        for (int i = 0; i < input.shape()[0]; i++)
        {
            xt::view(res, i, xt::all(), xt::all()) = xt::view(max_pooling(xt::view(input, i, xt::all(), xt::all()), pool_size), 1, xt::all(), xt::all());
        }
    }
    return res;
}
xt::xarray<double> maxpool_de(xt::xarray<double> Z, xt::xarray<double> dP, xt::xarray<double> pos,int pool_size = 2) {
    int n_cols = Z.shape()[1];  
    xt::xarray<double> dZ = xt::zeros_like(Z);
    for (int i = 0; i < dP.shape()[0]; i++) {
        for (int k = 0; k < dP.shape()[1]; k++) {
            // cout << (i * pool_size + int(pos(i, k) / 2)) * n_cols + k * 2 + int(pos(i, k) / 2) << endl;
            dZ((i * pool_size + int(pos(i, k)) / 2) * n_cols + k * 2 + int(pos(i, k)) % 2) = dP(i, k);
        }
    }
    return dZ;
}

xt::xarray<double> avgpool_de(xt::xarray<double> Z, xt::xarray<double> dP, int pool_size = 2) {
    xt::xarray<double> dZ = xt::zeros_like(Z);
    for (int i = 0; i < Z.shape()[0] - pool_size + 1; i+=pool_size)
    {
        for (int k = 0; k < Z.shape()[1] - pool_size + 1; k++)
        {
            xt::view(dZ, xt::range(i, i + pool_size), xt::range(k, k + pool_size))  = dP(i / pool_size, k / pool_size) / (pool_size * pool_size);
        }
        
    }
    return dZ;
}

xt::xarray<double> max_pool_de_3d(xt::pyarray<double> Z,xt::pyarray<double> dP, xt::pyarray<double> pos,int pool_size, string pool_type = "max") {
    // Z AND dP IS 3D ARRAY
    xt::xarray<double> dZ = xt::zeros_like(Z);
    for (int i = 0; i < Z.shape()[0]; i++)
    {
        xt::view(dZ, i, xt::all(), xt::all()) = maxpool_de(
            xt::view(Z, i, xt::all(), xt::all()),
            xt::view(dP, i, xt::all(), xt::all()),
            xt::view(pos, i, xt::all(), xt::all()),
            pool_size
        );
    }
    return dZ;
}




