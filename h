#include "itensor/all.h"
#include "itensor/util/print_macro.h"

#include <vector>
#include <cmath>
#include <complex>

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <fstream>
#include <sstream>

using namespace itensor;
using namespace std::chrono;

// Helper: Get memory usage in MB (Linux)
size_t get_memory_usage_MB()
{
    std::ifstream status_file("/proc/self/status");
    std::string line;
    size_t mem_kb = 0;

    while(std::getline(status_file, line))
    {
        if(line.rfind("VmRSS:", 0) == 0)
        {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> mem_kb;
            break;
        }
    }
    return mem_kb / 1024;
}

#define LOG_RESOURCE(tag, start_time) { \
    auto end = std::chrono::high_resolution_clock::now(); \
    double duration = std::chrono::duration<double>(end - start_time).count(); \
    size_t memory = get_memory_usage_MB(); \
    auto thread_id = std::this_thread::get_id(); \
    pid_t pid = getpid(); \
    std::cout << "[Node PID " << pid << " | Thread " << thread_id << "] " \
              << tag << " took " << duration << "s | Mem: " << memory << "MB" << std::endl; \
}

struct data
{
    std::string symbol;
    int index1;
    int index2;
    float alpha;
};

MPS apply_gates3(std::vector<std::tuple<int,int,int,double>> circuits, Qubit site_inds, int N, double cutoff){

    std::vector<MPS> return_vector;

    auto init = InitState(site_inds);
    for(auto n : range1(N))
    {
        init.set(n,"Up");
    }

    auto psi = MPS(init);

    std::complex<double> i(0.0, 1.0);
    const double pi = 3.14159265359;

    for (std::tuple<int,int,int,double> gate : circuits){
        
        auto sym = std::get<0>(gate);
        auto i1 = std::get<1>(gate);
        auto i2 = std::get<2>(gate);
        auto a = std::get<3>(gate);
        double theta = (pi*a)/2;

        if (sym ==0) {
            auto G = op(site_inds,"H",i1+1, {"alpha=",a});
            psi.position(i1+1);
            auto new_MPS = G*psi(i1+1);
            new_MPS.noPrime();
            psi.set(i1+1,new_MPS);

        } else if (sym == 1){
            auto G = op(site_inds,"Rx",i1+1, {"alpha=",a});
            psi.position(i1+1);
            auto new_MPS = G*psi(i1+1);
            new_MPS.noPrime();
            psi.set(i1+1,new_MPS);

        } else if (sym == 2){
            auto G = op(site_inds,"Rz",i1+1, {"alpha=",a});
            psi.position(i1+1);
            auto new_MPS = G* psi(i1+1);
            new_MPS.noPrime();
            psi.set(i1+1,new_MPS);

        } else if (sym == 3){
            psi.position(i1+1);
            auto opx1 = op(site_inds,"X",i1+1, {"alpha=",a});
            psi.position(i2+1);
            auto opx2 = op(site_inds,"X",i2+1, {"alpha=",a});
            auto G = expHermitian(opx2 * opx1, -i*theta);
            psi.position(i1+1);
            auto wf1 = psi(i1+1)*psi(i2+1);
            auto wf = G*wf1;
            wf.noPrime();

            auto start = std::chrono::high_resolution_clock::now();
            auto [U,S,V] = svd(wf,commonInds(wf, psi(i1+1)),{"Cutoff=",1E-10,"SVDMethod=","automatic"});
            LOG_RESOURCE("SVD XXPhase", start);

            psi.set(i1+1,U);
            psi.position(i2+1);
            psi.set(i2+1,S*V);

        } else if (sym == 4){
            auto op1 = op(site_inds,"Z",i1+1, {"alpha=",a});
            auto op2 = op(site_inds,"Z",i2+1, {"alpha=",a});
            auto G = expHermitian(op1 * op2,-i*theta);
            psi.position(i1+1);
            auto wf = psi(i1+1)*psi(i2+1);
            wf *= G;
            wf.noPrime();

            auto start = std::chrono::high_resolution_clock::now();
            auto [U,S,V] = svd(wf,inds(psi(i1+1)),{"Cutoff=",1E-10});
            LOG_RESOURCE("SVD ZZPhase", start);

            psi.set(i1+1,U);
            psi.position(i2+1);
            psi.set(i2+1,S*V);

        } else if (sym == 5){
            auto G = op(site_inds,"Z",i1+1, {"alpha=",a})*op(site_inds,"Z",i2+1, {"alpha=",a});
            G.set(1,1,2,2, 0);
            G.set(2,2,1,1, 0);
            G.set(1,2,2,1, 1);
            G.set(2,1,1,2, 1);

            psi.position(i1+1);
            auto wf1 = psi(i1+1)*psi(i2+1);
            auto wf = G*wf1;
            wf.noPrime();

            auto start = std::chrono::high_resolution_clock::now();
            auto [U,S,V] = svd(wf,inds(psi(i1+1)),{"Cutoff=",1E-10});
            LOG_RESOURCE("SVD Custom Gate", start);

            psi.set(i1+1,U);
            psi.position(i2+1);
            psi.set(i2+1,S*V);

        } else if (sym == 6){
            auto G = op(site_inds,"T",i1+1, {"alpha=",a});
            psi.position(i1+1);
            auto new_MPS = G* psi(i1+1);
            new_MPS.noPrime();
            psi.set(i1+1,new_MPS);

        } else if (sym == 7){
            auto G = op(site_inds,"Z",i1+1, {"alpha=",a})*op(site_inds,"Z",i2+1, {"alpha=",a});
            G.set(1,1,2,2, 1);
            G.set(2,2,1,1, 1);
            G.set(2,2,2,2, -1);

            psi.position(i1+1);
            auto wf1 = psi(i1+1)*psi(i2+1);
            auto wf = G*wf1;
            wf.noPrime();

            auto start = std::chrono::high_resolution_clock::now();
            auto [U,S,V] = svd(wf,inds(psi(i1+1)),{"Cutoff=",1E-10});
            LOG_RESOURCE("SVD CZ Gate", start);

            psi.set(i1+1,U);
            psi.position(i2+1);
            psi.set(i2+1,S*V);
        }
        else {
            std::cout << "Incorrect Gate" << std::endl;
        }
    }

    return psi;
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

int hello(){
    std::cout << "Hello" << std::endl;
    return 0;
}

int main(){
    hello();
    return 0;
}

template <typename T>
T add(T i, T j) {
    return i + j;
}

template <typename T>
std::vector<T> list_return(std::vector<T> vector1) {
    std::cout<<"Vector_Function"<<std::endl;
    std::vector<T> vector2;
    for (T i=0; i<vector1.size(); i++){
        vector2.push_back(vector1[i]);
    }
    return vector2;
}

template< typename T1, typename T2>
int tuple_return(std::vector<std::tuple<T1,T1,T1,T2>> vect_tup) {
    std::cout<<"Vec_Tup_Function"<<std::endl;
    for (auto i=0; i<vect_tup.size(); i++){
        std::cout << "["<< std::get<0>(vect_tup[i]) <<","<< std::get<1>(vect_tup[i]) <<","<< std::get<2>(vect_tup[i]) <<","<< std::get<3>(vect_tup[i]) <<"]"<< std::endl;
    }
    return 0;
}

template< typename T1, typename T2>
std::vector<std::vector<T2>> circuit_xyz_exp(std::vector<std::tuple<T1,T1,T1,T2>> tensor_vec, int no_sites) {
    
    auto tensor_sites = Qubit(no_sites);

    MPS tensor_mps = apply_gates3(tensor_vec, tensor_sites, no_sites, 1E-16);
    
    int chi = maxLinkDim(tensor_mps);
    std::vector<std::vector<T2>> return_vec;

    auto start_itensor = std::chrono::high_resolution_clock::now();

    for (int i=0; i<no_sites; i++)
    {
        std::vector<T2> xyz;
        tensor_mps.position(i+1);
        auto scalar_x = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("X_half",i+1)*tensor_mps.A(i+1))).real();
        auto scalar_y = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("Y_half",i+1)*tensor_mps.A(i+1))).real();
        auto scalar_z = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("Z_half",i+1)*tensor_mps.A(i+1))).real();
        xyz.push_back(scalar_x);
        xyz.push_back(scalar_y);
        xyz.push_back(scalar_z);

        return_vec.push_back(xyz);
    }

    auto end_itensor = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff_itensor = end_itensor - start_itensor;

    std::cout << "Total XYZ extraction time: " << diff_itensor.count() << " s\n";

    return return_vec;
}

PYBIND11_MODULE(helloitensor, m) {
    m.doc() = "pybind11 test";

    m.def("add", &add<float>, "A function that adds two numbers");
    m.def("hello", &hello, "Hello function");
    m.def("vec_return", &list_return<int>, "Return input list as a vector");
    m.def("tuple_return", &tuple_return<int,float>, "Print vector of tuples");
    m.def("tuple_return", &tuple_return<int,int>, "Print vector of tuples");
    m.def("tuple_return", &tuple_return<float,float>, "Return vector of tuples");

    m.def("circuit_xyz_exp", &circuit_xyz_exp<int,double>, "Function to extract single qubit expectation values from circuit. Returns list of num_qubit x,y,z exp values.");
}
