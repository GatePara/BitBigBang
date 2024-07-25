#include "hnswlib/hnswlib.h"
#include <thread>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <algorithm>
#include "io.h"
#include <omp.h>
#include <mutex>
#include <unordered_set>
#include <cfloat>

using namespace std;
using namespace std::chrono;

int BASE = 1000000, QUERY = 10000;
const int DIM = 100, K = 100;

// vector with a mutex pushback
template <typename T>
class ThreadSafeVector
{
    std::vector<T> v;
    std::mutex m;

public:
    void push_back(T a)
    {
        m.lock();
        v.push_back(a);
        m.unlock();
    }
    size_t size()
    {
        m.lock();
        size_t s = v.size();
        m.unlock();
        return s;
    }
    // access without lock
    T operator[](size_t i)
    {
        return v[i];
    }
};

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
{
    if (numThreads <= 0)
    {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1)
    {
        for (size_t id = start; id < end; id++)
        {
            fn(id, 0);
        }
    }
    else
    {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId)
        {
            threads.push_back(std::thread([&, threadId]
                                          {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                } }));
        }
        for (auto &thread : threads)
        {
            thread.join();
        }
        if (lastException)
        {
            std::rethrow_exception(lastException);
        }
    }
}

class Picks2 : public hnswlib::BaseFilterFunctor
{
    std::vector<float> &a2;
    std::vector<float> &f2;

public:
    Picks2(std::vector<float> &a2, std::vector<float> &f2) : a2(a2), f2(f2)
    {
    }
    bool operator()(hnswlib::labeltype label_id, size_t q_id)
    {
        return a2[label_id] >= f2[2 * q_id] && a2[label_id] <= f2[2 * q_id + 1];
    }
};

int main()
{
    string filepath = "../dummy-data.bin";
    string filepath_q = "../dummy-queries.bin";
    int M = 32;                // Tightly connected with internal dimensionality of the data
                               // strongly affects the memory consumption
    int ef_construction = 170; // Controls index search speed/build speed tradeoff
    int num_threads = 32;      // Number of threads for operations with index
    // print info
    int ef1 = 2300;
    int ef2 = 2300;
    int ef3 = 1500;
    int ef4 = 1500;
    int M2 = 32;
    int ef_construction2 = 170;
    int bt_q3 = 360000;
    int bt_q4 = 200000;
    int exp_thr1=1100,exp_thr2=1100,exp_thr3=1900,exp_thr4=1900;
    cout << "SQ16 With New AVX512" << endl;
    omp_set_num_threads(32);
    auto s1 = high_resolution_clock::now();

    ifstream file(filepath, ios::binary | ios::ate);
    if (!file.is_open())
    {
        cerr << "Failed to open file: " << filepath << endl;
        return -1;
    }

    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        cerr << "Failed to read file: " << filepath << endl;
        return -1;
    }
    file.close();

    int vectorNum;
    memcpy(&vectorNum, buffer.data(), sizeof(int));
    cout << "vector_num=" << vectorNum << endl;

    int dim = (size / 4 - 1) / vectorNum;
    int real_dim = dim - 2;
    vector<float> base((float *)(buffer.data() + sizeof(int)), (float *)(buffer.data() + size));
    vector<char>().swap(buffer);
    cout << "vector_shape=[" << vectorNum << ", " << real_dim << "]" << endl;

    vector<float> attribute1(vectorNum), attribute2(vectorNum);
    vector<double> attribute3(vectorNum);
    vector<float> vectorBase(vectorNum * real_dim);
    vector<unsigned> id_sorted_by_time(vectorNum);
    vector<unsigned> id_sorted_by_all(vectorNum);

    std::iota(id_sorted_by_time.begin(), id_sorted_by_time.end(), 0);
    std::iota(id_sorted_by_all.begin(), id_sorted_by_all.end(), 0);
    unsigned a1_count[10000] = {0};
#pragma omp parallel for reduction(+ : a1_count[ : 10000])
    for (int i = 0; i < vectorNum; ++i)
    {
        attribute1[i] = base[i * 102];
        a1_count[(int)attribute1[i]]++;
        attribute2[i] = base[i * 102 + 1];
        attribute3[i] = double(attribute1[i]) + double(attribute2[i]);
    }
    auto max_a1_count = std::max_element(a1_count, a1_count+10000);
    auto maxnum_label = max_a1_count-a1_count;
    vector<float> sorted_by_time = attribute2;
    vector<double> sorted_by_all = attribute3;
    std::sort(sorted_by_time.begin(), sorted_by_time.end());
    std::sort(sorted_by_all.begin(), sorted_by_all.end());
    std::sort(id_sorted_by_time.begin(), id_sorted_by_time.end(),
              [&attribute2](float i1, float i2)
              {
                  return attribute2[i1] < attribute2[i2];
              });
    std::sort(id_sorted_by_all.begin(), id_sorted_by_all.end(),
              [&attribute3](float i1, float i2)
              {
                  return attribute3[i1] < attribute3[i2];
              });

#pragma omp parallel for
    for (int i = 0; i < vectorNum; ++i)
    {
        std::copy(base.data() + i * 102 + 2, base.data() + (i + 1) * 102, vectorBase.data() + i * 100);
    }
    vector<float>().swap(base);

    float max_val = FLT_MIN;
    float min_val = FLT_MAX;

    for (int i = 0; i < vectorNum; i++)
    {
        for (int j = 0; j < real_dim; j++)
        {
            if (vectorBase[i * real_dim + j] > max_val)
            {
                max_val = vectorBase[i * real_dim + j];
            }
            if (vectorBase[i * real_dim + j] < min_val)
            {
                min_val = vectorBase[i * real_dim + j];
            }
        }
    }

    vector<unsigned short> vectorBaseSQ16(vectorNum * real_dim);
    int len = 65536;
#pragma omp parallel for
    for (int i = 0; i < vectorNum; i++)
    {
        for (int j = 0; j < real_dim; j++)
        {
            vectorBaseSQ16[i * real_dim + j] = static_cast<unsigned short>(static_cast<int>((static_cast<float>(vectorBase[i * real_dim + j]) - min_val) / (max_val - min_val) * (len - 1)));
        }
    }

    vector<unsigned char> vectorBaseSQ8(vectorNum * real_dim);
#pragma omp parallel for
    for (int i = 0; i < vectorNum; i++)
    {
        for (int j = 0; j < real_dim; j++)
        {
            vectorBaseSQ8[i * real_dim + j] = static_cast<unsigned char>(static_cast<int>((static_cast<float>(vectorBase[i * real_dim + j]) - min_val) / (max_val - min_val) * 255));
        }
    }

    vector<float>().swap(vectorBase);

    vector<unsigned short> BaseSortByLabelSQ16(vectorNum * real_dim);
    vector<unsigned short> BaseSortByTimeSQ16(vectorNum * real_dim);
    vector<unsigned short> BaseSortByALLSQ16(vectorNum * real_dim);
#pragma omp parallel for
    for (int i = 0; i < vectorNum; i++)
    {
        std::copy(vectorBaseSQ16.data() + id_sorted_by_time[i] * 100, vectorBaseSQ16.data() + (id_sorted_by_time[i] + 1) * 100, BaseSortByTimeSQ16.data() + i * 100);
    }
#pragma omp parallel for
    for (int i = 0; i < vectorNum; i++)
    {
        std::copy(vectorBaseSQ16.data() + id_sorted_by_all[i] * 100, vectorBaseSQ16.data() + (id_sorted_by_all[i] + 1) * 100, BaseSortByALLSQ16.data() + i * 100);
    }

    auto max_it = std::max_element(attribute1.begin(), attribute1.end());
    int a1_max = *max_it;
    unsigned a1_count_count[10000] = {0};
    vector<vector<int>> base_id_by_a1(a1_max + 1);
    for (size_t i = 0; i < a1_max + 1; i++)
    {
        base_id_by_a1[i].resize(a1_count[i]);
    }

    for (size_t i = 0; i < vectorNum; i++)
    {
        int type = attribute1[i];
        base_id_by_a1[type][a1_count_count[type]] = i;
        a1_count_count[type]++;
    }
    vector<int> base_id_by_a1_1d;
    for (auto &row : base_id_by_a1)
    {
        for (auto id : row)
        {
            base_id_by_a1_1d.push_back(id);
        }
    }
    vector<vector<int>>().swap(base_id_by_a1);
#pragma omp parallel for
    for (int i = 0; i < vectorNum; i++)
    {
        std::copy(vectorBaseSQ16.data() + base_id_by_a1_1d[i] * 100, vectorBaseSQ16.data() + (base_id_by_a1_1d[i] + 1) * 100, BaseSortByLabelSQ16.data() + i * 100);
    }

    vector<unsigned char> BaseSortByLabelSQ8(vectorNum * real_dim);
#pragma omp parallel for
    for (int i = 0; i < vectorNum; i++)
    {
        std::copy(vectorBaseSQ8.data() + base_id_by_a1_1d[i] * 100, vectorBaseSQ8.data() + (base_id_by_a1_1d[i] + 1) * 100, BaseSortByLabelSQ8.data() + i * 100);
    }

    vector<int> base_pointer_by_a1;
    for (int i = 0; i < 10000; i++)
    {
        int sum = 0;
        for (int j = 0; j < i; j++)
        {
            sum += a1_count_count[j];
        }
        base_pointer_by_a1.push_back(sum);
    }
    // timer
    auto s2 = high_resolution_clock::now();
    auto d1 = duration_cast<seconds>(s2 - s1);
    cout << "---Read base " << d1.count() << " seconds ---" << endl;

    ifstream file_q(filepath_q, ios::binary | ios::ate);
    if (!file_q.is_open())
    {
        cerr << "Failed to open file: " << filepath_q << endl;
        return -1;
    }

    streamsize size_q = file_q.tellg();
    file_q.seekg(0, ios::beg);

    vector<char> buffer_q(size_q);
    if (!file_q.read(buffer_q.data(), size_q))
    {
        cerr << "Failed to read file: " << filepath_q << endl;
        return -1;
    }
    file_q.close();

    int queryNum;
    memcpy(&queryNum, buffer_q.data(), sizeof(int));

    cout << "queryNum=" << queryNum << endl;

    int dim_q = (size_q / 4 - 1) / queryNum;
    int real_dim_q = dim_q - 4;

    vector<float> base_q((float *)(buffer_q.data() + sizeof(int)), (float *)(buffer_q.data() + size_q));
    vector<char>().swap(buffer_q);
    cout << "query_shape=[" << queryNum << ", " << real_dim_q << "]" << endl;

    vector<int> query_type(queryNum);
    vector<float> filter1(queryNum), filter2(queryNum * 2);
    vector<double> filter3(queryNum * 2);
    vector<float> vectorQuery(queryNum * real_dim_q);
#pragma omp parallel for
    for (int i = 0; i < queryNum; ++i)
    {
        query_type[i] = base_q[i * 104];
        filter1[i] = base_q[i * 104 + 1];
        filter2[2 * i] = base_q[i * 104 + 2];
        filter2[2 * i + 1] = base_q[i * 104 + 3];
        filter3[2 * i] = double(filter2[2 * i]) + double(filter1[i]);
        filter3[2 * i + 1] = double(filter2[2 * i + 1]) + double(filter1[i]);
    }
#pragma omp parallel for
    for (int i = 0; i < queryNum; ++i)
    {
        std::copy(base_q.data() + i * 104 + 4, base_q.data() + (i + 1) * 104, vectorQuery.data() + i * 100);
    }
    vector<float>().swap(base_q);

    vector<unsigned short> vectorQuerySQ16(queryNum * real_dim_q);
#pragma omp parallel for
    for (int i = 0; i < queryNum; i++)
    {
        for (int j = 0; j < real_dim_q; j++)
        {
            if (vectorQuery[i * real_dim_q + j] > max_val)
            {
                vectorQuerySQ16[i * real_dim_q + j] = len - 1;
            }
            else if (vectorQuery[i * real_dim_q + j] < min_val)
            {
                vectorQuerySQ16[i * real_dim_q + j] = 0;
            }
            else
            {
                vectorQuerySQ16[i * real_dim_q + j] = static_cast<unsigned short>(static_cast<int>((static_cast<float>(vectorQuery[i * real_dim_q + j]) - min_val) / (max_val - min_val) * (len - 1)));
            }
        }
    }
    vector<float>().swap(vectorQuery);

    ThreadSafeVector<int> q1tsv, q2tsv, q3tsv, q4tsv;

#pragma omp parallel for
    for (int i = 0; i < queryNum; ++i)
    {
        if (query_type[i] == 0)
        {
            q1tsv.push_back(i);
        }
        else if (query_type[i] == 1)
        {
            q2tsv.push_back(i);
        }
        else if (query_type[i] == 2)
        {
            q3tsv.push_back(i);
        }
        else if (query_type[i] == 3)
        {
            q4tsv.push_back(i);
        }
    }

    // timer
    auto s3 = high_resolution_clock::now();
    auto d2 = duration_cast<seconds>(s3 - s2);
    cout << "---Read Query " << d2.count() << " seconds ---" << endl;

    BASE = vectorNum;
    QUERY = queryNum;
    cout << "BASE: " << BASE << " QUERY: " << QUERY << endl;

    cout << "Data read successfully\n";

    cout << "M: " << M << " ef_construction: " << ef_construction << " num_threads: " << num_threads << endl;
    cout << "M_part: " << M2 << " ef_construction_part: " << ef_construction2 << endl;
    cout << "ef1: " << ef1 << " ef2: " << ef2 << " ef3: " << ef3 << " ef4: " << ef4 << endl;
    cout << "btq3: " << bt_q3 << " btq4: " << bt_q4 << endl;
    cout << "thr1: " << exp_thr1 << " thr2: " << exp_thr2 << " thr3: " << exp_thr3 << " thr4: " << exp_thr4 << endl;

    // Initing index
    hnswlib::L2SpaceI spaceSQ8(DIM);
    hnswlib::L2SpaceSQ16 space(DIM);
    auto fstdistfunc_ = space.get_dist_func();
    auto dist_func_param_ = space.get_dist_func_param();
    int hnsw_num = a1_max + 1;
    Picks2 picks2(attribute2, filter2);
    std::vector<hnswlib::labeltype> neighbors(queryNum * K);

    hnswlib::HierarchicalNSW<float> *hnsw_array[hnsw_num];
    for (size_t i = 0; i < hnsw_num; i++)
    {
        unsigned char *dataOffsetSQ8 = BaseSortByLabelSQ8.data() + DIM * base_pointer_by_a1[i];
        int *idOffset = base_id_by_a1_1d.data() + base_pointer_by_a1[i];
        hnsw_array[i] = new hnswlib::HierarchicalNSW<float>(&spaceSQ8, a1_count[i], M2, ef_construction2);
        ParallelFor(0, a1_count[i], num_threads, [&](size_t row, size_t threadId)
                    { hnsw_array[i]->addPoint((void *)(dataOffsetSQ8 + DIM * row), *(idOffset + row)); });
    }
    vector<unsigned char>().swap(BaseSortByLabelSQ8);
    // timer
    auto s4 = high_resolution_clock::now();
    auto d3 = duration_cast<seconds>(s4 - s3);
    cout << "---Build Index Partion " << d3.count() << " seconds ---" << endl;

    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&spaceSQ8, vectorNum, M, ef_construction);
    alg_hnsw->CopyHNSW(hnsw_array[maxnum_label]);
    for (size_t i = 0; i < hnsw_num; i++)
    {
        hnsw_array[i]->reSaveNeighbors();
    }
    ParallelFor(0, vectorNum, num_threads, [&](size_t row, size_t threadId)
                { 
                    
                    if(attribute1[row]!=maxnum_label)
                    {
                        alg_hnsw->addPoint((void *)(vectorBaseSQ8.data() + DIM * row), row); 
                    }
                });
    alg_hnsw->reSaveNeighbors();
    vector<unsigned char>().swap(vectorBaseSQ8);
    // timer
    auto s5 = high_resolution_clock::now();
    auto d4 = duration_cast<seconds>(s5 - s4);
    cout << "---Build Index Full " << d4.count() << " seconds ---" << endl;

    for (size_t i = 0; i < hnsw_num; i++)
    {
        unsigned short *dataOffsetSQ16 = BaseSortByLabelSQ16.data() + DIM * base_pointer_by_a1[i];
        int *idOffset = base_id_by_a1_1d.data() + base_pointer_by_a1[i];
        hnsw_array[i]->reSetSpace(&space);
        ParallelFor(0, a1_count[i], num_threads, [&](size_t row, size_t threadId)
                    { hnsw_array[i]->updateDatapoint((void *)(dataOffsetSQ16 + DIM * row), *(idOffset + row)); });
        hnsw_array[i]->freeNeighbors();
    }

    alg_hnsw->reSetSpace(&space);
    ParallelFor(0, vectorNum, num_threads, [&](size_t row, size_t threadId)
                { alg_hnsw->updateDatapoint((void *)(vectorBaseSQ16.data() + DIM * row), row); });
    alg_hnsw->freeNeighbors();
    auto s_swap = high_resolution_clock::now();
    auto d_swap = duration_cast<seconds>(s_swap - s5);
    cout << "---Swap Dataset " << d_swap.count() << " seconds ---" << endl;

    // Search
    // q2__________________________________________________________________
    for (size_t i = 0; i < hnsw_num; i++)
    {
        hnsw_array[i]->setEf(ef2);
    }
    ParallelFor(0, q2tsv.size(), num_threads, [&](size_t row, size_t threadId)
                {
                    int id = q2tsv[row];
                    int hnsw_id = filter1[id];
                    if (a1_count[hnsw_id] != 0)
                    {
                        std::priority_queue<std::pair<float, hnswlib::labeltype>> topResults = hnsw_array[hnsw_id]->searchKnnWithoutFilter(vectorQuerySQ16.data() + DIM * id, K,exp_thr2);
                        int kk = K < topResults.size() ? K : topResults.size();
                        int i = 0;
                        for (; i < kk; i++)
                        {
                            hnswlib::labeltype label = topResults.top().second;
                            topResults.pop();
                            neighbors[id * K + i] = label;
                        }
                        if(kk != K){
                            uint32_t s = 1;
                            while(i < K) {
                                neighbors[id * K + i] = vectorNum - s;
                                i++;
                                s = s + 1;
                            }
                        }
                    } });
    // timer
    auto s6 = high_resolution_clock::now();
    auto d5 = duration_cast<seconds>(s6 - s_swap);
    cout << "---Q2 " << d5.count() << " seconds ---" << endl;

    // q4__________________________________________________________________
    for (size_t i = 0; i < hnsw_num; i++)
    {
        hnsw_array[i]->setEf(ef4);
    }
    ParallelFor(0, q4tsv.size(), num_threads, [&](size_t row, size_t threadId)
                {
        int id = q4tsv[row];
        int hnsw_id = filter1[id];
        if (a1_count[hnsw_id] != 0){
                    auto start = lower_bound(sorted_by_all.begin(), sorted_by_all.end(), filter3[2 * id]);
                    auto end = upper_bound(sorted_by_all.begin(), sorted_by_all.end(), filter3[2 * id + 1]);
                    auto n1 = std::distance(sorted_by_all.begin(), start);
                    auto n2 = std::distance(sorted_by_all.begin(), end);
                    std::priority_queue<std::pair<float, size_t>> topResults;
                    auto dis = n2 - n1;

                    if (dis > bt_q4)
                    {
                        topResults = hnsw_array[hnsw_id]->searchKnnWithFilter(vectorQuerySQ16.data() + DIM * id, K, exp_thr4, id, &picks2);
                    }
                    else
                    {
                        int bt_first = K < dis ? K : dis;
                        for (int i = n1; i < n1 + bt_first && i < vectorNum; i++)
                        {
                            size_t real_id = id_sorted_by_all[i];
                            _mm_prefetch(BaseSortByALLSQ16.data() + DIM * i,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByALLSQ16.data() + DIM * i + 64,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByALLSQ16.data() + DIM * i + 128,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByALLSQ16.data() + DIM * i + 256,_MM_HINT_T0);
                            float dist = fstdistfunc_(vectorQuerySQ16.data() + DIM * id, BaseSortByALLSQ16.data() + DIM * i, dist_func_param_);
                            topResults.emplace(dist, real_id);
                        }
                        float lastdist = topResults.empty() ? std::numeric_limits<float>::max() : topResults.top().first;
                        for (int i = n1 + K; i < n1 + dis && i <vectorNum; i++)
                        {
                            size_t real_id = id_sorted_by_all[i];
                            _mm_prefetch(BaseSortByALLSQ16.data() + DIM * i,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByALLSQ16.data() + DIM * i + 64,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByALLSQ16.data() + DIM * i + 128,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByALLSQ16.data() + DIM * i + 256,_MM_HINT_T0);
                            float dist = fstdistfunc_(vectorQuerySQ16.data() + DIM * id, BaseSortByALLSQ16.data() + DIM * i, dist_func_param_);
                            if (dist <= lastdist)
                            {
                                topResults.emplace(dist, real_id);

                                if (topResults.size() > K)
                                    topResults.pop();

                                if (!topResults.empty())
                                {
                                    lastdist = topResults.top().first;
                                }
                            }
                        }
                   }
                    int kk = K < topResults.size() ? K : topResults.size();
                    int i=0;
                    for (; i < kk; i++) {
                        hnswlib::labeltype label = topResults.top().second;
                        topResults.pop();
                        neighbors[id * K + i] = label;
                    }
                    if(kk != K){
                        uint32_t s = 1;
                        while(i < K) {
                            
                            neighbors[id * K + i] = vectorNum - s;
                            i++;
                            s = s + 1;
                        }
                    }
        } });

    // timer
    auto s7 = high_resolution_clock::now();
    auto d6 = duration_cast<seconds>(s7 - s6);
    cout << "---Q4 " << d6.count() << " seconds ---" << endl;

    for (size_t i = 0; i < hnsw_num; i++)
    {
        delete hnsw_array[i];
    }

    // q1__________________________________________________________________
    alg_hnsw->setEf(ef1);
    ParallelFor(0, q1tsv.size(), num_threads, [&](size_t row, size_t threadId)
                {
        int id = q1tsv[row];
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnWithoutFilter(vectorQuerySQ16.data() + DIM * id, K,exp_thr1);
        int kk = K < result.size() ? K : result.size();
        int i=0;
        for (; i < kk; i++) {
            hnswlib::labeltype label = result.top().second;
            result.pop();
            neighbors[id * K + i] = label;
        }
        if(kk != K){
            uint32_t s = 1;
            while(i < K) {
                neighbors[id * K + i] = vectorNum - s;
                i++;
                s = s + 1;
            }
        } 
         });
    // timer
    auto s8 = high_resolution_clock::now();
    auto d7 = duration_cast<seconds>(s8 - s7);
    cout << "---Q1 " << d7.count() << " seconds ---" << endl;

    // q3__________________________________________________________________
    alg_hnsw->setEf(ef3);
    ParallelFor(0, q3tsv.size(), num_threads, [&](size_t row, size_t threadId)
                {
                    int id = q3tsv[row];
                    auto start = lower_bound(sorted_by_time.begin(), sorted_by_time.end(), filter2[2 * id]);
                    auto end = upper_bound(sorted_by_time.begin(), sorted_by_time.end(), filter2[2 * id + 1]);
                    auto n1 = std::distance(sorted_by_time.begin(), start);
                    auto n2 = std::distance(sorted_by_time.begin(), end);
                    std::priority_queue<std::pair<float, size_t>> topResults;
                    auto dis = n2 - n1;

                    if (dis > bt_q3)
                    {
                        topResults = alg_hnsw->searchKnnWithFilter(vectorQuerySQ16.data() + DIM * id, K,exp_thr3, id, &picks2);
                    }
                    else
                    {
                        int bt_first = K < dis ? K : dis;
                        for (int i = n1; i < n1 + bt_first && i < vectorNum; i++)
                        {
                            size_t real_id = id_sorted_by_time[i];
                            _mm_prefetch(BaseSortByTimeSQ16.data() + DIM * i,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByTimeSQ16.data() + DIM * i + 64,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByTimeSQ16.data() + DIM * i + 128,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByTimeSQ16.data() + DIM * i + 192,_MM_HINT_T0);                           
                            float dist = fstdistfunc_(vectorQuerySQ16.data() + DIM * id, BaseSortByTimeSQ16.data() + DIM * i, dist_func_param_);
                            // size_t label = i;
                            topResults.emplace(dist, real_id);
                        }
                        float lastdist = topResults.empty() ? std::numeric_limits<float>::max() : topResults.top().first;
                        for (int i = n1 + K; i < n1 + dis && i <vectorNum; i++)
                        {
                            size_t real_id = id_sorted_by_time[i];
                            _mm_prefetch(BaseSortByTimeSQ16.data() + DIM * i,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByTimeSQ16.data() + DIM * i + 64,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByTimeSQ16.data() + DIM * i + 128,_MM_HINT_T0);
                            _mm_prefetch(BaseSortByTimeSQ16.data() + DIM * i + 192,_MM_HINT_T0); 
                            float dist = fstdistfunc_(vectorQuerySQ16.data() + DIM * id, BaseSortByTimeSQ16.data() + DIM * i, dist_func_param_);
                            if (dist <= lastdist)
                            {
                                // size_t label = i;
                                topResults.emplace(dist, real_id);

                                if (topResults.size() > K)
                                    topResults.pop();

                                if (!topResults.empty())
                                {
                                    lastdist = topResults.top().first;
                                }
                            }
                        }
                   }
                    int kk = K < topResults.size() ? K : topResults.size();
                    int i=0;
                    for (; i < kk; i++)
                    {
                        hnswlib::labeltype label = topResults.top().second;
                        topResults.pop();
                        neighbors[id * K + i] = label;
                    }
                    if(kk != K){
                        uint32_t s = 1;
                        while(i < K) {
                            
                            neighbors[id * K + i] = vectorNum - s;
                            i++;
                            s = s + 1;
                        }
                    }  
                     });
    // time
    auto s9 = high_resolution_clock::now();
    auto d8 = duration_cast<seconds>(s9 - s8);
    cout << "---Q3 " << d8.count() << " seconds ---" << endl;
    delete alg_hnsw;

    // save knn
    vector<vector<uint32_t>> result_uint32(QUERY, vector<uint32_t>(K, 0));
    for (int i = 0; i < QUERY; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            result_uint32[i][j] = neighbors[i * K + j];
        }
    }

    SaveKNN(result_uint32);
    // timer
    auto s10 = high_resolution_clock::now();
    auto d9 = duration_cast<seconds>(s10 - s9);
    cout << "---Save KNN " << d9.count() << " seconds ---" << endl;

    // all time
    auto s11 = high_resolution_clock::now();
    auto d10 = duration_cast<seconds>(s11 - s1);
    cout << "---All " << d10.count() << " seconds ---" << endl;

    return 0;
}
