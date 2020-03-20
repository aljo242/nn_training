#include "./nn.h"
#include <shmem.h>


constexpr int BW {128};
constexpr int gpu {0};
constexpr int iterations {1000};
constexpr int random_seed {1};
constexpr int classify {1};

constexpr int batch_size {64};

constexpr bool pretrained {false};
constexpr bool save_data {false};
const char* train_images {"train-images.idx3-ubyte"};
const char* test_images {"t10k-images.idx3-ubyte"};
const char* train_labels {"train-labels.idx1-ubyte"};
const char* test_labels {"t10k-labels.idx1-ubyte"};

constexpr double learning_rate {0.01};
constexpr double lr_gamma {0.0001};
constexpr double lr_power {0.75};

constexpr int MAIN_NODE = 0;

constexpr int channels {1};
constexpr int width {28};
constexpr int height {28};

constexpr double gpu_time_ms {1.180974};
constexpr double gpu_time_no_update_ms {1.144221};
constexpr double gpu_time_update_ms {gpu_time_ms - gpu_time_no_update_ms};


static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator);




int main(int argc, char** argv)
{
    shmem_init();
    int npes = shmem_n_pes();
    int me = shmem_my_pe();

    static long pSync[SHMEM_COLLECT_SYNC_SIZE];
    static float pWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
    for (int i = 0; i < SHMEM_COLLECT_SYNC_SIZE; i++)
    {
        pSync[i] = SHMEM_SYNC_VALUE;
    }

    printf("Hello from %d of %d\n", me, npes);  

    const int partitionSize = static_cast<int>(batch_size / npes);


     // Create the LeNet network architecture
    ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
    MaxPoolLayer pool1(2, 2);
    ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
    MaxPoolLayer pool2(2, 2);
    FullyConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 
                        500);
    FullyConnectedLayer fc2(fc1.outputs, 10);

    // network gradients init
    //std::vector<float> global_h_gconv1(conv1.pconv.size());
    //std::vector<float> global_h_gconvbias(conv1.pbias.size());
    //std::vector<float> global_h_gconv2(conv2.pconv.size());
    //std::vector<float> global_h_gconv2bias(conv2.pbias.size());
    //std::vector<float> global_h_gfc1(fc1.pneurons.size());
    //std::vector<float> global_h_gfc1bias(fc1.pbias.size());
    //std::vector<float> global_h_gfc2(fc2.pneurons.size());
    //std::vector<float> global_h_gfc2bias(fc2.pbias.size());

    float* global_h_gconv1 = (float*)shmem_malloc(conv1.pconv.size()*sizeof(float));
    float* global_h_gconvbias = (float*)shmem_malloc(conv1.pbias.size()*sizeof(float));
    float* global_h_gconv2 = (float*)shmem_malloc(conv2.pconv.size()*sizeof(float));
    float* global_h_gconv2bias = (float*)shmem_malloc(conv2.pbias.size()*sizeof(float));
    float* global_h_gfc1 = (float*)shmem_malloc(fc1.pneurons.size()*sizeof(float));
    float* global_h_gfc1bias = (float*)shmem_malloc(fc1.pbias.size()*sizeof(float));
    float* global_h_gfc2 = (float*)shmem_malloc(fc2.pneurons.size()*sizeof(float));
    float* global_h_gfc2bias = (float*)shmem_malloc(fc2.pbias.size()*sizeof(float));

    float* local_h_gconv1 = (float*)shmem_malloc(conv1.pconv.size()*sizeof(float));
    float* local_h_gconvbias = (float*)shmem_malloc(conv1.pbias.size()*sizeof(float));
    float* local_h_gconv2 = (float*)shmem_malloc(conv2.pconv.size()*sizeof(float));
    float* local_h_gconv2bias = (float*)shmem_malloc(conv2.pbias.size()*sizeof(float));
    float* local_h_gfc1 = (float*)shmem_malloc(fc1.pneurons.size()*sizeof(float));
    float* local_h_gfc1bias = (float*)shmem_malloc(fc1.pbias.size()*sizeof(float));
    float* local_h_gfc2 = (float*)shmem_malloc(fc2.pneurons.size()*sizeof(float));
    float* local_h_gfc2bias = (float*)shmem_malloc(fc2.pbias.size()*sizeof(float));

    
    char* h_train_images =           new char[width * height * channels * partitionSize];
    char* h_train_labels =           new char[partitionSize];

    {
        // Create random network
        unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
        std::random_device rd;
        std::mt19937 gen(seed1);

        // Xavier weight filling
        float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
        std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
        float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
        std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
        float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
        std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
        float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
        std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

        // Randomize network
        for (auto&& iter : conv1.pconv)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv1.pbias)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv2.pconv)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : conv2.pbias)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : fc1.pneurons)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc1.pbias)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc2.pneurons)
            iter = static_cast<float>(dfc2(gen));
        for (auto&& iter : fc2.pbias)
            iter = static_cast<float>(dfc2(gen));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int iter = 0; iter < 2; iter++)
    {   

        // wait compute time while gradients are being computed
        usleep(gpu_time_no_update_ms * 1000.0f);

        {
            shmem_float_sum_to_all(local_h_gconv1, global_h_gconv1, conv1.pconv.size(), 0, 0, npes, pWrk, pSync);
            shmem_float_sum_to_all(local_h_gconvbias, global_h_gconvbias, conv1.pbias.size(), 0, 0, npes, pWrk, pSync);
            shmem_float_sum_to_all(local_h_gconv2, global_h_gconv2, conv2.pconv.size(), 0, 0, npes, pWrk, pSync);
            shmem_float_sum_to_all(local_h_gconv2bias, global_h_gconv2bias, conv2.pbias.size(), 0, 0, npes, pWrk, pSync);


        }

        // now add the time it takes to update
        usleep(gpu_time_update_ms * 1000.0f);

        // sync 
        shmem_barrier_all();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("Total Average time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / (int)iterations);

    delete[] h_train_labels;
    delete[] h_train_images;

    shmem_free(global_h_gconv1);
    shmem_free(global_h_gconvbias);
    shmem_free(global_h_gconv2);
    shmem_free(global_h_gconv2bias);
    shmem_free(global_h_gfc1);
    shmem_free(global_h_gfc1bias);    
    shmem_free(global_h_gfc2);
    shmem_free(global_h_gfc2bias);

    shmem_free(local_h_gconv1);
    shmem_free(local_h_gconvbias);
    shmem_free(local_h_gconv2);
    shmem_free(local_h_gconv2bias);
    shmem_free(local_h_gfc1);
    shmem_free(local_h_gfc1bias);    
    shmem_free(local_h_gfc2);
    shmem_free(local_h_gfc2bias);


    shmem_finalize();
	return 0;
}




static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}
