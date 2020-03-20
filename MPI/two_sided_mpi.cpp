#include "./include/nn.h"
#include <mpi.h>


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
void AvgReduce_float(float* sendData, float* recvData, const int size, const int world_size);




int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int world_size;
	int world_rank;

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	// print hello world
	printf("hello world from processor %s, rank %d, out of %d processors\n",
		processor_name, world_rank, world_size);

    const int partitionSize = static_cast<int>(batch_size / world_size);


	/*if (world_rank == MAIN_NODE)
	{
		printf("I AM THE MAIN NODE\n");

        constexpr int train_size {60000};
        constexpr int test_size {10000};
    
    	//std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
    	//std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

    	char* train_images = new char[train_size * width * height * channels];
    	char* train_labels = new char[train_size];

    	char* test_images = new char[test_size * width * height * channels];
    	char* test_labels = new char[test_size];

		printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
    	printf("Batch size: %lld, iterations: %d\n", batch_size, iterations);
	}*/

     // Create the LeNet network architecture
    ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
    MaxPoolLayer pool1(2, 2);
    ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
    MaxPoolLayer pool2(2, 2);
    FullyConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 
                        500);
    FullyConnectedLayer fc2(fc1.outputs, 10);

    // network gradients init
    std::vector<float> local_h_gconv1(conv1.pconv.size());
    std::vector<float> local_h_gconvbias(conv1.pbias.size());
    std::vector<float> local_h_gconv2(conv2.pconv.size());
    std::vector<float> local_h_gconv2bias(conv2.pbias.size());
    std::vector<float> local_h_gfc1(fc1.pneurons.size());
    std::vector<float> local_h_gfc1bias(fc1.pbias.size());
    std::vector<float> local_h_gfc2(fc2.pneurons.size());
    std::vector<float> local_h_gfc2bias(fc2.pbias.size());

    // network gradients init
    std::vector<float> global_h_gconv1(conv1.pconv.size());
    std::vector<float> global_h_gconvbias(conv1.pbias.size());
    std::vector<float> global_h_gconv2(conv2.pconv.size());
    std::vector<float> global_h_gconv2bias(conv2.pbias.size());
    std::vector<float> global_h_gfc1(fc1.pneurons.size());
    std::vector<float> global_h_gfc1bias(fc1.pbias.size());
    std::vector<float> global_h_gfc2(fc2.pneurons.size());
    std::vector<float> global_h_gfc2bias(fc2.pbias.size());

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
    for(int iter = 0; iter < iterations; iter++)
    {   

        // wait compute time while gradients are being computed
        usleep(gpu_time_no_update_ms * 1000.0f);

        AvgReduce_float(local_h_gconv1.data(), global_h_gconv1.data(), conv1.pconv.size(), world_size);
        AvgReduce_float(local_h_gconvbias.data(), global_h_gconvbias.data(), conv1.pbias.size(), world_size);
        AvgReduce_float(local_h_gconv2.data(), global_h_gconv2.data(), conv2.pconv.size(), world_size);
        AvgReduce_float(local_h_gconv2bias.data(), global_h_gconv2bias.data(), conv2.pbias.size(), world_size);
        AvgReduce_float(local_h_gfc1.data(), global_h_gfc1.data(), fc1.pneurons.size(), world_size);
        AvgReduce_float(local_h_gfc1bias.data(), global_h_gfc1bias.data(), fc1.pbias.size(), world_size);
        AvgReduce_float(local_h_gfc2.data(), global_h_gfc2.data(), fc2.pneurons.size(), world_size);
        AvgReduce_float(local_h_gfc2bias.data(), global_h_gfc2bias.data(), fc2.pbias.size(), world_size);

        // now add the time it takes to update
         usleep(gpu_time_update_ms * 1000.0f);

        // sync 
        MPI_Barrier(MPI_COMM_WORLD);

    }
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("Total Average time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / ((int)iterations));

    delete[] h_train_labels;
    delete[] h_train_images;

	MPI_Finalize();
	return 0;
}


void AvgReduce_float(float* sendData, float* recvData, const int size, const int world_size)
{
    MPI_Allreduce(sendData, recvData, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    for(int i = 0; i < size; i++)
    {
        recvData[i] = recvData[i] / world_size;
    }
}

static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}
