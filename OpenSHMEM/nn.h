#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <unistd.h>

#include <thread>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>



/**
 * Represents a convolutional layer with bias.
 */
struct ConvBiasLayer
{
    int in_channels, out_channels, kernel_size;
    int in_width, in_height, out_width, out_height;

    std::vector<float> pconv, pbias;
    
    ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_, 
                  int in_w_, int in_h_) : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_), 
                  pbias(out_channels_)
    {
        in_channels = in_channels_;
        out_channels = out_channels_;
        kernel_size = kernel_size_;
        in_width = in_w_;
        in_height = in_h_;
        out_width = in_w_ - kernel_size_ + 1;
        out_height = in_h_ - kernel_size_ + 1;
    }

    bool FromFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";
        
        // Read weights file
        FILE *fp = fopen(ssf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            return false;
        }
        fread(&pconv[0], sizeof(float), in_channels * out_channels * kernel_size * kernel_size, fp);
        fclose(fp);

        // Read bias file
        fp = fopen(ssbf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            return false;
        }
        fread(&pbias[0], sizeof(float), out_channels, fp);
        fclose(fp);
        return true;
    }

    void ToFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Write weights file
        FILE *fp = fopen(ssf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            exit(2);
        }
        fwrite(&pconv[0], sizeof(float), in_channels * out_channels * kernel_size * kernel_size, fp);
        fclose(fp);

        // Write bias file
        fp = fopen(ssbf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            exit(2);
        }
        fwrite(&pbias[0], sizeof(float), out_channels, fp);
        fclose(fp);
    }
};

/**
 * Represents a max-pooling layer.
 */
struct MaxPoolLayer
{
    int size, stride;
    MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};

/**
 * Represents a fully-connected neural network layer with bias.
 */
struct FullyConnectedLayer
{
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_) : outputs(outputs_), inputs(inputs_),
        pneurons(inputs_ * outputs_), pbias(outputs_) {}

    bool FromFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Read weights file
        FILE *fp = fopen(ssf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            return false;
        }
        fread(&pneurons[0], sizeof(float), inputs * outputs, fp);
        fclose(fp);

        // Read bias file
        fp = fopen(ssbf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            return false;
        }
        fread(&pbias[0], sizeof(float), outputs, fp);
        fclose(fp);
        return true;
    }

    void ToFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Write weights file
        FILE *fp = fopen(ssf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            exit(2);
        }
        fwrite(&pneurons[0], sizeof(float), inputs * outputs, fp);
        fclose(fp);

        // Write bias file
        fp = fopen(ssbf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            exit(2);
        }
        fwrite(&pbias[0], sizeof(float), outputs, fp);
        fclose(fp);
    }
};