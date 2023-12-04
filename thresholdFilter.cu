#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace std;

__global__ void thresholdFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channel;
        unsigned char pixel = 0;
        for (int c = 0; c < channel; ++c) {
            pixel += srcImage[idx + c] / channel;
        }

        if (pixel > 100) {
            for (int c = 0; c < channel; ++c) {
                dstImage[idx + c] = 255; 
            }
        }
        else {
            for (int c = 0; c < channel; ++c) {
                dstImage[idx + c] = 0;
            }
        }
    }
    
}


extern "C" void thresholdFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int channel = input.step / input.cols;

    const int inputSize = input.cols * input.rows * channel;
    const int outputSize = output.cols * output.rows * channel;
    unsigned char* d_input, * d_output;

    cudaMalloc<unsigned char>(&d_input, inputSize);
    cudaMalloc<unsigned char>(&d_output, outputSize);

    cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

    cudaEventRecord(start);

    thresholdFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);

    cudaEventRecord(stop);
    cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\nProcessing time for GPU (ms): " << milliseconds << "\n";
}