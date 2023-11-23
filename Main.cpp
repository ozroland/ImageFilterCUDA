#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

extern "C" bool boxFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool laplacianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool medianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool sharpeningFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool sobelFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool tvFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);

int main(int argc, char** argv) {
    string image_location;
    string image_name;
    int filter_choice;

    cout << "Enter the location of the image: ";
    cin >> image_location;
    cout << "Enter the name of the image (with extension, e.g., image.jpg): ";
    cin >> image_name;

    string input_file = image_location + "/" + image_name;

    cv::Mat srcImage = cv::imread(input_file);
    if (srcImage.empty()) {
        cout << "Image Not Found: " << input_file << endl;
        return -1;
    }

    cout << "\nInput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

    cout << "Choose a filter: \n";
    cout << "1. Box filter\n";
    cout << "2. Laplacian filter\n";
    cout << "3. Median filter\n";
    cout << "4. Sharpening filter\n";
    cout << "5. Sobel edge detection filter\n";
    cout << "6. TV filter\n";
    cout << "Enter the filter number (1-6): ";
    cin >> filter_choice;

    cv::Mat dstImage(srcImage.size(), srcImage.type());

    switch (filter_choice) {
    case 1:
        boxFilter_GPU_wrapper(srcImage, dstImage);
        break;

    case 2:
        laplacianFilter_GPU_wrapper(srcImage, dstImage);
        break;

    case 3:
        medianFilter_GPU_wrapper(srcImage, dstImage);
        break;

    case 4:
        sharpeningFilter_GPU_wrapper(srcImage, dstImage);
        break;

    case 5:
        sobelFilter_GPU_wrapper(srcImage, dstImage);
        break;

    case 6:
        tvFilter_GPU_wrapper(srcImage, dstImage);
        break;

    default:
        cout << "Invalid choice\n";
        return -1;
    }

    string output_file = image_location + "/" + "output_" + image_name;

    imwrite(output_file, dstImage);

    return 0;
}