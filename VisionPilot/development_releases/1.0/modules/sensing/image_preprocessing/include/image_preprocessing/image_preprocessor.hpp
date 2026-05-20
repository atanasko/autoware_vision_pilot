#ifndef VISIONPILOT_IMAGE_PREPROCESSOR_HPP
#define VISIONPILOT_IMAGE_PREPROCESSOR_HPP
#include <string>
#include <opencv2/opencv.hpp>

class ImagePreprocessor {
public:
    ImagePreprocessor();

    ~ImagePreprocessor() = default;

    void preprocess(const cv::Mat &image, cv::Mat &warped_image, cv::Mat &resized_image, const cv::Size &size) const;

private:
    std::string homography_C_matrix_path;
    cv::Mat C;
};

#endif //VISIONPILOT_IMAGE_PREPROCESSOR_HPP
